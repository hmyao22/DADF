import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import yaml
from ignite.contrib import metrics
import constants as const
import dataset
import fastflow
import utils
from reconstruction_train import Rec_train
from reconstruction_dataset import denormalize
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import compute_pro
from get_spro import Get_SPRO_Fun
from sklearn.metrics import roc_curve, auc
from scipy import interp


def build_train_data_loader(Config):
    train_dataset = dataset.MVTecDataset(
        root=Config.data_root,
        category=Config.class_name,
        input_size=256,
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.FLOW_batch,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(Config):
    test_dataset = dataset.MVTecLOCODataset(
        root=Config.data_root,
        category=Config.class_name,
        input_size=256,
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.FLOW_batch,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(Config):
    model = fastflow.FastFlow_v2(
        backbone_name=Config.backbone_name,
        class_name=Config.class_name,
        flow_steps=8,
        input_size=256,
        conv3x3_only=False,
        hidden_ratio=1)
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model, Config):
    return torch.optim.Adam(
        model.parameters(), lr=Config.Flow_LR, weight_decay=Config.WEIGHT_DECAY)


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()

    for step, data in enumerate(dataloader):
        # forward
        data = data.to(Config.device)
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % Config.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model):
    model.eval()
    pixel_auroc_metric = metrics.ROC_AUC()
    image_auroc_metric = metrics.ROC_AUC()


    i=0
    for data, targets in tqdm(dataloader):
        data, targets = data.to(Config.device), targets.to(Config.device)
        with torch.no_grad():
            ret = model(data)

        image_label = [1 if np.array(target.cpu().detach()).any() else 0 for target in targets]
        image_label = torch.from_numpy(np.array(image_label))

        outputs = ret["anomaly_map"].cpu().detach()

        image_score = [output.max() for output in outputs]
        image_score = torch.from_numpy(np.array(image_score))


        if dataloader.batch_size == 1:
            anomaly_map = outputs.squeeze(0).squeeze(0)

            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            image = denormalize(data.clone().squeeze(0).cpu().detach().numpy())

            plt.figure()
            plt.subplot(131)
            plt.axis('off')
            plt.imshow(image)
            plt.subplot(132)
            plt.axis('off')
            plt.imshow(anomaly_map, cmap='jet')
            plt.title(str(round(np.array(image_score)[0],2)))
            plt.subplot(133)
            plt.axis('off')
            plt.imshow(targets.cpu().detach().squeeze(0).squeeze(0), cmap='jet')
            plt.savefig('Result/'+str(i)+'.png')
            plt.close()

        ################### pro and sPRO ################
        # anomaly_map_gt = targets.cpu().detach().squeeze(0).numpy() if i == 0 \
        #     else np.concatenate((anomaly_map_gt, targets.cpu().detach().squeeze(0).numpy()), axis=0)
        #
        # anomaly_map_pre = outputs.squeeze(0).cpu().detach().numpy() if i == 0 \
        #     else np.concatenate((anomaly_map_pre, outputs.squeeze(0).cpu().detach().numpy()), axis=0)
        # i = i + 1




        outputs = outputs.flatten()
        targets = targets.flatten()

        pixel_auroc_metric.update((outputs, targets))
        image_auroc_metric.update((image_score, image_label))




    pixel_auroc_metric = pixel_auroc_metric.compute()
    image_auroc_metric = image_auroc_metric.compute()


    # anomaly_map_gt = np.squeeze(anomaly_map_gt)
    # anomaly_map_pre = np.squeeze(anomaly_map_pre)
    # anomaly_map_gt[anomaly_map_gt > 0.5] = 1
    # anomaly_map_gt[anomaly_map_gt <= 0.5] = 0
    # aupro_pixel = compute_pro(anomaly_map_gt, anomaly_map_pre)
    # print("==============================")
    # print("overall_pixel_AUPRO_metric: {}".format(aupro_pixel))
    # print(np.squeeze(anomaly_map_pre).shape)
    # Get_SPRO_Fun(np.squeeze(anomaly_map_pre), data_type=Config.class_name)



    print("==============================")
    print("overall_pixel_auroc_metric: {}".format(pixel_auroc_metric))
    print("overall_image_auroc_metric: {}".format(image_auroc_metric))

    result = {}
    result['pixel_auroc_metric'] = pixel_auroc_metric
    result['image_auroc_metric'] = image_auroc_metric
    result['pixel_aupro_metric'] = 0
    return result


def train(Config):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, Config.class_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = build_model(Config)
    optimizer = build_optimizer(model, Config)

    train_dataloader = build_train_data_loader(Config)
    test_dataloader = build_test_data_loader(Config)
    model.to(Config.device)

    max_auc = 0

    for epoch in range(Config.flow_epoch):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % Config.EVAL_INTERVAL == 0:
            aucroc = eval_once(test_dataloader, model)
            ave_aucroc = (aucroc['pixel_auroc_metric']+aucroc['image_auroc_metric'])/2.
            if ave_aucroc > max_auc:
                torch.save({"model_norms_state_dict": model.norms.state_dict(),
                            "model_nf_flows": model.nf_flows.state_dict()},
                           os.path.join(checkpoint_dir, Config.backbone_name+"_fuseflow.pt"),)
                max_auc = ave_aucroc


def evaluate(Config):
    model = build_model(Config)
    Config.parse({'FLOW_batch': 1})
    checkpoint = torch.load(os.path.join(os.path.join(const.CHECKPOINT_DIR, Config.class_name), Config.backbone_name+"_lightflow.pt"))
    model.nf_flows.load_state_dict(checkpoint["model_nf_flows"])
    model.norms.load_state_dict(checkpoint["model_norms_state_dict"])
    test_dataloader = build_test_data_loader(Config)
    model.to(Config.device)
    result = eval_once(test_dataloader, model)
    pixel_auc = result['pixel_auroc_metric']
    image_auc = result['image_auroc_metric']
    pixel_pro = result['pixel_aupro_metric']
    return pixel_auc, image_auc, pixel_pro


if __name__ == "__main__":

    texture = [
              "tile",
        "wood",
        "carpet",
        "grid",
        "leather",
               ]

    mvtec_loco1 = [
        "pushpins",
        "breakfast_box",
        "splicing_connectors",
        "screw_bag",
        "juice_bottle",
    ]

    mvtec_loco2 = [
        "pushpins",
    ]



    mvtec1 = [
        "bottle",
        "capsule",
        "cable",
        "metal_nut",
        "hazelnut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
    ]

    mvtec2 = [
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
    ]

    CIFAR1 = [
        'horse',
        'frog',
        'dog',
        'deer',
        'cat',

    ]
    CIFAR2 = [
    'bird',
    'automobile',
    'airplane',
    'truck',
    'ship',]

    MNIST1=[
        '0',
        '1',
        '2',
        '3',
        '4',
    ]
    MNIST2 = [
        '5',
        '6',
        '7',
        '8',
        '9',
    ]
    real=['PCB']

    Config = const.DefaultConfig()
    back = ['VGG16', 'VGG19', 'Resnet18', 'Resnet34', 'WResnet50']
    backbone_name = back[-1]
    Config.parse({'backbone_name': backbone_name})
    Config.parse({'device': "cuda:1"})


    Train_phase = True
    pixel_pro = 0
    pixel_auc = 0
    image_auc = 0

    for mvtec_class in reversed(mvtec_loco1):
        print('class:  '+mvtec_class)
        Config.parse_model_root({'class_name': mvtec_class})
        if Train_phase:
            # print('=============Reconstruction Training=============')
            # Rec_train(Config)
            print('=============Flow Training=============')
            train(Config)
        else:
            result = evaluate(Config)
            pixel_auc = pixel_auc + result[0]
            image_auc = image_auc + result[1]
            pixel_pro = pixel_pro + result[2]


    print(pixel_auc / 10)
    print(image_auc / 10)
    print(pixel_pro / 10)




