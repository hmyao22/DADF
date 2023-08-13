import torch.optim as opti
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import init
import os
from dataset import TrainData, TestData
from constants import DefaultConfig
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import Models.ViT as entire_networks
from Models.misc import NativeScalerWithGradNormCount as NativeScaler
from Models.utils import adjust_learning_rate
import constants as const

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight)


def Rec_train(opt, show_feature_map=True):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, opt.class_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    loss_scaler = NativeScaler()
    device = opt.device
    print(device)

    model_name = opt.backbone_name + '_' + opt.class_name + '_' + 'LightCC.pth'
    Cycle_TR = entire_networks.CC_Model(opt)

    Cycle_TR.Feature_extractor = Cycle_TR.Feature_extractor.eval()
    Cycle_TR.Roncon_model = Cycle_TR.Roncon_model.train(True)

    if opt.use_gpu:
        Cycle_TR.to(device)
    if os.path.exists(os.path.join(opt.load_model_path,opt.class_name,model_name)):
        Cycle_TR.Roncon_model.load_state_dict(torch.load(os.path.join(opt.load_model_path,opt.class_name,model_name)))
        print("load weights!")

    optimizer = opti.AdamW(Cycle_TR.Roncon_model.parameters(), lr=opt.lr, betas=(0.9, 0.95))

    trainDataset = TrainData(opt=opt)
    train_dataloader = DataLoader(trainDataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True,
                                  num_workers=2, pin_memory=True)

    testDataset = TestData(opt=opt)
    test_dataloader = DataLoader(testDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    for epoch in range(opt.rec_epoch):
        adjust_learning_rate(optimizer, epoch)
        running_loss = 0.0

        for index, item in enumerate(tqdm(train_dataloader, ncols=80)):
            input_frame = item

            if opt.use_gpu:
                input_frame = input_frame.to(device, non_blocking=True)

            loss = Cycle_TR.loss(input_frame)

            loss_scaler(loss, optimizer, parameters=Cycle_TR.Roncon_model.parameters(),
                        update_grad=(index + 1) % 1 == 0)
            running_loss += loss.item()

            if index == len(train_dataloader) - 1:
                print(f"[{epoch}]  F_loss: {(running_loss / (1 * len(trainDataset))):.3f}")

        if epoch % 1 == 0:
            if epoch == 0:
                model_dict = Cycle_TR.Roncon_model.state_dict()
                torch.save(model_dict, os.path.join(opt.load_model_path,opt.class_name,model_name))
                Cycle_TR.eval()
            item = next(iter(test_dataloader))
            input_frame = item

            if opt.use_gpu:
                input_frame = input_frame.to(device, non_blocking=True)

            deep_feature, x_local, x_global = Cycle_TR(input_frame)

            if show_feature_map == True:
                plt.figure()
                plt.subplot(331)
                plt.imshow(deep_feature[0].cpu().detach().numpy()[0, 2, :, :])
                plt.subplot(332)
                plt.imshow(deep_feature[1].cpu().detach().numpy()[0, 2, :, :])
                plt.subplot(333)
                plt.imshow(deep_feature[2].cpu().detach().numpy()[0, 2, :, :])

                plt.subplot(334)
                plt.imshow(x_local[0].cpu().detach().numpy()[0, 2, :, :])
                plt.subplot(335)
                plt.imshow(x_local[1].cpu().detach().numpy()[0, 2, :, :])
                plt.subplot(336)
                plt.imshow(x_local[2].cpu().detach().numpy()[0, 2, :, :])

                plt.subplot(337)
                plt.imshow(x_global[0].cpu().detach().numpy()[0, 2, :, :])
                plt.subplot(338)
                plt.imshow(x_global[1].cpu().detach().numpy()[0, 2, :, :])
                plt.subplot(339)
                plt.imshow(x_global[2].cpu().detach().numpy()[0, 2, :, :])

                plt.savefig('feature/' + str(epoch) + '.png')
                plt.close()

            model_dict = Cycle_TR.Roncon_model.state_dict()
            torch.save(model_dict, os.path.join(opt.load_model_path,opt.class_name,model_name))
            Cycle_TR.Roncon_model = Cycle_TR.Roncon_model.train(True)


total_list_0 = [
    'wood',
]

cifar = [
    'horse',
    'ship',
    'airplane',
    'truck',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog'
]

MNIST = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
]

total_list = [
    'facemask',
]

video = ['ped2', 'avenue']
medical = ['OCT', 'brains']
opt = DefaultConfig()


backbone_name = ['WResnet50', 'Resnet34']

if __name__ == '__main__':
    opt.parse({'backbone_name': backbone_name[1]})
    for obj in total_list_0:
        opt.parse({'class_name': obj})
        print('training_dataset:' + str(opt.class_name))
        Rec_train(opt)

#
