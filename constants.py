CHECKPOINT_DIR = "weights"

MVTEC_CATEGORIES = [
    "tile",
    "wood",
    "bottle",
    "capsule",
    "cable",
    "metal_nut",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]


SUPPORTED_BACKBONES = [
    'EfficientNet',
    'MobileNet',
    'VGG16',
    'VGG19',
    'Resnet18',
    'Resnet34',
    'Resnet50',
    'Resnet101',
    'WResnet50',
    'WResnet101'
]




import os
import torch


class DefaultConfig(object):
    backbone_name = 'Resnet18'
    class_name = 'tile'
    data_root = r'D:\IMSN-YHM\dataset\mvtec_loco_anomaly_detection'
    train_raw_data_root = os.path.join(data_root, class_name, 'train')
    test_raw_data_root = os.path.join(data_root, class_name, 'test')
    load_model_path = r'./weights/'
    use_gpu = True
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    LOG_INTERVAL = 10
    EVAL_INTERVAL = 1
    train_batch_size = 8
    FLOW_batch = 16
    rec_epoch = 200
    flow_epoch = 15
    Flow_LR= 1E-3
    WEIGHT_DECAY = 1e-5
    lr = 0.0001
    lr_decay = 0.90

    def parse(self, dicts):
        for k, v in dicts.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def parse_model_root(self, dicts):
        for k, v in dicts.items():
            if hasattr(self, k):
                setattr(self, k, v)
                data_root = r'D:\IMSN-YHM\dataset\mvtec_loco_anomaly_detection'
                setattr(self, 'train_raw_data_root', os.path.join(data_root, v, 'train'))
                setattr(self, 'test_raw_data_root', os.path.join(data_root, v, 'test'))



