import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision import transforms as T

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.input_size = input_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.jpg")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).resize((self.input_size, self.input_size)).convert('RGB')
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("test", "ground_truth").replace(
                        ".png", "_mask.png"
                    )
                ).resize((self.input_size, self.input_size))
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)



class MVTecLOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).resize((256, 256)).convert('RGB')
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("test", "ground_truth").replace(
                        ".png", "/000.png"
                    )
                ).resize((256, 256))
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)



image_size=256
transform_x = T.Compose([T.Resize(image_size, InterpolationMode.BICUBIC),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])


def GetFiles(file_dir, file_type, IsCurrent=False):
    file_list = []
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            for type in file_type:
                if filename.endswith(('.%s' % type)):
                    file_list.append(os.path.join(parent, filename))

        if IsCurrent == True:
            break
    return file_list


class TrainData(data.Dataset):
    def __init__(self, opt):
        root = opt.train_raw_data_root
        print(root)
        imgs = GetFiles(root, ["jpg", "png", "jpeg"])
        self.imgs = [img for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        x = Image.open(img_path).convert('RGB').resize((image_size, image_size))
        img = transform_x(x)

        return img

    def __len__(self):
        return len(self.imgs)


class TestData(data.Dataset):
    def __init__(self, opt):
        root = opt.test_raw_data_root
        imgs = GetFiles(root, ["jpg", "png", "jpeg"])
        self.imgs = [img for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        x = Image.open(img_path).convert('RGB').resize((image_size, image_size))
        img = transform_x(x)
        return img

    def __len__(self):
        return len(self.imgs)

