import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, wide_resnet50_2, wide_resnet101_2
from torchvision.models import mobilenet_v3_large,mobilenet_v3_small
from torchvision.models import mnasnet
from Models.efficientnet import model



class EfficientNet(nn.Module):

    def __init__(self):
        super(EfficientNet, self).__init__()
        efficient_net = model.EfficientNet.from_pretrained('efficientnet-b4')
        self.efficient_net = efficient_net.eval()
        modules = list(self.efficient_net.children())

    def forward(self, input_):
        with torch.no_grad():
            features = self.efficient_net.extract_features(input_)

        f2 = features[5]
        f3 = features[9]
        f4 = features[21]

        return [f2, f3, f4]


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        mobilenet = mobilenet_v3_large(True)
        layers = mobilenet.features
        self.layer1 = layers[:2]
        self.layer2 = layers[2:4]
        self.layer3 = layers[4:7]
        self.layer4 = layers[7:11]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return [out2, out3, out4]


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg = vgg16(True)
        layers = vgg.features
        print(layers)
        self.layer1 = layers[:5]
        self.layer2 = layers[5:10]
        self.layer3 = layers[10:17]
        self.layer4 = layers[17:24]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return [out2, out3, out4]



class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg = vgg19(True)
        layers = vgg.features
        self.layer1 = layers[:5]
        self.layer2 = layers[5:10]
        self.layer3 = layers[10:19]
        self.layer4 = layers[19:28]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return [out2, out3, out4]



class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet = resnet18(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]

    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)


        return [out2,out3,out4]



class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        resnet = resnet34(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]

    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)


        return [out2,out3,out4]


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        resnet = resnet50(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]

    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)


        return [out2,out3,out4]


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        resnet = resnet101(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]

    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        return [out2,out3,out4]


class WResnet50(nn.Module):
    def __init__(self):
        super(WResnet50, self).__init__()
        resnet = wide_resnet50_2(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]

    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)


        return [out2,out3,out4]


class WResnet101(nn.Module):
    def __init__(self):
        super(WResnet101, self).__init__()
        resnet = wide_resnet101_2(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]

    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        return [out2, out3, out4]


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    model = WResnet50().cuda()
    input_tensor = torch.rand(1, 3, 256, 256).cuda()
    output_tensor = model(input_tensor)
    import time

    for i in range(10):
        t1 = time.time()
        output_tensor = model(input_tensor)
        t2 = time.time()
        print(t2 - t1)
        print(output_tensor[0].shape)
        print(output_tensor[1].shape)
        print(output_tensor[2].shape)

    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops)
    print(params)
