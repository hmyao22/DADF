import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants as const
from Models.ViT import VIT
import os
from Models.backbone_networks import *
import time

# def subnet_conv_func(kernel_size, hidden_ratio):
#     def subnet_conv(in_channels, out_channels):
#         hidden_channels = int(in_channels * hidden_ratio)
#         return nn.Sequential(
#             nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
#         )
#
#     return subnet_conv
#


# def mahalanobis_torch(embedding_vectors, mean, sigma):
#     B, C, H, W = embedding_vectors.size()
#     embedding_vectors = embedding_vectors.view(B, C, H * W)
#     embedding_vectors = embedding_vectors.contiguous().transpose(1, 2)
#     embedding_vectors = embedding_vectors.contiguous().view(B * H * W, C)
#     sigma = 1/(sigma+1e-5)
#     delta = mean - embedding_vectors
#     m = torch.matmul(torch.matmul(delta, sigma), delta.T)
#     output = torch.diagonal(m).view(B, H, W)
#
#     return torch.sqrt(output)



def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding="same", groups=in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes



class FastFlow_v2(nn.Module):
    def __init__(
        self,
        backbone_name,
        class_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1,
    ):
        super(FastFlow_v2, self).__init__()
        scales = [4, 8, 16]
        assert (backbone_name in const.SUPPORTED_BACKBONES), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)
        if backbone_name == 'VGG16':
            self.feature_extractor = VGG16()
            channels = [128, 256, 512]
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights', class_name, backbone_name+'_'+class_name+'_CC.pth')))
        if backbone_name == 'VGG19':
            self.feature_extractor = VGG19()
            channels = [128, 256, 512]
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights', class_name, backbone_name+'_'+class_name+'_CC.pth')))
        if backbone_name == 'MobileNet':
            self.feature_extractor = MobileNet()
            channels = [24, 40, 80]
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights', class_name, backbone_name+'_'+class_name+'_CC.pth')))
        if backbone_name == 'Resnet18':
            self.feature_extractor = Resnet18()
            channels = [64, 128, 256]
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights', class_name, backbone_name+'_'+class_name+'_CC.pth')))
        if backbone_name == 'Resnet34':
            self.feature_extractor = Resnet34()
            channels = [64, 128, 256]
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights', class_name, backbone_name+'_'+class_name+'_CC.pth')))
        if backbone_name == 'Resnet50':
            self.feature_extractor = Resnet50()
            channels = [256, 512, 1024]
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights', class_name, backbone_name+'_'+class_name+'_CC.pth')))
        if backbone_name == 'Resnet101':
            self.feature_extractor = Resnet101()
            channels = [256, 512, 1024]
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights',class_name, backbone_name + '_' + class_name + '_CC.pth')))
        if backbone_name == 'WResnet50':
            channels = [256, 512, 1024]
            self.feature_extractor = WResnet50()
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights',class_name,backbone_name+'_'+class_name+'_CC.pth')))
        if backbone_name == 'WResnet101':
            channels = [256, 512, 1024]
            self.feature_extractor = Resnet101()
            self.reconstructor = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=channels)
            self.reconstructor.load_state_dict(torch.load(os.path.join(f'weights', class_name, backbone_name+'_'+class_name+'_LightCC.pth')))



        self.norms = nn.ModuleList()


        for in_channels, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm([in_channels*3, int(input_size / scale), int(input_size / scale)],
                                 elementwise_affine=True,)
            )


        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.reconstructor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels*3, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )

        self.input_size = input_size

    def forward(self, x):
        self.feature_extractor.eval()
        self.reconstructor.eval()


        features = self.feature_extractor(x)

        patch_features, semantics_features = self.reconstructor.forward_(features)
        total_features = []


        total_features.append(torch.cat([features[0], patch_features[0], semantics_features[0]], dim=1))
        total_features.append(torch.cat([features[1], patch_features[1], semantics_features[1]], dim=1))
        total_features.append(torch.cat([features[2], patch_features[2], semantics_features[2]], dim=1))

        # total_features.append(torch.cat([features[0]], dim=1))
        # total_features.append(torch.cat([features[1]], dim=1))
        # total_features.append(torch.cat([features[2]], dim=1))

        total_features = [self.norms[i](feature) for i, feature in enumerate(total_features)]

        loss = 0

        outputs = []
        for i, feature, in enumerate(total_features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(0.5 * torch.sum(output ** 2, dim=(1, 2, 3)) - log_jac_dets)
            outputs.append(output)


        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []

            for output in outputs:
                log_prob = 1-torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(-prob, size=[self.input_size, self.input_size], mode="bilinear", align_corners=False,)
                anomaly_map_list.append(a_map)

            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map

        return ret


if __name__ =="__main__":
    model = Resnet34()
    tensor = torch.rand(1, 3, 256, 256)
    outpt = model(tensor)

