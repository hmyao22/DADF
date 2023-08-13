# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from Models.dual_attention import Dual_Block
from Models.utils import get_2d_sincos_pos_embed
from Models.backbone_networks import *
from torch import nn



class VIT(nn.Module):

    def __init__(self, img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=[64, 128, 256],
                 embed_dim=480, depth=8, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.depth = depth

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed1 = PatchEmbed(img_size[0], patch_size[0], in_chans[0], embed_dim)
        self.patch_embed2 = PatchEmbed(img_size[1], patch_size[1], in_chans[1], embed_dim)
        self.patch_embed3 = PatchEmbed(img_size[2], patch_size[2], in_chans[2], embed_dim)

        num_patches = self.patch_embed1.num_patches
        self.sem_token = nn.Parameter(torch.zeros(1, num_patches, embed_dim*3))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim*3),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Dual_Block(embed_dim*3, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim*3)

        self.decoder_pred1 = nn.Linear(embed_dim*3, patch_size[0] ** 2 * in_chans[0], bias=True)
        self.decoder_pred2 = nn.Linear(embed_dim * 3, patch_size[1] ** 2 * in_chans[1], bias=True)
        self.decoder_pred3 = nn.Linear(embed_dim * 3, patch_size[2] ** 2 * in_chans[2], bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed1.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w1 = self.patch_embed1.proj.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.patch_embed2.proj.weight.data
        torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        w3 = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w3.view([w3.shape[0], -1]))

        torch.nn.init.normal_(self.sem_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x, index=0):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if index == 0:
            p = self.patch_embed1.patch_size[0]
            in_chans = self.in_chans[0]
        if index == 1:
            p = self.patch_embed2.patch_size[0]
            in_chans = self.in_chans[1]
        if index == 2:
            p = self.patch_embed3.patch_size[0]
            in_chans = self.in_chans[2]

        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, h * p))
        return imgs

    def patchify(self, imgs, index=0):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if index == 0:
            p = self.patch_embed1.patch_size[0]
            in_chans = self.in_chans[0]
        if index == 1:
            p = self.patch_embed2.patch_size[0]
            in_chans = self.in_chans[1]
        if index == 2:
            p = self.patch_embed3.patch_size[0]
            in_chans = self.in_chans[2]

        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_chans))
        return x

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.patch_embed1(x1)
        x2 = self.patch_embed2(x2)
        x3 = self.patch_embed3(x3)
        N, L, D = x1.shape  # batch, length, dim

        input_token = torch.cat([x1, x2, x3], dim=-1)
        sematics_token = self.sem_token.repeat(N, 1, 1)

        input_token = input_token + self.pos_embed[:, 1:, :]
        sematics_token = sematics_token + self.pos_embed[:, 1:, :]
        x_ = [input_token, sematics_token]

        for encoder_phase, blk in enumerate(self.blocks):
            x_ = blk(x_)
        x_local_latent, x_global_latent = x_
        x_local_latent = self.norm(x_local_latent)
        x_global_latent = self.norm(x_global_latent)

        ########## encoder-decoder ###########
        x_local1 = self.decoder_pred1(x_local_latent)
        x_global1 = self.decoder_pred1(x_global_latent)

        x_local2 = self.decoder_pred2(x_local_latent)
        x_global2 = self.decoder_pred2(x_global_latent)

        x_local3 = self.decoder_pred3(x_local_latent)
        x_global3 = self.decoder_pred3(x_global_latent)


        x_local = [x_local1, x_local2, x_local3]
        x_global = [x_global1, x_global2, x_global3]

        return x_local, x_global

    def loss(self, deep_features):
        patch_out, semantics_out = self(deep_features)

        target1 = self.patchify(deep_features[0], index=0)
        target2 = self.patchify(deep_features[1], index=1)
        target3 = self.patchify(deep_features[2], index=2)


        loss_semantics1 = ((semantics_out[0] - target1) ** 2).mean(dim=-1)
        loss_semantics2 = ((semantics_out[1] - target2) ** 2).mean(dim=-1)
        loss_semantics3 = ((semantics_out[2] - target3) ** 2).mean(dim=-1)



        loss_local1 = ((patch_out[0] - target1) ** 2).mean(dim=-1)
        loss_local2 = ((patch_out[1] - target2) ** 2).mean(dim=-1)
        loss_local3 = ((patch_out[2] - target3) ** 2).mean(dim=-1)


        return loss_semantics1.mean() + loss_semantics2.mean() + loss_semantics3.mean() \
               + loss_local1.mean() + loss_local2.mean() + loss_local3.mean()

    def a_map(self, deep_features):
        x_locals, x_globals = self.forward_(deep_features)

        def anomaly_map(x_local, x_global, deep_feature):
            batch_size = deep_feature.shape[0]
            global_map = torch.mean((deep_feature - x_global) ** 2, dim=1)
            global_map = global_map.reshape(batch_size, 1, deep_feature.shape[2], deep_feature.shape[2])
            global_map = nn.functional.interpolate(global_map, size=(256, 256), mode="bilinear",
                                                   align_corners=True).squeeze(1)
            global_map = global_map.clone().squeeze(0).cpu().detach().numpy()

            local_map = torch.mean((deep_feature - x_local) ** 2, dim=1)
            local_map = local_map.reshape(batch_size, 1, deep_feature.shape[2], deep_feature.shape[2])
            local_map = nn.functional.interpolate(local_map, size=(256, 256), mode="bilinear",
                                                  align_corners=True).squeeze(1)
            local_map = local_map.clone().squeeze(0).cpu().detach().numpy()
            return global_map, local_map

        global_map1, local_map1 = anomaly_map(x_locals[0], x_globals[0], deep_features[0])
        global_map2, local_map2 = anomaly_map(x_locals[1], x_globals[1], deep_features[1])
        global_map3, local_map3 = anomaly_map(x_locals[2], x_globals[2], deep_features[2])

        return local_map1+local_map2+local_map3, global_map1+global_map2+global_map3


    def forward_(self, x):
        patch_out, semantics_out = self(x)
        patch_out1, patch_out2, patch_out3 = patch_out
        semantics_out1, semantics_out2, semantics_out3 = semantics_out

        patch_out1 = self.unpatchify(patch_out1, index=0)
        patch_out2 = self.unpatchify(patch_out2, index=1)
        patch_out3 = self.unpatchify(patch_out3, index=2)

        semantics_out1 = self.unpatchify(semantics_out1, index=0)
        semantics_out2 = self.unpatchify(semantics_out2, index=1)
        semantics_out3 = self.unpatchify(semantics_out3, index=2)

        patch_out = [patch_out1, patch_out2, patch_out3]
        semantics_out = [semantics_out1, semantics_out2, semantics_out3]

        return patch_out, semantics_out


class CC_Model(nn.Module):
    def __init__(self, opt):
        super(CC_Model, self).__init__()

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = VIT(img_size=[64, 32, 16], patch_size=[4, 2, 1], in_chans=[32, 56, 160]).cuda()
        if opt.backbone_name == 'VGG16':
            self.Feature_extractor = VGG16().eval()
            self.Roncon_model = VIT(img_size=[64, 32, 16], patch_size=[4,2,1], in_chans=[128, 256, 512]).cuda()

        if opt.backbone_name == 'VGG19':
            self.Feature_extractor = VGG19().eval()
            self.Roncon_model = VIT(img_size=[64, 32, 16], patch_size=[4,2,1], in_chans=[128, 256, 512]).cuda()

        if opt.backbone_name == 'Resnet18':
            self.Feature_extractor = Resnet18().eval()
            self.Roncon_model = VIT(img_size=[64, 32, 16], patch_size=[4,2,1], in_chans=[64,128,256]).cuda()

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = VIT(img_size=[64, 32, 16], patch_size=[4,2,1], in_chans=[64,128,256]).cuda()

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = VIT(img_size=[64, 32, 16], patch_size=[4,2,1], in_chans=[64,128,256]).cuda()

        if opt.backbone_name == 'WResnet50':
            self.Feature_extractor = WResnet50().eval()
            self.Roncon_model = VIT(img_size=[64,32,16], patch_size=[4,2,1], in_chans=[256, 512, 1024]).cuda()

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = VIT(img_size=[64,32,16], patch_size=[4,2,1], in_chans=[64,128,256]).cuda()

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = VIT(img_size=[64,32,16], patch_size=[4, 2, 1], in_chans=[24, 40, 80]).cuda()

    def forward(self, imgs):
        deep_feature = self.Feature_extractor(imgs)
        x_local, x_global = self.Roncon_model.forward_(deep_feature)
        return deep_feature, x_local, x_global

    def loss(self, imgs):
        deep_features = self.Feature_extractor(imgs)
        return self.Roncon_model.loss(deep_features)

    def a_map(self, imgs):
        deep_features = self.Feature_extractor(imgs)
        return self.Roncon_model.a_map(deep_features)




if __name__ =='__main__':
    import time

    input_tensor1 = torch.rand(1, 64, 64, 64).cuda()
    input_tensor2 = torch.rand(1, 128, 32, 32).cuda()
    input_tensor3 = torch.rand(1, 256, 16, 16).cuda()
    input_tensor = [input_tensor1, input_tensor2, input_tensor3]
    model = VIT(img_size=[64,32,16], patch_size=[4,2,1], in_chans=[64,128,256]).cuda()

    for i in range(10):
        t1 = time.time()
        output = model.loss(input_tensor)
        t2= time.time()
        print(t2-t1)


