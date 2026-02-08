# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
# from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
import sys
sys.path.append('/home/jiayi/Baseline/Segment/Ours/models')
# from image_encoder import load_image_encoder
from functools import partial
from timm import create_model
import time

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def resize_pos_embed(posemb, new_img_size, patch_size=16):
    """
    posemb: (1, H, W, C) 或 (1, N, C)
    """
    # 如果已经是 [1, H, W, C] 形式，直接返回，不做插值
    if posemb.dim() == 4:
        print("posemb.shape:", posemb.shape)
        print("Already 4D, returning as-is")
        return posemb
    
    # 如果是 [1, N, C]，进行插值
    num_patches_new = (new_img_size // patch_size) ** 2
    num_patches_old = posemb.shape[1]

    print("posemb.shape:", posemb.shape)
    print("num_patches_old:", num_patches_old, "num_patches_new:", num_patches_new)

    if num_patches_new == num_patches_old:
        return posemb
    
    dim = posemb.shape[-1]
    gs_old = int(num_patches_old ** 0.5)
    gs_new = int(num_patches_new ** 0.5)
    posemb = posemb.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)
    posemb = F.interpolate(
        posemb, size=(gs_new, gs_new),
        mode='bicubic', align_corners=False
    )
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return posemb


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        # self.config = config
        img_size = _pair(img_size)
        # grid_size=(16, 16)

        # if grid_size is not None:   # ResNet
        # grid_size = grid_size
        # print(img_size)
        # patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        
        # patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        # n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        patch_size=(1,1)
        n_patches=((img_size[0] // 16)*(img_size[1] // 16))
        self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #     self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=768,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class ViTBackbone(torch.nn.Module):
    def __init__(self, pre_trained=True,model_name='vit_base_patch16_224'):
        super(ViTBackbone, self).__init__()
        self.model = create_model(model_name, pretrained=pre_trained)

    def forward(self, x):
        # batch_size, _, height, width = x.shape
        # x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # features = self.model(x)
        # features = features[-1]  # Get the last feature map
        for block in self.model.blocks:
            x = block(x)
        # Resize features back to original size
        # features = torch.nn.functional.interpolate(features, size=(height, width), mode='bilinear', align_corners=False)
        return x
    
class Encoder(nn.Module):
    def __init__(self, vis,checkpoint_path,ours,finetune='lp'):
        super(Encoder, self).__init__()
        self.vis = vis
        if ours!=None:
            print(ours)
            self.backbone=load_image_encoder(ours)

        elif checkpoint_path!=None:
            if 'lvmmed' in checkpoint_path:
                from ref.lvmmed_vit import ImageEncoderViT
                prompt_embed_dim = 256
                image_size = 1024
                vit_patch_size = 16
                image_embedding_size = image_size // vit_patch_size
                encoder_embed_dim=768
                encoder_depth=12
                encoder_num_heads=12
                encoder_global_attn_indexes=[2, 5, 8, 11]

                self.backbone =ImageEncoderViT(
                        depth=encoder_depth,
                        embed_dim=encoder_embed_dim,
                        img_size=image_size,
                        mlp_ratio=4,
                        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                        num_heads=encoder_num_heads,
                        patch_size=vit_patch_size,
                        qkv_bias=True,
                        use_rel_pos=True,
                        use_abs_pos = False,
                        global_attn_indexes=encoder_global_attn_indexes,
                        window_size=14,
                        out_chans=prompt_embed_dim,
                    )
                
                check_point = torch.load(checkpoint_path)
                self.backbone.load_state_dict(check_point,strict=False)
                print('LVM-Med vit-b loaded')  
            elif 'medsam' in checkpoint_path:
                from ref.medsam_vit import ImageEncoderViT
                encoder_embed_dim=768
                encoder_depth=12
                encoder_num_heads=12
                encoder_global_attn_indexes=[2, 5, 8, 11]
                prompt_embed_dim = 256
                image_size = 1024
                vit_patch_size = 16
                image_embedding_size = image_size // vit_patch_size
                self.backbone=ImageEncoderViT(
                    depth=encoder_depth,
                    embed_dim=encoder_embed_dim,
                    img_size=image_size,
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=prompt_embed_dim,
                )
                check_point=torch.load(checkpoint_path)
                new_state_dict = {}
                for key, value in check_point.items():
                    new_key = key.replace('image_encoder.', '')
                    new_state_dict[new_key] = value
                if 'pos_embed' in new_state_dict:
                    new_state_dict['pos_embed'] = resize_pos_embed(
                        new_state_dict['pos_embed'], 
                        new_img_size=1024,  # 保持 MedSAM checkpoint 原始大小
                        patch_size=16      # ViT-B 默认 patch_size
                    )
                self.backbone.load_state_dict(new_state_dict, strict=False)
                print('MedSAM vit-b loaded')  
            elif 'mama' in checkpoint_path:
                from MaMA.load_weight import load_model
                self.backbone=load_model(checkpoint_path)
                print('mama vit-b loaded')
        # else:
        #     self.backbone = ViTBackbone(pre_trained=pretrained)
        # Freeze all parameters in the model
        # if finetune=='lp':
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

    def forward(self, hidden_states):
        # MedSAM ImageEncoderViT 会内部处理 patch embedding 和 pos embedding
        # 直接调用 backbone
        encoded = self.backbone(hidden_states)
        
        return encoded


class Transformer(nn.Module):
    def __init__(self, img_size, vis,checkpoint_path,pretrained,ours):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(vis=vis,checkpoint_path=checkpoint_path,ours=ours)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features



class ViTClassifier(nn.Module):
    def __init__(
        self,
        img_size=224,
        num_classes=2,
        checkpoint_path=None,
        pretrained=False,
        ours=None,
        finetune='lp',
        vis=False,
    ):
        super().__init__()
        
        # 直接用 Encoder（MedSAM），不要用 Embeddings 和 Transformer
        self.encoder = Encoder(vis=vis, checkpoint_path=checkpoint_path, ours=ours, finetune=finetune)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        if finetune == 'lp':
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x, target=None):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # MedSAM backbone 直接处理输入
        tokens = self.encoder(x)   # (B, N, 768)

        # 取平均 pooling
        feat = tokens.mean(dim=1)         # (B, 768)

        logits = self.head(feat)
        # if target is not None:
            # print('logits shape:', logits.shape, 'target shape:', target.shape)
        return logits

