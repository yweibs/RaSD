import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class Swin(nn.Module):
    def __init__(self, args):
        super(Swin, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=8 * args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 8,
            out_channels=args.feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 4,
            out_channels=args.feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 2,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.head = nn.Linear(args.feature_size, 3)  # 保持3维输出
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x_in, return_features=False, feature_layer='head'):
        b = x_in.size()[0]
        x_in = torch.cat([x_in, x_in], dim=2)
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        # Extract features
        features = F.adaptive_avg_pool3d(out, (1, 1, 1))
        features_flat = features.view(b, -1)
        
        if return_features:
            # 返回通过head层的3维特征
            return self.head(features_flat)
            
        output = self.head(features_flat)
        return output

class MultiPlaneMRNet(nn.Module):
    def __init__(self, backbone_model1, backbone_model2, backbone_model3):
        super().__init__()
        self.axial_backbone = backbone_model1
        self.coronal_backbone = backbone_model2
        self.sagittal_backbone = backbone_model3
        
        # 保持原始的9维输入
        self.fc1 = nn.Linear(9, 2)
        self.fc2 = nn.Linear(9, 2) 
        self.fc3 = nn.Linear(9, 2) 

    def forward(self, x_axial, x_coronal, x_sagittal, return_features=False):
        feat_axial = self.axial_backbone(x_axial, return_features=True)
        feat_coronal = self.coronal_backbone(x_coronal, return_features=True)
        feat_sagittal = self.sagittal_backbone(x_sagittal, return_features=True)
        
        combined = torch.cat((feat_axial, feat_coronal, feat_sagittal), dim=1)
        
        if return_features:
            return combined
            
        output1 = self.fc1(combined)
        output2 = self.fc2(combined)
        output3 = self.fc3(combined)
        
        return output1, output2, output3
