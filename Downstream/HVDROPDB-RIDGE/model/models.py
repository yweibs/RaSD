import os

import torch
from monai.networks.nets import UNet
import segmentation_models_pytorch as smp
import torch.nn as nn


def MedSAM(args, pretrained_path='medsam_vit_b.pth'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    from TransUNet.vit_seg_modeling import VisionTransformer
    model = VisionTransformer(img_size=(args.roi_x, args.roi_y), checkpoint_path=pretrained_path, pretrained=False)
    print("Using MedSAM TransUNet structure and weights !!!!!!!")
    return model

def LVM_R50(args, pretrained_path='lvmmed_resnet.torch'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = smp.Unet(encoder_name="resnet50", 
                    encoder_weights=None,
                    in_channels=args.in_channels, 
                    classes=args.out_channels)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using LVM_R50 pretrained backbone weights !!!!!!!")
    return model

def LVM_Vit(args, pretrained_path='lvmmed_vit.pth'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    from TransUNet.vit_seg_modeling import VisionTransformer
    model = VisionTransformer(img_size=(args.roi_x, args.roi_y), checkpoint_path=pretrained_path, pretrained=False)
    print("Using LVM_Vit TransUNet structure and weights !!!!!!!")
    return model


def RaSE_2D(args, pretrained_path='RaSE_2D.pth'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = UNet(
            spatial_dims=2,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=[64, 128, 256, 512, 1024],
            strides=[2, 2, 2, 2],
            num_res_units=4,
            act='relu',
            dropout=0.15,
            norm='batch'
        )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using RaSE_2D pretrained backbone weights !!!!!!!")
    return model


def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    elif "student" in model_dict.keys():
        state_dict = model_dict["student"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        print("Tag 'backbone.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()

    # for k in current_model_dict.keys():
    #     if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
    #         print(k)

    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VoCo models")
    parser.add_argument("--pretrained_root", default='/home/linshan/pretrained/', type=str, help="pretrained_root")
    parser.add_argument("--pretrained_path", default='model_B.pt', help="checkpoint name for voco")

    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")

    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

    args = parser.parse_args()
    model = VoCo(args)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    input = torch.rand(1, 1, 96, 96, 96)
    output = model(input)
    print(output.shape)

    from thop import profile
    import torch
    import torchvision.models as models

    flops, params = profile(model, inputs=(input,))
    gflops = flops / 1e9
    print(f"GFLOPS: {gflops}")