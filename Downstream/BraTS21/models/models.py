import os

import torch
from monai.networks.nets import SwinUNETR


def VoCo(args, pretrained_path='VoCo.pt'):
    # CVPR 2024 extention
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=True
    )
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using VoCo pretrained backbone weights !!!!!!!")
    return model


def SuPrem(args, pretrained_path='SuPreM.pth'):
    # ICLR 2024
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=False
    )
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using Suprem-ICLR24 pretrained backbone weights !!!!!!!")
    return model


def Swin(args, pretrained_path='SwinUNETR.pt'):
    # CVPR 2023
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=False
    )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False)
    model = load(model, model_dict)
    print("Using Swin-CVPR23 pretrained backbone weights !!!!!!!")
    return model

def RaSE_B(args, pretrained_path='RaSE_B.pth'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_v2=True
        )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using RaSE_B pretrained backbone weights !!!!!!!")
    return model

def Anatomix(args, pretrained_path='anatomix.pth'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_v2=True
        )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using Anatomix pretrained backbone weights !!!!!!!")
    return model

def RaSE_L(args, pretrained_path='RaSE_L.pth'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_v2=True
        )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using RaSE_L pretrained backbone weights !!!!!!!")
    return model


def RaSE_H(args, pretrained_path='RaSE_H.pth'):
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_v2=True
        )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using RaSE_H pretrained backbone weights !!!!!!!")
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


