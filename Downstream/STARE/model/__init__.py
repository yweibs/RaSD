from model.models import *


def get_model(args):
    if args.name == 'RaSE_2D':
        return RaSE_2D(args)
    elif args.name == 'MedSAM':
        return MedSAM(args)
    elif args.name == 'LVM_R50':
        return LVM_R50(args)
    elif args.name == 'LVM_Vit':
        return LVM_Vit(args)
    else:
        print('Without pre-training !')
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
        return model