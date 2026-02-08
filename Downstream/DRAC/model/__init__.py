from model.models import *


def get_model(args):
    if args.name == 'RaSE_B':
        return RaSE_B(args)
    elif args.name == 'RaSE_L':
        return RaSE_L(args)
    elif args.name == 'RaSE_H':
        return RaSE_H(args) 
    elif args.name == 'RaSE_2D':
        return RaSE_2D(args)
    elif args.name == 'RaSE_swin':
        return RaSE_swin(args)
    elif args.name == 'Scratch_swin':
        return Scratch_swin(args)
    elif args.name == 'scratch':
        return scratch(args)
    elif args.name == 'MedSAM':
        return MedSAM(args)
    elif args.name == 'LVM_R50':
        return LVM_R50(args)
    elif args.name == 'LVM_Vit':
        print("model found")
        return LVM_Vit(args)
    else:
        print('Without pre-training !')
        return scratch(args)