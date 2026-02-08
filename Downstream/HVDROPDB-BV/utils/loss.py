import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.networks.utils import one_hot
import torch.nn.functional as F

class EfficientClDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.dice_loss = DiceLoss(to_onehot_y=False, include_background=True, sigmoid=True)
        self.dummy_param = nn.Parameter(torch.tensor(0.0))  # 添加一个参数，用于获取设备信息

    def compute_skeleton(self, x):
        """ GPU加速的中心线计算 """
        # 距离变换
        dist_map = self.distance_transform(x)  # 计算输入图像的距离变换，得到每个像素到最近边界的距离
        # 寻找局部极大值
        max_pool = F.max_pool2d(dist_map, kernel_size=3, stride=1, padding=1)
        skeleton = (dist_map > 0) & (dist_map == max_pool) # 确定中心线
        return skeleton.float()

    def distance_transform(self, x):
        """ 可微分的距离变换近似 """
        # 使用卷积近似距离变换
        x_vessel = x[:, 1, :, :].to(next(self.parameters()).device)  # 确保在同一设备上
        x_vessel = x_vessel.float()
        x_inv = 1 - x_vessel
        dt_foreground = F.conv2d(x_vessel.unsqueeze(1), self.gaussian_kernel(), padding=2)
        dt_background = F.conv2d(x_inv.unsqueeze(1), self.gaussian_kernel(), padding=2)
        return dt_foreground - dt_background

    def gaussian_kernel(self, size=5, sigma=1.0):
        """ 生成高斯卷积核 """
        coords = torch.arange(size).float() - size//2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g /= g.sum()
        return g.outer(g).view(1, 1, size, size).to(next(self.parameters()).device)

    def forward(self, pred, target):
        # 标准Dice损失作为基础
        dice_loss = self.dice_loss(pred, target)
        
        # 计算中心线
        target_center = self.compute_skeleton(target).to(pred.device)
        pred_center = self.compute_skeleton(torch.sigmoid(pred) > 0.5).to(pred.device)
        
        # print(target_center.shape)
        # print(pred_center.shape)
        # print(target.shape)
        # print(pred.shape)
        
        # 计算 clPrec 和 clRec
        intersection_prec = (pred_center * target).sum(dim=(1,2,3))
        clPrec = intersection_prec / (pred_center.sum(dim=(1,2,3)) + self.smooth)
        
        intersection_rec = (target_center * (torch.sigmoid(pred) > 0.5)).sum(dim=(1,2,3))
        clRec = intersection_rec / (target_center.sum(dim=(1,2,3)) + self.smooth)
        
        # 组合ClDice
        clDice = 1 - (2 * clPrec * clRec) / (clPrec + clRec + self.smooth)
        
        # 混合损失：50%标准Dice + 50%拓扑约束
        return 0.5 * dice_loss + 0.5 * clDice.mean()

class HybridVesselLoss(nn.Module):
    """ 视网膜血管分割专用损失 """
    def __init__(self, lambda_cl=0.7, num_classes=2):
        super().__init__()
        self.dice_ce = DiceCELoss(
            to_onehot_y=True,
            include_background=True,
            sigmoid=True,
            lambda_dice=0.5,
            lambda_ce=0.5
        )
        self.cldice = EfficientClDiceLoss()
        self.lambda_cl = lambda_cl
        self.num_classes = num_classes

    def forward(self, pred, target):
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        base_loss = self.dice_ce(pred, target)
        topo_loss = self.cldice(pred, target_one_hot)
        return (1 - self.lambda_cl) * base_loss + self.lambda_cl * topo_loss