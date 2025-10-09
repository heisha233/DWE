import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.distributions.uniform import Uniform
import numpy as np
BatchNorm2d = nn.BatchNorm2d
relu_inplace = True

BN_MOMENTUM = 0.1
# BN_MOMENTUM = 0.01


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
    def forward(self, x):
        x = self.down(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn2(out) + identity
        out = self.relu(out)

        return out

class DoubleBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(DoubleBasicBlock, self).__init__()

        self.DBB = nn.Sequential(
            BasicBlock(inplanes=inplanes, planes=planes, downsample=downsample),
            BasicBlock(inplanes=planes, planes=planes)
        )
        self.res = nn.Sequential(
            conv1x1(inplanes,planes),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
    def forward(self, x):
        out = self.DBB(x)+self.res(x)
        return out

class ChannelFrequency(nn.Module):
    def __init__(self, in_planes):
        super(ChannelFrequency, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialFrequency(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialFrequency, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class QFAU(nn.Module):
    """Quadrant Frequency-Aware Unit (input: N, C, H, W)."""
    def __init__(self, in_channel):
        super(QFAU, self).__init__()
        self.CFM = ChannelFrequency(in_channel)
        self.SFE = SpatialFrequency()

    def _attn(self, quad):
        y = self.CFM(quad) * quad
        y = self.SFE(y) * y
        return y

    def forward(self, x):
        x_top, x_bottom = x.chunk(2, dim=2)
        x_tl, x_tr = x_top.chunk(2, dim=3)
        x_bl, x_br = x_bottom.chunk(2, dim=3)

        x_tl = self._attn(x_tl)
        x_tr = self._attn(x_tr)
        x_bl = self._attn(x_bl)
        x_br = self._attn(x_br)

        top = torch.cat((x_tl, x_tr), dim=3)
        bottom = torch.cat((x_bl, x_br), dim=3)
        x_local = torch.cat((top, bottom), dim=2)

        x_global = self.ca(x) * x
        x_global = self.sa(x_global) * x_global

        out = x_local + x_global
        return out

class DWE(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DWE, self).__init__()

        l1c, l2c, l3c, l4c = 64, 128, 256, 512

        self.xi = nn.Parameter(torch.tensor(0.0))
        self.eta = nn.Parameter(torch.tensor(0.0))

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_4_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_4_up = up_conv(l4c, l3c)
        self.unit1_1 = QFAU(l1c)
        self.unit1_2 = QFAU(l2c)
        self.unit1_3 = QFAU(l3c)
        self.unit1_4 = QFAU(l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_4_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_4_up = up_conv(l4c, l3c)
        self.unit2_1 = QFAU(l1c)
        self.unit2_2 = QFAU(l2c)
        self.unit2_3 = QFAU(l3c)
        self.unit2_4 = QFAU(l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):

        x1 = x1 + self.xi * x2
        x2 = x2 + self.eta * x1

        # branch1
        x1_1 = self.b1_1_1(x1)
        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)
        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)
        x1_4_1 = self.b1_3_2_down(x1_3)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_1 = self.unit1_1(x1_1)
        x1_2 = self.unit1_2(x1_2)
        x1_3 = self.unit1_3(x1_3)
        x1_4_1 = self.unit1_4(x1_4_1)

        # branch2
        x2_1 = self.b2_1_1(x2)
        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)
        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)
        x2_4_1 = self.b2_3_2_down(x2_3)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_1 = self.unit2_1(x2_1)
        x2_2 = self.unit2_2(x2_2)
        x2_3 = self.unit2_3(x2_3)
        x2_4_1 = self.unit2_4(x2_4_1)

        # branch1
        x1_4 = self.b1_4_4_up(x1_4_1)

        x1_3 = torch.cat((x1_3, x1_4), dim=1)
        x1_3 = self.b1_3_3(x1_3)
        x1_3 = self.b1_3_4_up(x1_3)

        x1_2 = torch.cat((x1_2, x1_3), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)

        # branch2
        x2_4 = self.b2_4_4_up(x2_4_1)

        x2_3 = torch.cat((x2_3, x2_4), dim=1)
        x2_3 = self.b2_3_3(x2_3)
        x2_3 = self.b2_3_4_up(x2_3)

        x2_2 = torch.cat((x2_2, x2_3), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from thop import profile

if __name__ == '__main__':
    model = DWE(3, 2)
    # path_to_pth_file = "./checkpoints/sup_xnet/GlaS/..."
    # model.load_state_dict(torch.load(path_to_pth_file))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_params = count_parameters(model)
    params_in_MB = num_params * 4 / 1e6
    params_in_Millions = num_params / 1e6

    input1 = torch.randn(1, 3, 128, 128).to(device)
    input2 = torch.randn(1, 3, 128, 128).to(device)

    macs, params = profile(model, inputs=(input1, input2))

    print(f"模型计算量（MACs）：{macs / 1e9:.2f}G")
    print(f"模型参数数量为：{num_params}个")
    print(f"模型参数占用内存大小约为：{params_in_MB:.2f}MB")
    print(f"模型参数数量为：{params_in_Millions:.2f}M（百万）")

