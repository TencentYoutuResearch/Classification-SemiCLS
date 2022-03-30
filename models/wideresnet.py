import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
backbone option "WideResNet"
code in this file is adpated from
https://github.com/kekmodel/FixMatch-pytorch/blob/master/models/wideresnet.py
thanks!
"""
logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Normalize(nn.Module):
    """ Ln normalization copied from
    https://github.com/salesforce/CoMatch
    """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class WideResNet(nn.Module):
    def __init__(self,
                 num_classes,
                 depth=28,
                 widen_factor=2,
                 drop_rate=0.0,
                 proj=False,
                 proj_after=False,
                 low_dim=64):
        super(WideResNet, self).__init__()
        # prepare self values
        self.widen_factor = widen_factor
        self.depth = depth
        self.drop_rate = drop_rate
        # if use projection head
        self.proj = proj
        # if use the output of projection head for classification
        self.proj_after = proj_after
        self.low_dim = low_dim

        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # if proj after means we classify after projection head
        # so we must change the in channel to low_dim of laster fc
        if self.proj_after:
            self.fc = nn.Linear(self.low_dim, num_classes)
        else:
            self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        # projection head
        if self.proj:
            self.l2norm = Normalize(2)

            self.fc1 = nn.Linear(64 * self.widen_factor, 64 * self.widen_factor)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(64 * self.widen_factor, self.low_dim)

        self.init_wegihts()

    # init_wegihts
    def init_wegihts(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.relu(self.bn1(feat))
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.view(-1, self.channels)

        if self.proj:
            pfeat = self.fc1(feat)
            pfeat = self.relu_mlp(pfeat)
            pfeat = self.fc2(pfeat)
            pfeat = self.l2norm(pfeat)

            # if projection after classifiy, we classify last
            if self.proj_after:
                out = self.fc(pfeat)
            else:
                out = self.fc(feat)

            return out, pfeat

        # output
        out = self.fc(feat)
        return out


def build_wideresnet(depth,
                     widen_factor,
                     dropout,
                     num_classes,
                     proj=False,
                     low_dim=64,
                     **kwargs):
    logger.info(
        f"Model: WideResNet {depth}x{widen_factor} proj {proj}x{low_dim}")
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                      proj=proj,
                      low_dim=low_dim,
                      **kwargs)


def build(**kwargs):
    return build_wideresnet(**kwargs)
