import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_ch=1):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target


class ResNet18NoStride(nn.Module):
    def __init__(self, in_ch=1):
        super(ResNet18NoStride, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(2)
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.pool1(out)
        out = F.interpolate(out, scale_factor=1/2)
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.pool2(out)
        out = F.interpolate(out, scale_factor=1 / 2)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class ResNet18NoStride1x1(nn.Module):
    def __init__(self, in_ch=1):
        super(ResNet18NoStride1x1, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(2)
        self.layer1 = self.make_layer(ResidualBlock, 8,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 16, 2, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = self.make_layer(ResidualBlock, 32, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 64, 2, stride=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, 1))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.pool1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.pool2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target



""" Dense Feature Extractor Network """
def preconv2d(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bn=True):
    if bn:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))

class BasicDenseLayer(nn.Module):

    def __init__(self, inplanes, planes, stride = 1, pad = 1, dilation = 1):
        super(BasicDenseLayer, self).__init__()

        self.conv1 = preconv2d(inplanes, 4*planes, 1, 1, 0)
        self.conv2 = preconv2d(4*planes, planes, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.cat([x, out], 1)

        return out

class TransitionLayer(nn.Module):

    def __init__(self, inplanes, reduction = 0.5, stride = 2):
        super(TransitionLayer, self).__init__()

        self.trans = nn.Sequential()
        if reduction < 1:
            self.trans = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, math.ceil(inplanes*reduction), 1, padding=0, stride=1, bias=False))

        if reduction == 1 or stride > 1:
            self.trans.add_module('avg_pool', nn.AvgPool2d(stride))

    def forward(self, x):
        out = self.trans(x)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.dense = nn.Sequential(OrderedDict([
            ('norm1', nn.BatchNorm2d(num_input_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(num_input_features, bn_size *
                                growth_rate, kernel_size=1, stride=1, bias=False)),
            ('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                kernel_size=3, stride=1, padding=1, bias=False),)]))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.dense(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseFeature(nn.Module):
    def __init__(self, init_channels, growth_rate=8, scales=5, num_layers=6):
        super(DenseFeature, self).__init__()

        self.init_channels = init_channels
        nC = self.init_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.bn_size = 4
        self.num_blocks = scales

        self.block0 = nn.Sequential(nn.Conv2d(3,  nC, 3, 1, 1), # 512x256
                                    preconv2d(nC, nC, 3, 2, 1), # 256x128
                                    nn.MaxPool2d(2, 2))  # new: downsample 4 times


        block_config = [self.num_layers] * self.num_blocks
        self.blocks = []
        self.transs =[]

        num_features = nC
        list_num_features = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=self.bn_size, growth_rate=self.growth_rate, drop_rate=0)
            self.blocks.append(block)
            num_features = num_features + num_layers * self.growth_rate
            list_num_features.append(num_features)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transs.append(trans)
                num_features = num_features // 2

        self.blocks = nn.ModuleList(self.blocks)
        self.transs = nn.ModuleList(self.transs)



        self.upblocks = []
        for i in reversed(range(1, len(list_num_features))):
            self.upblocks.append(
                nn.Sequential(preconv2d(list_num_features[i], list_num_features[i-1],3,1,1),
                              preconv2d(list_num_features[i-1], list_num_features[i-1],3,1,1))
            )
        self.upblocks = nn.ModuleList(self.upblocks)

    def forward(self, x, timer = False):
        init_features = self.block0(x)
        features = []
        for i in range(self.num_blocks):
            if i > 0:
                fe = self.blocks[i](self.transs[i-1](features[i-1]))
            else:
                fe = self.blocks[i](init_features)
            features.append(fe)

        outs = [features[-1]]
        for i in reversed(range(self.num_blocks-1)):
            out = features[i] + self.upblocks[self.num_blocks-2-i](
                F.upsample(outs[-1], (features[i].size()[2], features[i].size()[3]), mode='bilinear'))
            outs.append(out)

        return outs[-1]


class AlexNet(nn.Module):
    """
    AlexNet backbone
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature_channel = 256
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x):
        x = self.feature(x)
        return x






## resnet 50
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50_feature_extractor(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet50_feature_extractor, self).__init__()

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                else:
                    raise NotImplementedError

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResidualBlockCrop(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlockCrop, self).__init__()
        p = (kernel_size-1)//2
        self.p = p
#         p = 0
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=p, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=p, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out[:,:,self.p*2:-self.p*2,self.p*2:-self.p*2].contiguous()

class ResNet18Crop(nn.Module):
    def __init__(self, in_ch=3):
        super(ResNet18Crop, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlockCrop, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlockCrop, 128, 2, stride=1)
        self.layer3 = self.make_layer(ResidualBlockCrop, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlockCrop, 256, 2, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = out[:,:,2:-2,2:-2].contiguous()
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.pool(out)
        out = out[:,:,::2,::2].contiguous()
        out = self.layer3(out)
        out = self.layer4(out)
        return out




