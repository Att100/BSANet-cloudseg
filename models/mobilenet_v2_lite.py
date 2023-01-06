import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer=nn.BatchNorm2D):
        padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2D(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias_attr=False),
            norm_layer(out_planes),
            nn.ReLU6())


class InvertedResidual(nn.Layer):
    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 norm_layer=nn.BatchNorm2D):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(
                    inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer),
            nn.Conv2D(
                hidden_dim, oup, 1, 1, 0, bias_attr=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _MobileNetV2(nn.Layer):
    def __init__(self, scale=1.0):
        super(_MobileNetV2, self).__init__()
        input_channel = 2
        last_channel = 128

        block = InvertedResidual
        round_nearest = 8
        norm_layer = nn.BatchNorm2D
        inverted_residual_setting = [
            [1, 2, 1, 1],
            [6, 4, 2, 2],
            [6, 8, 3, 2],
            [6, 12, 4, 2],
            [6, 16, 3, 1],
            [6, 18, 3, 2],
            [6, 18, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * scale, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, scale),
                                            round_nearest)
        features = [
            ConvBNReLU(
                3, input_channel, stride=2, norm_layer=norm_layer)
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer))
                input_channel = output_channel

        features.append(
            ConvBNReLU(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer))

        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x

class MobileNetV2(nn.Layer):
    def __init__(self):
        super().__init__()

        mobilenet = _MobileNetV2()
        self.features = mobilenet.features

        # 5 x downsample --> 4 x downsample
        self.features._sub_layers['14']._sub_layers['conv'].\
            _sub_layers['1']._sub_layers['0']._stride = [1, 1]

    def forward(self, x):
        # stage 1
        feat1 = self.features._sub_layers['0'](x)

        # stage 2
        feat2 = feat1
        for key in ['1', '2', '3']:
            feat2 = self.features[key](feat2)

        # stage 3
        feat3 = feat2
        for key in ['4', '5', '6']:
            feat3 = self.features[key](feat3)

        # stage 4
        feat4 = feat3
        for key in ['7', '8', '9', '10']:
            feat4 = self.features[key](feat4)

        # stage 5
        out = feat4
        for key in ['11', '12', '13', '14', '15', '16', '17', '18']:
            out= self.features[key](out)

        return feat1, feat2, feat3, feat4, out

