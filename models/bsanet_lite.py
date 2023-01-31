import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .mobilenet_v2_lite import InvertedResidual
from .mobilenet_v2_lite import MobileNetV2



class ConvBnReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()

        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = F.relu(out)
        return self.bn(out)

class DoubleConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = InvertedResidual(in_channels, out_channels, 1, 1)
        self.conv2 = InvertedResidual(out_channels, out_channels, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class Up(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = InvertedResidual(in_channels, out_channels, 1, 1)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out

class BSAM(nn.Layer):
    """
    Bilateral Segregation and Agregation Module
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.reduce0 = ConvBnReLU(in_channels*2, in_channels, 1)
        self.reduce1 = ConvBnReLU(in_channels, in_channels//2, 1)
        self.reduce2 = ConvBnReLU(in_channels, in_channels//2, 1)

        self.backg_branch = nn.Sequential(
            nn.Conv2D(in_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.final1 = InvertedResidual(in_channels, out_channels, 1, 1)
        self.final2 = InvertedResidual(in_channels, out_channels, 1, 1)

        self.classifier = nn.Conv2D(out_channels, 1, 3, 1, 1)

    def forward(self, x, shortcut, mask):
        addit = self.reduce0(paddle.concat([x, shortcut], axis=1))
        multi = addit * F.sigmoid(mask)
        subtr = addit - multi

        left = paddle.concat([
            self.reduce1(addit), self.reduce2(multi)], axis=1)
        right = left * self.backg_branch(subtr)

        out = self.final1(left) + self.final2(right)
        out = F.upsample(out, scale_factor=2, mode='bilinear', align_corners=True)
        seg = self.classifier(out)
        return out, seg

class BSANet(nn.Layer):
    def __init__(self, pretrain_path=None, mode='train'):
        super().__init__()

        self.mode = mode

        # encoder/backbone
        self.mobilenet_v2 = MobileNetV2()
        if pretrain_path is not None:
            self.mobilenet_v2.features.set_state_dict(paddle.load(pretrain_path))

        # conv 1x1
        self.conv4 = ConvBnReLU(16, 128, 1)
        self.conv3 = ConvBnReLU(8, 16, 1)
        self.conv2 = ConvBnReLU(8, 8, 1)
        self.conv1 = ConvBnReLU(8, 8, 1)

        # decoder
        self.dec1 = nn.Sequential(
            DoubleConv2D(128, 16),
            Up(16, 16))
        self.dec1_cls = nn.Conv2D(16, 1, 3, 1, 1)
        self.dec2 = BSAM(16, 8)
        self.dec3 = BSAM(8, 8)
        self.dec4 = BSAM(8, 8)

    def forward(self, x):
        # encode
        feat1, feat2, feat3, feat4, output = self.mobilenet_v2(x)

        feat4 = self.conv4(feat4)
        feat3 = self.conv3(feat3)
        feat2 = self.conv2(feat2)
        feat1 = self.conv1(feat1)

        #decode
        x2 = self.dec1(output + feat4)
        x2_out = self.dec1_cls(x2)
        x4, x4_out = self.dec2(x2, feat3, x2_out)
        x8, x8_out = self.dec3(x4, feat2, x4_out)
        _, out = self.dec4(x8, feat1, x8_out)
        
        if self.mode == 'train':
            return out, x8_out, x4_out, x2_out
        return out


