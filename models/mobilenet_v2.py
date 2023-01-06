import paddle
import paddle.nn as nn
from paddle.vision.models import mobilenet_v2

class MobileNetV2(nn.Layer):
    def __init__(self, pretrained=True):
        super().__init__()

        mobilenet = mobilenet_v2(pretrained=pretrained)
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