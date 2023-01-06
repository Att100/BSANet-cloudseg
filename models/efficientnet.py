import paddle
import paddle.nn as nn
from models._efficientnet.efficientnet import EfficientNet


class EfficientNetB0(nn.Layer):
    def __init__(self, pretrained=True):
        super().__init__()

        if not pretrained: self.backbone = EfficientNet.from_name('efficientnet-b0', features_only=True)
        else: self.backbone = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2, features_only=True)

    def forward(self, x):
        return self.backbone(x)


