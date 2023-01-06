import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
from PIL import Image
import numpy as np
from utils.progressbar import ProgressBar
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader

from models.mobilenet_v2_lite import _MobileNetV2


class MobileNetV2Pre(nn.Layer):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.last_channel = 128

        self.features = _MobileNetV2()
        self.pool2d_avg = nn.AdaptiveAvgPool2D(1)
        self.classifier = nn.Sequential(
                nn.Dropout(0.2), 
                nn.Linear(self.last_channel, num_classes if num_classes !=2 else 1)
        )

    def forward(self, x):
        features = self.features(x)
        x = self.pool2d_avg(features)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save_backbone_state_dict(self, path):
        paddle.save(self.features.features.state_dict(), path)


class SWINySEG_CLS(Dataset):
    def __init__(self, path, label_path):
        super().__init__()

        self.data = []
        self.path = path
        self.retrieve(label_path)

    def retrieve(self, label_path):
        lines = open(label_path, 'r').readlines()
        for line in lines:
            name, label = line.strip().split(',')
            label = int(label)
            self.data.append([name, label])

    def __getitem__(self, idx):
        name, label = self.data[idx]
        img = Image.open(os.path.join(self.path, 'images', name))
        img_arr = np.array(img, dtype='float32').transpose(2, 0, 1) / 255
        gt_arr = np.array([label], dtype='float32') / 255
        img_tensor = paddle.to_tensor(img_arr)
        gt_tensor = paddle.to_tensor(gt_arr)
        return img_tensor, gt_tensor

    def __len__(self):
        return len(self.data)


def accuracy(pred, label):
    pred_t = F.sigmoid(pred)
    return float(
        paddle.mean(((pred_t>0.5).astype('int64')==label).astype('float32')))


if __name__ == "__main__":
    path = "./dataset/SWINySEG/pretrain"
    label_path = "./dataset/SWINySEG/pretrain/labels.txt"
    save_path = "./pretrained/mobilenetv2_lite_swinyseg_pretr.pdparams"

    dataset = SWINySEG_CLS(path, label_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    model = MobileNetV2Pre(2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(learning_rate=1e-3, parameters=model.parameters(), weight_decay=4e-5)

    steps = len(dataloader)

    model.train()
    for i in range(5):
        train_acc = 0
        bar = ProgressBar(maxStep=steps)
        for j, (img, label) in enumerate(dataloader):
            optimizer.clear_grad()

            out = model(img)
            loss = criterion(F.sigmoid(out), label)
            train_acc += accuracy(out, label)

            loss.backward()
            optimizer.step()

            if i != steps-1:
                bar.updateBar(
                        j+1, headData={'Epoch':i+1, 'Status':'training'}, 
                        endData={'Train Acc': "{:.5f}".format(train_acc/(j+1))})
            else:
                bar.updateBar(
                        j+1, headData={'Epoch':i+1, 'Status':'finished'}, 
                        endData={'Train Acc': "{:.5f}".format(train_acc/(j+1))})

    model.save_backbone_state_dict(save_path)





