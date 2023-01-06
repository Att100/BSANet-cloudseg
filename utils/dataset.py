import paddle
from paddle.io import Dataset
from PIL import Image
import os
import numpy as np


class SWINySEG(Dataset):
    def __init__(self, path="./dataset/SWINySEG", split='train', aug=True, img_size=(320, 320), gt_size=(320, 320), daynight='all'):
        super().__init__()

        self.path = path
        self.split = split
        self.aug = aug
        self.img_size = img_size
        self.gt_size = gt_size
        self.names = [line.strip() for line in open(os.path.join(path, split+".txt")).readlines()]
        
        if daynight == 'day':
            self.names = [name for name in self.names if name[0] == 'd']
        elif daynight == 'night':
            self.names = [name for name in self.names if name[0] == 'n']
        else:
            pass

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, 'images', self.names[idx]+".jpg"))
        gt = Image.open(os.path.join(self.path, 'GTmaps', self.names[idx]+".png"))
        img = img.resize(self.img_size)
        gt = gt.resize(self.gt_size)
        # to numpy array and normalize
        img_arr = np.array(img, dtype='float32').transpose(2, 0, 1) / 255
        gt_arr = np.array(gt, dtype='float32') / 255
        img_arr = (img_arr - 0.5) / 0.5
        if self.split == 'train' and self.aug:
            # random h flip
            choice = np.random.choice([0, 1])
            if choice == 1:
                img_arr = img_arr[:, :, ::-1]
                gt_arr = gt_arr[:, ::-1]
            # random v flip
            choice = np.random.choice([0, 1])
            if choice == 1:
                img_arr = img_arr[:, ::-1, :]
                gt_arr = gt_arr[::-1, :]
        img_tensor = paddle.to_tensor(img_arr)
        gt_tensor = paddle.to_tensor(gt_arr)
        return img_tensor, gt_tensor

    def __len__(self):
        return len(self.names)