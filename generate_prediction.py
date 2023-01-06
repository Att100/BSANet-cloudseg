import paddle
from PIL import Image
import numpy as np
import os
import tqdm
import paddle.nn.functional as F

from models.bsamnet_e import BSAMNetE0
from utils.dataset import SWINySEG


def generate_prediction(model, path, split_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    names = [name.strip() for name in open(split_path, 'r').readlines()]
    pbar = tqdm.trange(0, len(names))

    model.eval()
    for i, name in zip(pbar, names):
        img = Image.open(os.path.join(path, 'images', name+'.jpg'))
        img = img.resize((320, 320))
        img_arr = np.array(img, dtype='float32').transpose(2, 0, 1) / 255
        img_arr = (img_arr - 0.5) / 0.5
        img_tensor = paddle.to_tensor(img_arr).reshape((1, 3, 320, 320))

        pred = F.sigmoid(model(img_tensor)[0])[0][0]
        pred = (pred > 0.5).astype('int64')
        Image.fromarray(np.uint8(pred.numpy()) * 255).save(os.path.join(save_path, name+'.png'))


if __name__ == "__main__":
    path = "./dataset/SWINySEG"
    split_path = "./dataset/SWINySEG/val.txt"
    save_path = "./results/prediction/bsamnet-e0"

    model = BSAMNetE0(False)
    model.set_state_dict(paddle.load("./ckpts/bsamnet_e0_epochs_100.pdparam"))

    generate_prediction(model, path, split_path, save_path)






