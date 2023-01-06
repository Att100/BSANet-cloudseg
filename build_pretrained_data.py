from PIL import Image
import numpy as np
import os
import tqdm


def build_dataset(path, split_path):
    names = [name.strip() for name in open(split_path, 'r').readlines() if '_' not in name]
    if not os.path.exists(os.path.join(path, 'pretrain')):
        os.makedirs(os.path.join(path, 'pretrain', 'images'))
    labels = open(os.path.join(path, 'pretrain', 'labels.txt'), 'w')
    num_pos, num_neg = 0, 0
    pbar = tqdm.trange(0, len(names))
    for name, _ in zip(names, pbar):
        gt = np.array(Image.open(os.path.join(path, 'GTmaps', name+".png")))
        img = np.array(Image.open(os.path.join(path, 'images', name+".jpg")))
        h, w = gt.shape
        assert h%4 == 0 and w%4==0
        h2, w2 = h//4, w//4
        ne = h2 * w2
        for i in range(4):
            for j in range(4):
                gt2 = gt[i*h2:(i+1)*h2, j*w2:(j+1)*w2] // 255
                n_pos = np.sum(gt2)
                if n_pos / ne >= 0.8:
                    labels.write("{},1\n".format("{}_{}.jpg".format(name, i*4+j)))
                    Image.fromarray(img[i*h2:(i+1)*h2, j*w2:(j+1)*w2, :]).resize((320, 320)).save(
                        os.path.join(path, 'pretrain', 'images', "{}_{}.jpg".format(name, i*4+j))
                    )
                    num_pos += 1
                elif n_pos / ne <= 0.2:
                    labels.write("{},0\n".format("{}_{}.jpg".format(name, i*4+j)))
                    Image.fromarray(img[i*h2:(i+1)*h2, j*w2:(j+1)*w2, :]).resize((320, 320)).save(
                        os.path.join(path, 'pretrain', 'images', "{}_{}.jpg".format(name, i*4+j))
                    )
                    num_neg += 1
    print("Number of Positive Samples: {}".format(num_pos))
    print("Number of Negative Samples: {}".format(num_neg))
    labels.close()


if __name__ == "__main__":
    path = "./dataset/SWINySEG"
    split_path = "./dataset/SWINySEG/train.txt"

    build_dataset(path, split_path)