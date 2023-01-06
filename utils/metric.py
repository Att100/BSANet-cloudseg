from typing import Tuple
import paddle
import time
import paddle.nn.functional as F
from paddle.io import DataLoader
import numpy as np
import os
from PIL import Image

from utils.dataset import SWINySEG
from utils.progressbar import ProgressBar
from trt.predictor import TrtPredictor


def postprocess(output):
    pred_t = F.sigmoid(paddle.squeeze(output[0], 1))
    return (pred_t > 0.5).astype('int')

def get_cfm(pred, label):
    tp = paddle.sum((pred==1).astype('int')+(label==1).astype('int') == 2)
    fn = paddle.sum((pred==0).astype('int')+(label==1).astype('int') == 2)
    fp = paddle.sum((pred==1).astype('int')+(label==0).astype('int') == 2)
    tn = paddle.sum((pred==0).astype('int')+(label==0).astype('int') == 2)
    return tp, fn, fp, tn

def get_cfm_np(pred, label):
    tp = np.sum((pred==1).astype('int')+(label==1).astype('int') == 2)
    fn = np.sum((pred==0).astype('int')+(label==1).astype('int') == 2)
    fp = np.sum((pred==1).astype('int')+(label==0).astype('int') == 2)
    tn = np.sum((pred==0).astype('int')+(label==0).astype('int') == 2)
    return tp, fn, fp, tn

def get_img_metrics(pred, label):
    tp, fn, fp, tn = get_cfm(pred, label)
    accuracy = float(paddle.mean((pred==label).astype('float32')))
    precision = int(tp)/(int(tp)+int(fp))
    recall = int(tp)/(int(tp)+int(fn))
    f_measure = (2 * precision * recall) / (precision + recall)
    error_rate = (int(fp)+int(fn))/(int(tp)+int(fp)+int(tn)+int(fn))
    miou = ((int(tp)/(int(tp)+int(fp)+int(fn)))+(int(tn)/(int(tn)+int(fp)+int(fn))))/2
    return accuracy, precision, recall, f_measure, error_rate, miou

def get_pr_curve(model, weight_path='', dataset_path='./dataset/SWINySEG', split='all') -> Tuple:
    """
    Thresholds: 0-255

    return: 
        p: precision, based on 256 thresholds, shape (256,)
        r: recall, based on 256 thresholds, shape (256,) 
    """
    test_set = SWINySEG(dataset_path, split, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.set_state_dict(paddle.load(weight_path))
    model.eval()

    bar = ProgressBar(maxStep=len(test_loader))

    precision = paddle.zeros((256, len(test_loader)))
    recall = paddle.zeros((256, len(test_loader)))

    tp = paddle.zeros((256, len(test_loader)))
    fp = paddle.zeros((256, len(test_loader)))
    fn = paddle.zeros((256, len(test_loader)))

    thresholds = paddle.to_tensor(np.array([[[i for i in range(256)]]])).astype('int32')

    for i, (image, label) in enumerate(test_loader()):
        pred = model(image)

        pred_t = F.sigmoid(paddle.squeeze(pred[0], 1))[0]
        pred_t = pred_t * 255
        label_t = label[0]

        pred_mask = (paddle.unsqueeze(pred_t, -1) > thresholds).astype('int32')
        tfnp = 2 * pred_mask - paddle.unsqueeze(label_t, -1)
        tp[:, i] = paddle.sum((tfnp==1).astype('float32'), axis=(0, 1))
        fp[:, i] = paddle.sum((tfnp==2).astype('float32'), axis=(0, 1))
        fn[:, i] = paddle.sum((tfnp==-1).astype('float32'), axis=(0, 1))

        precision[:, i] = tp[:, i] / (tp[:, i]+fp[:, i])
        recall[:, i] = tp[:, i] / (tp[:, i]+fn[:, i])

        bar.updateBar(i+1, headData={}, endData={})

    p = paddle.mean(precision, axis=1)
    r = paddle.mean(recall, axis=1)
    return p.numpy(), r.numpy()


def get_roc_curve(model, weight_path='', dataset_path='./dataset/SWINySEG', split='all') -> Tuple:
    """
    Thresholds: 0-255

    return: 
        tpr: true positive rate, based on 256 thresholds, shape (256,) 
        fpr: false positive rate, based on 256 thresholds, shape (256,) 
    """
    test_set = SWINySEG(dataset_path, split, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.set_state_dict(paddle.load(weight_path))
    model.eval()

    bar = ProgressBar(maxStep=len(test_loader))

    fpr = paddle.zeros((256, len(test_loader)))
    tpr = paddle.zeros((256, len(test_loader)))

    tp = paddle.zeros((256, len(test_loader)))
    fp = paddle.zeros((256, len(test_loader)))
    tn = paddle.zeros((256, len(test_loader)))
    fn = paddle.zeros((256, len(test_loader)))

    thresholds = paddle.to_tensor(np.array([[[i for i in range(256)]]])).astype('int32')

    for i, (image, label) in enumerate(test_loader()):
        pred = model(image)

        pred_t = F.sigmoid(paddle.squeeze(pred, 1))[0]
        pred_t = pred_t * 255
        label_t = label[0]

        pred_mask = (paddle.unsqueeze(pred_t, -1) > thresholds).astype('int32')
        tfnp = 2 * pred_mask - paddle.unsqueeze(label_t, -1)
        tp[:, i] = paddle.sum((tfnp==1).astype('float32'), axis=(0, 1))
        fp[:, i] = paddle.sum((tfnp==2).astype('float32'), axis=(0, 1))
        tn[:, i] = paddle.sum((tfnp==0).astype('float32'), axis=(0, 1)) 
        fn[:, i] = paddle.sum((tfnp==-1).astype('float32'), axis=(0, 1))

        fpr[:, i] = fp[:, i] / (fp[:, i]+tn[:, i])
        tpr[:, i] = tp[:, i] / (tp[:, i]+fn[:, i])

        bar.updateBar(i+1, headData={}, endData={})

    tpr = paddle.mean(tpr, axis=1)
    fpr = paddle.mean(fpr, axis=1)
    return tpr.numpy(), fpr.numpy()

def get_metrics(model, weight_path='', dataset_path='./dataset/SWINySEG', daynight='all') -> Tuple:
    """
    Thresholds: 0.5
    """
    test_set = SWINySEG(dataset_path, 'val', gt_size=(300, 300), daynight=daynight)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model.set_state_dict(paddle.load(weight_path))
    model.eval()

    bar = ProgressBar(maxStep=len(test_loader))
    accuracy, precision, recall, f_measure, error_rate, miou = 0, 0, 0, 0, 0, 0

    for i, (image, label) in enumerate(test_loader()):
        pred = model(image)

        pred_t = F.interpolate(F.sigmoid(pred[0]), (300, 300), mode='bilinear', align_corners=True)
        pred_t = (pred_t[0][0] > 0.5).astype('int32')
        label_t = label[0]

        tp, fn, fp, tn = get_cfm(pred_t, label_t)

        accuracy += float(paddle.mean((pred_t==label_t).astype('float32')))
        precision += int(tp)/(int(tp)+int(fp))
        recall += int(tp)/(int(tp)+int(fn))
        error_rate += (int(fp)+int(fn))/(int(tp)+int(fp)+int(tn)+int(fn))
        miou += ((int(tp)/(int(tp)+int(fp)+int(fn)))+(int(tn)/(int(tn)+int(fp)+int(fn))))/2

        bar.updateBar(i+1, headData={}, endData={})

    accuracy /= len(test_loader)
    precision /= len(test_loader)
    recall /= len(test_loader)
    f_measure = (2 * precision * recall) / (precision + recall)
    error_rate /= len(test_loader)
    miou /= len(test_loader)

    return accuracy, precision, recall, f_measure, error_rate, miou

def get_metrics_trt(engine_path, dataset_path='./dataset/SWINySEG', daynight='all') -> Tuple:
    """
    Thresholds: 0.5
    """
    predictor = TrtPredictor(engine_path)
    names = [line.strip() for line in open(os.path.join(dataset_path, "val.txt")).readlines()]
    if daynight == 'day':
        names = [name for name in names if name[0] == 'd']
    elif daynight == 'night':
        names = [name for name in names if name[0] == 'n']
    else:
        pass

    bar = ProgressBar(maxStep=len(names))
    accuracy, precision, recall, f_measure, error_rate, miou = 0, 0, 0, 0, 0, 0

    for i, name in enumerate(names):
        img = Image.open(os.path.join(dataset_path, 'images', name+".jpg"))
        gt = Image.open(os.path.join(dataset_path, 'GTmaps', name+".png"))

        pred = predictor.run(img)
        label = np.array(gt)

        tp, fn, fp, tn = get_cfm_np(pred//255, label//255)

        accuracy += float(np.mean((pred==label).astype('float32')))
        precision += int(tp)/(int(tp)+int(fp))
        recall += int(tp)/(int(tp)+int(fn))
        error_rate += (int(fp)+int(fn))/(int(tp)+int(fp)+int(tn)+int(fn))
        miou += ((int(tp)/(int(tp)+int(fp)+int(fn)))+(int(tn)/(int(tn)+int(fp)+int(fn))))/2

        bar.updateBar(i+1, headData={}, endData={})

    accuracy /= len(names)
    precision /= len(names)
    recall /= len(names)
    f_measure = (2 * precision * recall) / (precision + recall)
    error_rate /= len(names)
    miou /= len(names)

    return accuracy, precision, recall, f_measure, error_rate, miou
