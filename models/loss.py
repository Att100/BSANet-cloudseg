import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def _iou_loss(pred, target, smooth=1):
    intersection = paddle.sum(target * pred, axis=[1,2,3])
    union = paddle.sum(target, axis=[1,2,3]) + paddle.sum(pred, axis=[1,2,3])
    iou = paddle.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1 - iou

def _bce_loss_with_aux(pred, target, weight=[1, 0.6, 0.4, 0.2]):
    # pred = (x1, x2, x4)
    pred, pred_down2, pred_down4, pred_down8 = tuple(pred)

    target_2x = F.interpolate(
        target, pred_down2.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, pred_down4.shape[2:], mode='bilinear', align_corners=True)
    target_8x = F.interpolate(
        target, pred_down8.shape[2:], mode='bilinear', align_corners=True)
    
    loss_1x = F.binary_cross_entropy(F.sigmoid(pred), target)
    loss_2x = F.binary_cross_entropy(F.sigmoid(pred_down2), target_2x)
    loss_4x = F.binary_cross_entropy(F.sigmoid(pred_down4), target_4x)
    loss_8x = F.binary_cross_entropy(F.sigmoid(pred_down8), target_8x)

    loss = weight[0] * loss_1x + weight[1] * loss_2x + weight[2] * loss_4x + weight[3] * loss_8x
    return loss

def _bce_iou_loss_with_aux(pred, target, weight=[1, 0.6, 0.4, 0.2]):
    # pred = (x1, x2, x4)
    pred, pred_down2, pred_down4, pred_down8 = tuple(pred)

    target_2x = F.interpolate(
        target, pred_down2.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, pred_down4.shape[2:], mode='bilinear', align_corners=True)
    target_8x = F.interpolate(
        target, pred_down8.shape[2:], mode='bilinear', align_corners=True)
    
    pred, pred_down2, pred_down4, pred_down8 = \
        F.sigmoid(pred), F.sigmoid(pred_down2), F.sigmoid(pred_down4), F.sigmoid(pred_down8)
    loss_1x = F.binary_cross_entropy(pred, target) + _iou_loss(pred, target)
    loss_2x = F.binary_cross_entropy(pred_down2, target_2x) + _iou_loss(pred_down2, target_2x)
    loss_4x = F.binary_cross_entropy(pred_down4, target_4x) + _iou_loss(pred_down4, target_4x)
    loss_8x = F.binary_cross_entropy(pred_down8, target_8x) + _iou_loss(pred_down8, target_8x)

    loss = weight[0] * loss_1x + weight[1] * loss_2x + weight[2] * loss_4x + weight[3] * loss_8x
    return loss