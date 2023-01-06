import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F
import paddle.optimizer as optim
import argparse
import os

from models.loss import _bce_loss_with_aux
from utils.dataset import SWINySEG
from utils.progressbar import ProgressBar

paddle.disable_static()

def bce_loss(pred, target):
    return F.binary_cross_entropy(
        F.sigmoid(paddle.squeeze(pred, 1)), 
        target)

def bce_loss_with_aux(pred, target, weight=[1, 0.6, 0.4, 0.2]):
    return _bce_loss_with_aux(pred, paddle.unsqueeze(target, 1), weight)

def accuracy(pred, label):
    pred_t = F.sigmoid(paddle.squeeze(pred[0], 1))
    return float(
        paddle.mean(((pred_t>0.5).astype('int64')==label).astype('float32')))

def train(args):
    print("# =============== Training Configuration =============== #")
    print("# Model: "+args.model_tag)
    print("# Batchsize: "+str(args.batch_size))
    print("# Learning rate: "+str(args.lr))
    print("# Epochs: "+str(args.epochs))
    print("# Evaluation interval: "+str(args.eval_interval))
    print("# Checkpoints interval: "+str(args.ckpt_interval))
    print("# ====================================================== #")

    paddle.seed(999)

    aug = True
    if args.model_tag == "bsacloudnet-lite":
        from models.bsacloudnet_lite import BSACloudNet
        assert os.path.exists("./pretrained/mobilenetv2_lite_swinyseg_pretr.pdparams")
        model = BSACloudNet("./pretrained/mobilenetv2_lite_swinyseg_pretr.pdparams")
        aug = False
    elif args.model_tag == "bsacloudnet-lite-pure":
        from models.bsacloudnet_lite import BSACloudNet
        model = BSACloudNet(None)
        aug = False
    elif args.model_tag == "baseline-lite":
        from models.baseline_lite import BSACloudNet
        model = BSACloudNet(None)
        aug = False
    elif args.model_tag == "bsacloudnet":
        from models.bsacloudnet import BSACloudNet
        model = BSACloudNet(True)
    elif args.model_tag == "bsacloudnet-large":
        from models.bsacloudnet_large import BSACloudNet
        assert os.path.exists("./pretrained/efficientnet-b0-355c32eb.pdparams")
        model = BSACloudNet(True) 
    else:
        raise Exception("Model name {} not found".format(args.model_tag))

    train_set = SWINySEG(args.dataset_path, 'train', aug)
    test_set = SWINySEG(args.dataset_path, 'val', False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=args.lr, gamma=0.95, verbose=True)
    optimizer = optim.Adam(scheduler, parameters=model.parameters(), weight_decay=4e-5)

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    for e in range(args.epochs):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0

        bar = ProgressBar(maxStep=train_steps)
        model.train()

        for i, (image, label) in enumerate(train_loader()):
            optimizer.clear_grad()
            pred = model(image)

            loss = bce_loss_with_aux(pred, label)

            loss.backward()
            optimizer.step()

            batch_loss = loss.numpy()[0]
            batch_acc = accuracy(pred, label)
            train_loss += batch_loss
            train_acc += batch_acc

            if i != train_steps-1:
                bar.updateBar(
                        i+1, headData={'Epoch':e+1, 'Status':'training'}, 
                        endData={
                            'Train loss': "{:.5f}".format(train_loss/(i+1)),
                            'Train Acc': "{:.5f}".format(train_acc/(i+1))})
            else:
                bar.updateBar(
                        i+1, headData={'Epoch':e+1, 'Status':'finished'}, 
                        endData={
                            'Train loss': "{:.5f}".format(train_loss/(i+1)),
                            'Train Acc': "{:.5f}".format(train_acc/(i+1))})

        if (e+1) % args.eval_interval == 0:
            bar = ProgressBar(maxStep=test_steps)
            model.eval()

            for i, (image, label) in enumerate(test_loader()):
                pred = model(image)

                loss = bce_loss_with_aux(pred, label)

                test_loss += loss.numpy()[0]
                test_acc += accuracy(pred, label)

                if i != test_steps-1:
                    bar.updateBar(
                            i+1, headData={'Epoch (Test)':e+1, 'Status':'testing'}, 
                            endData={
                                'Test loss': "{:.5f}".format(test_loss/(i+1)),
                                'Test Acc': "{:.5f}".format(test_acc/(i+1))})
                else:
                    bar.updateBar(
                            i+1, headData={'Epoch (Test)':e+1, 'Status':'finished'}, 
                            endData={
                                'Test loss': "{:.5f}".format(test_loss/(i+1)),
                                'Test Acc': "{:.5f}".format(test_acc/(i+1))})

        if (e+1) % args.ckpt_interval == 0:
            paddle.save(
                model.state_dict(), 
                "./ckpts/_ckpts/{}_epochs_{}.pdparam".format(args.model_tag, e+1))

        scheduler.step()

    paddle.save(
        model.state_dict(), 
        "./ckpts/{}_epochs_{}.pdparam".format(args.model_tag, args.epochs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_tag', type=str, default='bsacloudnet_lite', 
        help="the tag of model (default: bsacloudnet_lite)")
    parser.add_argument(
        '--batch_size', type=int, default=16, 
        help="batchsize for model training (default: 16)")
    parser.add_argument(
        '--lr', type=float, default=1e-3, 
        help="the learning rate for training (default: 1e-3)")
    parser.add_argument(
        '--epochs', type=int, default=100, 
        help="number of training epochs (default: 100)")
    parser.add_argument(
        '--dataset_path', type=str, default='./dataset/SWINySEG', 
        help="path of training dataset (default: ./dataset/SWINySEG)")
    parser.add_argument(
        '--eval_interval', type=int, default=5, 
        help="interval of model evaluation during training (default: 5)"
    )
    parser.add_argument(
        '--ckpt_interval', type=int, default=10, 
        help="interval of model checkpoints during training (default: 10)"
    )
    
    args = parser.parse_args()
    
    train(args)