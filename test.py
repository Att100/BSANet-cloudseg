import argparse
import os

from utils.metric import get_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_tag', type=str, default='bsanet_lite', 
        help="the tag of model (default: bsanet_lite)")
    parser.add_argument(
        '--iou', type=bool, default=False, 
        help="evaluate iou version (default: False)")
    parser.add_argument(
        '--dataset_path', type=str, default='./dataset/SWINySEG', 
        help="path of dataset (default: ./dataset/SWINySEG)")
    parser.add_argument(
        '--daynight', type=str, default='all', 
        help="day/night/all (default: all)")
    args = parser.parse_args()

    if args.model_tag == "bsanet-lite":
        assert not args.iou, "Currently not supported for lite version"
        from models.bsanet_lite import BSANet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSANet(), 
            "./ckpts/bsanet-lite_epochs_100.pdparam",
            args.dataset_path, args.daynight)
        print("----- Bsanet-lite -----")
    elif args.model_tag == "bsanet-lite-pure":
        assert not args.iou, "Currently not supported for lite version"
        from models.bsanet_lite import BSANet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSANet(), 
            "./ckpts/bsanet-lite-pure_epochs_100.pdparam",
            args.dataset_path, args.daynight)
        print("----- BSANet-lite-pure -----")
    elif args.model_tag == "bsanet":
        from models.bsanet import BSANet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSANet(), 
            "./ckpts/bsanet{}_epochs_100.pdparam".format("-iou" if args.iou else ""),
            args.dataset_path, args.daynight)
        print("----- BSANet -----")
    elif args.model_tag == "bsanet-large":
        from models.bsanet_large import BSANet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSANet(), 
            "./ckpts/bsanet-large{}_epochs_100.pdparam".format("-iou" if args.iou else ""),
            args.dataset_path, args.daynight)
        print("----- BSANet-large -----")
    else:
        raise Exception("Model name {} not found".format(args.model_tag))

    print("- Accuracy: ", accuracy)
    print("- Precision: ", precision)
    print("- Recall: ", recall)
    print("- F-measure: ", f_measure)
    print("- Error-rate: ", error_rate)
    print("- MIOU: ", miou)



