import argparse
import os

from utils.metric import get_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_tag', type=str, default='bsacloudnet_lite', 
        help="the tag of model (default: bsacloudnet_lite)")
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

    if args.model_tag == "bsacloudnet-lite":
        assert not args.iou, "Currently not supported for lite version"
        from models.bsacloudnet_lite import BSACloudNet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSACloudNet(), 
            "./ckpts/bsacloudnet-lite_epochs_100.pdparam",
            args.dataset_path, args.daynight)
        print("----- BSACloudNet-lite -----")
    elif args.model_tag == "bsacloudnet-lite-pure":
        assert not args.iou, "Currently not supported for lite version"
        from models.bsacloudnet_lite import BSACloudNet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSACloudNet(), 
            "./ckpts/bsacloudnet-lite-pure_epochs_100.pdparam",
            args.dataset_path, args.daynight)
        print("----- BSACloudNet-lite-pure -----")
    elif args.model_tag == "bsacloudnet":
        from models.bsacloudnet import BSACloudNet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSACloudNet(), 
            "./ckpts/bsacloudnet{}_epochs_100.pdparam".format("-iou" if args.iou else ""),
            args.dataset_path, args.daynight)
        print("----- BSACloudNet -----")
    elif args.model_tag == "bsacloudnet-large":
        from models.bsacloudnet_large import BSACloudNet
        accuracy, precision, recall, f_measure, error_rate, miou = get_metrics(
            BSACloudNet(), 
            "./ckpts/bsacloudnet-large{}_epochs_100.pdparam".format("-iou" if args.iou else ""),
            args.dataset_path, args.daynight)
        print("----- BSACloudNet-large -----")
    else:
        raise Exception("Model name {} not found".format(args.model_tag))

    print("- Accuracy: ", accuracy)
    print("- Precision: ", precision)
    print("- Recall: ", recall)
    print("- F-measure: ", f_measure)
    print("- Error-rate: ", error_rate)
    print("- MIOU: ", miou)



