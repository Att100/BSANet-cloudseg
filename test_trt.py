import numpy as np
from PIL import Image
import os
import tqdm
import argparse

from trt.engine import TrtEngine
from utils.metric import get_metrics_trt
    


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--engine', type=str, default='./ckpts/static/bsacloudnet-lite.fp16.trt', 
        help="the tag of model (default: ./ckpts/static/bsacloudnet-lite.fp16.trt)")
    parser.add_argument(
        '--nruns', type=int, default=1000, 
        help="number of runs for computing avg (default: 1000)")
    args = parser.parse_args()

    # path = "./dataset/SWINySEG"
    # split_path = "./dataset/SWINySEG/val.txt"
    # save_path = "./results/prediction/bsamnet-lite"

    # engine = TrtEngine(args.engine)
    # out = engine.test(args.nruns)
    
    accuracy, precision, recall, f_measure, error_rate, miou = get_metrics_trt(args.engine)
    print("- Accuracy: ", accuracy)
    print("- Precision: ", precision)
    print("- Recall: ", recall)
    print("- F-measure: ", f_measure)
    print("- Error-rate: ", error_rate)
    print("- MIOU: ", miou)