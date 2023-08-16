## BSANet for Real-time Cloud Segmentation
With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript: 

> Y. Li, H. Wang, S. Wang, Y. H. Lee, S. Dev. “BSANet: A Bilateral Segregation and Aggregation Network for Real-time Sky/Cloud Segmentation”, under review

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.

### Executive summary
Segmenting clouds from intensity images is an essential research topic at the intersection of atmospheric science and computer vision, which plays a vital role in weather forecasts, environmental monitoring, and climate evolution analysis. The ground-based sky/cloud image segmentation can help to extract the cloud from the original image and analyze the shape or additional features. In this paper, we introduced a novel sky/cloud segmentation network named Bilateral Segregation and Aggregation Network (BSANet) with 16.37 MBytes, which can reduce 70.68% of model size and achieve almost the same performance as the state-of-the-art method shown as the figure below. After the deployment via TensorRT, BSANet-large configuration can achieve 392 fps in FP16 while BSANet-lite can achieve 1390 fps, which all exceed real-time standards. Additionally, we proposed a novel and fast pre-training strategy for sky/cloud segmentation which can improve the accuracy when ImageNet pre-training is unavailable.

<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Accuracy%20and%20model%20size%20comparison.png" width="400"/></div>

### Code
* `./models/`: This folder contains BSANet model code.
* `./utils/`: This folder contains three assistant files (dataset, progressbar and metrics)
* `./weights/`: This folder contains the weights after model training.
* `notebook.ipynb`: This notebook comtains code block for evaluation.
* `train.py`: Script for model training


### Environment and Preparation

- We provide requirements.txt for all modules needed in training and testing, if `paddlepaddle-gpu` can't be installed successfully, please visit [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html) and follow the official instructions.

    ```
    conda create -n paddle python=3.9
    conda activate paddle
    conda install --yes --file requirements.txt
    ```

- Download SWINySEG dataset from [SWINySEG](http://vintage.winklerbros.net/swinyseg.html), and place the uncompressed folder under `./dataset` folder. Your `./dataset` directory should follow the structure below, if the name of uncompressed folder is not SWINySEG, please rename it to SWINySEG.

    ```
    └─SWINySEG
        ├─GTmaps
        └─images
    ```

- TensorRT (optional), if you want to try the TensorRT optimized version, please follow the scripts below to install dependencies

    ```
    python3 -m pip install --upgrade setuptools pip
    python3 -m pip install nvidia-pyindex
    pip install numpy==1.20.3
    python3 -m pip install --upgrade nvidia-tensorrt==8.2.0.6
    pip install pycuda
    pip install paddle2onnx
    ```

### Data
* `./dataset/`: This folder contains the full SWINySEG dataset.

### Model
* `BSANet Architecture.png`: It shows the overall architecture of BSANet and BSAM. (a) illustrate the overall pipeline of BSANet and (b) indicate the detailed design of bilateral segregation and aggregation module (BSAM). The procedure between the output of the model and the segmentation mask has been omitted in this figure.


<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Architecture%20of%20BSANet%20and%20BSAM.png" width="650"/></div>

### SWINySEG-based Pre-training Strategy
Figure below shows the schematic diagram of our proposed SWINySEG-based pre-training. (a) illustrates the positive and negative sample generation process. (b) indicates the negative samples. (c) is positive samples. (d) represents the modules involved in pre-training.

<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Schematic%20diagram%20of%20our%20proposed%20SWINySEG-based%20pre-training.png" width="700"/></div>



### Training

- generate SWINySEG-based pretrain dataset

    ```
    python build_pretrained_data.py
    ```

- train models

    ```
    # for BSANet-lite
    python train.py --model_tag bsanet-lite

    # for BSANet
    python train.py --model_tag bsanet

    # for BSANet-large
    python train.py --model_tag bsanet-large

    # for BSANet-large with IOU loss
    python train_iou.py --model_tag bsanet-large
    ```

### Testing

Evaluation on full SWINySEG test set

```
# for BSANet-lite
python test.py --model_tag bsanet-lite

# for BSANet
python test.py --model_tag bsanet

# for BSANet-large
python test.py --model_tag bsanet-large

# for BSANet-large with IOU loss
python test.py --model_tag bsanet-large --iou True
```

Evaluation on day-time or night-time images

```
# add '--daynight day' or '--daynight night' after script above
```

### TensorRT

```
python dynamic2static.py --model_tag bsanet-lite --ckpt_path ./ckpts/bsanet-lite_epochs_100.pdparam
paddle2onnx --model_dir ./ckpts/static --model_filename bsanet-lite.pdmodel --params_filename bsanet-lite.pdiparams --save_file ./ckpts/static/bsanet-lite.onnx

# FOR FP32
python onnx_to_tensorrt.py -m ./ckpts/static/bsanet-lite.onnx -d fp32
# FOR FP16
python onnx_to_tensorrt.py -m ./ckpts/static/bsanet-lite.onnx -d fp16
```

### Results
The figure below illustrates the qualitative results of our proposed models. The two leftmost columns show the source images and their corresponding ground truth. The red rectangle indicates the part that can reflect the performance of the model the most. It shows that our BSANet-lite, the lightest version, can accurately capture the overall shape and content of clouds in both night-time and day-time, but the details of the images are not clear and precise enough. As for the results of BSANet, some details such as the boundary and the middle-size patch of the source image can be captured more accurately. In terms of our biggest configuration, BSANet-large, the overall shape and details can both be predicted precisely. For example, in two night-time images, the small patch of cloud in the red square is completely captured compared to our other two networks with smaller model sizes.
<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Qualitative%20visualization.png" width="700"/></div>
