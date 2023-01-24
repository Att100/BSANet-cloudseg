## BSANet for Real-time Cloud Segmentation
With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript: 

> Y. Li, H. Wang, S. Wang, Y. H. Lee, S. Dev. “BSANet: A Bilateral Segregation and Aggregation Network for Real-time Sky/Cloud Segmentation”, IEEE Transactions on Geoscience and Remote Sensing, under review

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.

### Executive summary
Segmenting clouds from intensity images is an essential research topic at the intersection of atmospheric science and computer vision, which plays a vital role in weather forecasts, environmental monitoring, and climate evolution analysis. The ground-based sky/cloud image segmentation can help to extract the cloud from the original image and analyze the shape or additional features. In this paper, we introduced a novel sky/cloud segmentation network named Bilateral Segregation and Aggregation Network (BSANet) with 16.37 MBytes, which can reduce 70.68% of model size and achieve almost the same performance as the state-of-the-art method shown as the figure below. After the deployment via TensorRT, BSANet-large configuration can achieve 392 fps in FP16 while BSANet-lite can achieve 1390 fps, which all exceed real-time standards. Additionally, we proposed a novel and fast pre-training strategy for sky/cloud segmentation which can improve the accuracy of segmentation when ImageNet pre-training is not available.

<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Accuracy%20and%20model%20size%20comparison.png" width="700"/></div>

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

### Data
* `./dataset/`: This folder contains the full SWINySEG dataset.

### Model
* `BSANet Architecture.png`: It shows the overall architecture of BSANet and BSAM. (a) illustrate the overall pipeline of BSANet and (b) indicate the detailed design of bilateral segregation and aggregation module (BSAM). The procedure between the output of the model and the segmentation mask has been omitted in this figure.


<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Architecture%20of%20BSANet%20and%20BSAM.png" width="700"/></div>

<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Schematic%20diagram%20of%20our%20proposed%20SWINySEG-based%20pre-training.png" width="700"/></div>

<div align=center><img src="https://github.com/Att100/BSANet-cloudseg/blob/main/figures/Qualitative%20visualization.png" width="700"/></div>

### Training

- help

    ```
    python train.py -h

    usage: train.py [-h] [--model_tag MODEL_TAG] [--k K] [--batch_size BATCH_SIZE] [--lr LR] [--lr_decay LR_DECAY] [--aux AUX] [--epochs EPOCHS]
                    [--dataset_split DATASET_SPLIT] [--dataset_path DATASET_PATH] [--eval_interval EVAL_INTERVAL]

    optional arguments:
    -h, --help            show this help message and exit
    --model_tag MODEL_TAG
                            the tag of model (default: ucloudnet_k_2_aux_lr_decay)
    --k K                 the k value of model (default: 2)
    --batch_size BATCH_SIZE
                            batchsize for model training (default: 16)
    --lr LR               the learning rate for training (default: 1e-3)
    --lr_decay LR_DECAY   enable learning rate decay when training, [1, 0] (default: 1)
    --aux AUX             enable deep supervision when training, [1, 0] (default: 1)
    --epochs EPOCHS       number of training epochs (default: 100)
    --dataset_split DATASET_SPLIT
                            split of SWINySEG dataset, ['all', 'd', 'n'] (default: all)
    --dataset_path DATASET_PATH
                            path of training dataset (default: ./dataset/SWINySEG)
    --eval_interval EVAL_INTERVAL
                            interval of model evaluation during training (default: 5)

    ```

- experiments


### Testing

```
# follow instructions in notebook.ipynb
```

### Results


