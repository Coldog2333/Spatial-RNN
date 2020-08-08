# Spatial-RNN

It is the repository of pytorch implementation of the ECCV 2016 paper: 
Learning Recursive Filters for Low-Level Vision via a Hybrid Neural Network.

The official implementation (Caffe version) can be accessed from 
[here](https://github.com/Liusifei/caffe-lowlevel).

## Introduction
This paper proposed a hybrid neural network incorporating several spatially variant recurrent neural networks (RNNs) 
as equivalents of a group of distinct recursive filters for each pixel. With the spatial RNN, we can tackle a lot of 
low-level task including Edge-preserving Smoothing, Denoising, Inpainting and color interpolation.

Although the spatial RNN introduce a deep CNN, however, the deep CNN only learns regulations of recurrent propagation 
to guide recurrent propagation over an entire image. That is to say, the deep CNN is not responsible 
for extracting complicated image features, so that it doesn't require convolutional layer with large channels, 
which makes it lighter for Image Processing. Therefore, the spatial RNN can be faster than the traditional CNN methods.

## Prerequisites
### PyTorch
Our implementation is based on PyTorch 1.4.0 ([https://pytorch.org/](https://pytorch.org/)) .

### matplotlib
For loading and saving images. Version: 2.2.2

### CUDA \[optional\]
CUDA is suggested ([https://developer.nvidia.com/cuda-toolkit]https://developer.nvidia.com/cuda-toolkit) for fast inference. 
The demo code is still runnable without CUDA, but much slower.

## Installation
Our current release has been tested on Ubuntu 16.04 LTS.

### Clone the repository
```shell script
git clone https://github.com/Coldog2333/Spatial-RNN.git
```

### Install some required packages

### Download tiny MSCOCO dataset (1% of MSCOCO)
You would like to have a quick start to understand how Spatial-RNN does.
Then you can download the tiny MSCOCO dataset and have a try. 
The tiny MSCOCO dataset is chosen randomly from 
the origin [MSCOCO 2017 dev dataset](http://images.cocodataset.org/zips/val2017.zip).

You can download the tiny-MSCOCO from [here](http://coldog.cn/dataset/tiny-MSCOCO.zip) and unzip the data.
Or you can just simply run the shell script like

```shell script
sh download_dataset.sh
```

### Preprocessing
After downloading and unziping the dataset, you should do some preprocessing to generate the data for training and testing.

Here provided a preprocessing script for generate 96x96 image patches as the paper suggested.
And it can also generate the corresponding patches with white Gaussian noise.

```shell script
python3 preprocessing.py --command generate_train_test_set --augment 10
```

## Train
```shell script
python3 main.py
```

## References
1. Liu, Sifei, Jinshan Pan, and Ming-Hsuan Yang. "Learning recursive filters for low-level vision via a hybrid neural network." European Conference on Computer Vision. Springer, Cham, 2016.