# Spatial-RNN

It is the repository of pytorch implementation of the ECCV 2016 paper: 
Learning Recursive Filters for Low-Level Vision via a Hybrid Neural Network.

The official implementation (Caffe version) can be accessed from 
[here](https://github.com/Liusifei/caffe-lowlevel)

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

### Preprocessing

## Train
```shell script
python3 main.py
```

## References
1. Liu, Sifei, Jinshan Pan, and Ming-Hsuan Yang. "Learning recursive filters for low-level vision via a hybrid neural network." European Conference on Computer Vision. Springer, Cham, 2016.