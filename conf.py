import os
import torch
import platform

from utils import get_free_gpu


class config_general():
    def __init__(self):
        self.seed = 1234
        self.EPOCH = 100
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-5
        self.BATCH_SIZE_TRAIN = 50
        self.BATCH_SIZE_TEST = 50
        self.HIDDEN_SIZE = 200

        if platform.system() == "Darwin":
            self.DATA_ROOT_PATH = "/Users/didi/Desktop/Coldog/Github/Spatial-RNN/MSCOCO/"
            self.MODEL_PATH = "/Users/didi/Desktop/Coldog/Github/Spatial-RNN/model/"
        elif platform.system() == "Linux":
            self.DATA_ROOT_PATH = "/home/amax/data/val2017/"
            # self.DATA_ROOT_PATH = "/home/amax/data/val2017_tiny/"
            # self.DATA_ROOT_PATH = "/data/data_tune/"
            self.MODEL_PATH = self.DATA_ROOT_PATH + "model/"
        self.MODEL_NAME = "model.pkl"

        # cuda option
        if torch.cuda.is_available():
            self.device = "cuda"
            try:
                self.device_index = os.environ["CUDA_VISIBLE_DEVICES"]  # 限定device
            except:
                self.device_index = ','.join(get_free_gpu())            # 没有限定device, 取所有空间device.
        else:
            self.device = "cpu"
            self.device_index = "-1"                                    # -1表示没有gpu device

    def update(self, args):
        pass