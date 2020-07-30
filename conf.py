import os
import torch
import platform

from utils import get_free_gpu


class config_general():
    def __init__(self):
        self.EPOCH = 100
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 0.
        self.BATCH_SIZE_TRAIN = 10
        self.BATCH_SIZE_TEST = 10
        self.HIDDEN_SIZE = 200

        if platform.system() == "Darwin":
            self.DATA_ROOT_PATH = "/Users/didi/Desktop/Coldog/Media/generated_data/"
            self.MODEL_PATH = "/Users/didi/Desktop/Coldog/Media/model/"
        elif platform.system() == "Linux":
            self.DATA_ROOT_PATH = "/data/dstc8/"
            # self.DATA_ROOT_PATH = "/data/data_debug/"
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
        self.MODEL_NAME = args.name
        self.EPOCH = args.epoch
        self.device_index = args.device

        self.DATA_ROOT_PATH = args.root
        self.MODEL_PATH = self.DATA_ROOT_PATH + "model/"

        self.BATCH_SIZE_TRAIN *= len(self.device_index.split(','))
        self.BATCH_SIZE_TEST *= len(self.device_index.split(','))
