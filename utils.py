import datetime
import functools
import os
import numpy as np
import torch


def timestamp(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        print("%s | " % datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"), end="")
        # print("%s | " % time.asctime(), end="")
        return func(*args, **kwargs)
    return decorated


def timer(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        start = datetime.datetime.now()
        res = func(*args, **kwargs)
        end = datetime.datetime.now()
        print("Time costs: " + ("%s" % (end - start)).split('.')[0])
        return res
    return decorated


@timestamp
def tprint(*args, **kwargs):
    print(*args, **kwargs)


def get_batch_PSNR(batch_img, batch_ground_truth):
    # batch_img: torch.tensor [-1, 3, height, width]
    assert (len(batch_img.shape) == 4)
    MAX_PIXEL = 1. if batch_img[0, 0, 0, 0] > 1 else 255.
    mse = torch.mean((batch_img - batch_ground_truth) ** 2, dim=[1, 2, 3])
    psnr = torch.mean(10 * torch.log10(MAX_PIXEL / mse))
    return psnr


def get_free_gpu():
    """
    取闲置的gpu号
    return: list
    """
    command = os.popen("nvidia-smi -q -d PIDS | grep Processes")
    lines = command.read().split("\n")  # 如果显卡上有进程那么这一行只会有一个Processes
    free_gpu = []
    for i in range(len(lines)):
        if "None" in lines[i]:
            free_gpu.append(str(i))
    return free_gpu