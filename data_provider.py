import copy
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from preprocessing import add_noise # for debug

def initialize_image(img):
    # input: img: torch.tensor, (-1, 3, 96, 96)
    pooled_img = copy.deepcopy(img)
    for scalar in [2, 4, 8, 16]:
        pooled_img = F.avg_pool2d(pooled_img, kernel_size=(2, 2))
        new_img = F.interpolate(pooled_img, scale_factor=scalar, mode="bilinear", align_corners=False)
        img = torch.cat([img, new_img], dim=1)
    return img


def load_img_from_dir(data_dir, image_format="jpg"):
    img_list = []
    count = 0
    for path in os.listdir(data_dir):
        if image_format in path:
            img = plt.imread(os.path.join(data_dir, path))
            img_list.append(img)
        else:
            print("Skip %s." % path)
        count += 1
        if count > 500:
            break
    return img_list


def plt_imread_batch(absolute_paths):
    img_list = []
    for absolute_path in absolute_paths:
        img = plt.imread(absolute_path)
        img_list.append(img)
    return img_list


def load_img_from_dir_multiprocessing(data_dir, image_format="jpg"):
    paths = os.listdir(data_dir)
    results = []
    pool = Pool(processes=cpu_count())
    load_batch = 10

    for i in range(int(len(paths) / load_batch) - 1):
        results.append(pool.apply_async(func=plt_imread_batch, args=([os.path.join(data_dir, path) for path in paths[i*load_batch:(i+1)*load_batch]])))
    results.append(pool.apply_async(func=plt_imread_batch, args=([os.path.join(data_dir, path) for path in paths[i:]])))

    pool.close()
    pool.join()
    img_list = []
    for i in range(len(results)):
        result = results[i].get()
        img_list.extend(result)
    return img_list


class cv_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ground_truth_dir):
        self.img_list = torch.Tensor(load_img_from_dir(data_dir)).permute(0, 3, 1, 2) / 255
        self.img_list = initialize_image(self.img_list)
        self.target_img_list = torch.Tensor(load_img_from_dir(ground_truth_dir)).permute(0, 3, 1, 2) / 255

    def __len__(self):
        return self.img_list.shape[0]

    def __getitem__(self, idx):
        return self.img_list[idx], self.target_img_list[idx]


class cv_dataset_inference(torch.utils.data.Dataset):
    def __init__(self, data_dir, ground_truth_dir):
        self.img_list = load_img_from_dir(data_dir)
        self.target_img_list = load_img_from_dir(ground_truth_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        torch_img = torch.Tensor(self.img_list[idx]).permute(2, 0, 1) / 255
        torch_target = torch.Tensor(self.target_img_list[idx]).permute(2, 0, 1) / 255
        return initialize_image(torch_img), torch_target


if __name__ == "__main__":
    img_path = "./data/im1.png"
    img = plt.imread(img_path)
    # img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

    # fake_img = torch.rand((5, 3, 96, 96))
    # print(initialize_image(img).shape)
    plt.imshow(img)
    plt.show()
    img = add_noise(img)
    plt.imshow(img)
    plt.show()

    # dataset = train_dataset("./generated_data")