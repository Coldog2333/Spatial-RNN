import copy
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        if count >= 100:
            break
    return img_list


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ground_truth_dir):
        self.img_list = torch.Tensor(load_img_from_dir(data_dir)).permute(0, 3, 1, 2) / 255
        self.img_list = initialize_image(self.img_list)
        self.target_img_list = torch.Tensor(load_img_from_dir(ground_truth_dir)).permute(0, 3, 1, 2) / 255

    def __len__(self):
        return self.img_list.shape[0]

    def __getitem__(self, idx):
        return self.img_list[idx], self.target_img_list[idx]


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