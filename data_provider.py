import copy
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from preprocessing import add_noise  # for debug
from conf import config_general

def initialize_image(img):
    # input: img: torch.tensor, (-1, 3, 96, 96)
    original_size = img.shape
    pooled_img = copy.deepcopy(img)
    for scalar in [2, 4, 8, 16]:
        pooled_img = F.avg_pool2d(pooled_img, kernel_size=(2, 2))
        new_img = F.interpolate(pooled_img, mode="bilinear", align_corners=False, size=original_size[-2:])
        img = torch.cat([img, new_img], dim=1)
    return img


def load_img_from_dir(data_dir, image_format="jpg"):
    img_list = []
    path_list = []
    count = 0
    for path in os.listdir(data_dir):
        if image_format in path:
            img = plt.imread(os.path.join(data_dir, path))
            if len(img.shape) < 3:
                continue
            # # pad some images with odd size.
            # width, height, _ = img.shape
            # if width % 2 != 0:
            #     img = img[:width-1, :, :]
            # if height % 2 != 0:
            #     img = img[:width-1, :, :]

            img_list.append(img)
            path_list.append(path)
        else:
            print("Skip %s." % path)
        count += 1
        if count > 10:
            break
    return img_list, path_list


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
        self.img_list = torch.Tensor(load_img_from_dir(data_dir)[0]).permute(0, 3, 1, 2) / 255
        self.img_list = initialize_image(self.img_list)
        self.target_img_list = torch.Tensor(load_img_from_dir(ground_truth_dir)[0]).permute(0, 3, 1, 2) / 255

    def __len__(self):
        return self.img_list.shape[0]

    def __getitem__(self, idx):
        return self.img_list[idx], self.target_img_list[idx]


class cv_dataset_inference(torch.utils.data.Dataset):
    def __init__(self, data_dir, ground_truth_dir):
        self.img_list, _ = load_img_from_dir(data_dir)
        self.target_img_list, self.path_list = load_img_from_dir(ground_truth_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        torch_img = initialize_image(torch.Tensor(self.img_list[idx]).unsqueeze(0).permute(0, 3, 1, 2) / 255).squeeze()
        torch_target = torch.Tensor(self.target_img_list[idx]).permute(2, 0, 1) / 255
        path = self.path_list[idx]
        return torch_img, torch_target, path


if __name__ == "__main__":
    config = config_general()
    dataset_inference = cv_dataset_inference(data_dir=os.path.join(config.DATA_ROOT_PATH, "inference_preprocessed/"),
                                                  ground_truth_dir=os.path.join(config.DATA_ROOT_PATH, "test/"))

    dataloader_inference = torch.utils.data.DataLoader(dataset=dataset_inference, batch_size=1, shuffle=False)

    for step, (img, target, path) in enumerate(dataloader_inference):
        plt.imshow(target[0, :3, :, :].squeeze().permute(1, 2, 0).detach().cpu().numpy())
        plt.title(path[0])
        plt.show()