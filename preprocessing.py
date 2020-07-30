import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def crop(img, size, augment_time=1):
    # img: numpy.array [height, width, 3]
    height, width, _ = img.shape
    height_mark = np.random.randint(0, height - size[0] - 1, size=augment_time)
    width_mark= np.random.randint(0, height - size[1] - 1, size=augment_time)

    imgs = []
    for i in range(augment_time):
        imgs.append(img[height_mark[i]:height_mark[i] + size[0], width_mark[i]:width_mark[i] + size[1], :])
    return imgs


def add_noise(img):
    if np.max(img) > 10:
        img = img / 255
    noise = np.random.normal(scale=0.1, size=img.shape)
    img = img + noise
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)
    return img


def augment(raw_data_path, generated_data_path, image_format, augment_time):
    img_list = []
    for path in os.listdir(raw_data_path):
        if image_format in path:
            img = plt.imread(os.path.join(raw_data_path, path))
            img_list.extend(crop(img, size=(96, 96), augment_time=augment_time))
        else:
            print("Skip %s." % path)

    for i in range(len(img_list)):
        plt.imsave(os.path.join(generated_data_path, "im%s.jpg" % (i+1)), img_list[i])


def preprocess(raw_data_path, preprocessed_data_path, image_format="jpg"):
    for path in os.listdir(raw_data_path):
        if image_format in path:
            img = plt.imread(os.path.join(raw_data_path, path))
            plt.imsave(os.path.join(preprocessed_data_path, path), add_noise(img))
        else:
            print("Skip %s." % path)


if __name__ == "__main__":
    raw_data_path = "./data/"
    generated_data_path = "./generated_data/"
    preprocessed_data_path = "./preprocessed_data/"
    augment_time = 200

    augment(raw_data_path, generated_data_path, image_format="png", augment_time=augment_time)
    preprocess(generated_data_path, preprocessed_data_path, image_format="jpg")