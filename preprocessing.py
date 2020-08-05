import os
import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt

from conf import config_general


def crop(img, size, augment_time=1):
    # img: numpy.array [height, width, 3]
    height, width, _ = img.shape
    height_mark = np.random.randint(0, height - size[0] - 1, size=augment_time)
    width_mark = np.random.randint(0, width - size[1] - 1, size=augment_time)

    imgs = []
    for i in range(augment_time):
        cropped_img = img[height_mark[i]:height_mark[i] + size[0], width_mark[i]:width_mark[i] + size[1], :]
        imgs.append(cropped_img)
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
            if len(img.shape) != 3:  # is a grep image.
                continue
            img_list.extend(crop(img, size=(96, 96), augment_time=augment_time))
        else:
            print("Skip %s." % path)

    if not os.path.exists(generated_data_path):
        os.mkdir(generated_data_path)

    for i in range(len(img_list)):
        # avoid an unexpected bug
        plt.imsave(os.path.join(generated_data_path, "im%s.jpg" % (i+1)), img_list[i])


def preprocess(raw_data_path, preprocessed_data_path, image_format="jpg"):
    if not os.path.exists(preprocessed_data_path):
        os.mkdir(preprocessed_data_path)

    for path in os.listdir(raw_data_path):
        if image_format in path:
            img = plt.imread(os.path.join(raw_data_path, path))
            plt.imsave(os.path.join(preprocessed_data_path, path), add_noise(img))
        else:
            print("Skip %s." % path)


def split_train_test(data_root_path):
    # TODO: bug exists
    pos_num, neg_num = 4000, 1000
    files = os.listdir(data_root_path)

    if not os.path.exists(os.path.join(data_root_path, "train")):
        os.mkdir(os.path.join(data_root_path, "train"))
    if not os.path.exists(os.path.join(data_root_path, "test")):
        os.mkdir(os.path.join(data_root_path, "test"))

    for i in range(pos_num + neg_num):
        if i < pos_num:
            shutil.move(os.path.join(data_root_path, files[i]), os.path.join(data_root_path, "train", files[i]))
        else:
            shutil.move(os.path.join(data_root_path, files[i]), os.path.join(data_root_path, "test", files[i]))
    print("Splitting done.")


def generate_train_test_set(root_path, augment_time=10):
    # split_train_test(root_path)
    train_data = root_path + "train/"
    train_generated_data = root_path + "train_generated/"
    train_preprocessed_data = root_path + "train_preprocessed/"
    test_data = root_path + "test/"
    test_generated_data = root_path + "test_generated/"
    test_preprocessed_data = root_path + "test_preprocessed/"

    augment(train_data, train_generated_data, image_format="jpg", augment_time=augment_time)
    preprocess(train_generated_data, train_preprocessed_data, image_format="jpg")

    augment(test_data, test_generated_data, image_format="jpg", augment_time=augment_time)
    preprocess(test_generated_data, test_preprocessed_data, image_format="jpg")


def generate_inference_set(root_path):
    image_format = "jpg"
    data_path = root_path + "test/"
    inference_preprocessed_data = root_path + "inference_preprocessed/"

    if not os.path.exists(inference_preprocessed_data):
        os.mkdir(inference_preprocessed_data)

    for file in os.listdir(data_path):
        if image_format in file:
            img = plt.imread(os.path.join(data_path, file))
            plt.imsave(os.path.join(inference_preprocessed_data, file), add_noise(img))
        else:
            print("Skip %s." % file)

    print("Done")


if __name__ == "__main__":
    config = config_general()

    root_path = config.DATA_ROOT_PATH

    generate_inference_set(root_path)