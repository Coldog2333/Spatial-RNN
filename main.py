import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional
import matplotlib.pyplot as plt
import argparse

from data_provider import cv_dataset
from network import Spatial_RNN
from utils import tprint
from conf import config_general

# --- config ---
config = config_general()

# --- dump ---


# ------------
dataset_train = cv_dataset(data_dir=os.path.join(config.DATA_ROOT_PATH, "train_preprocessed/"),
                              ground_truth_dir=os.path.join(config.DATA_ROOT_PATH, "train_generated/"))
dataset_test = cv_dataset(data_dir=os.path.join(config.DATA_ROOT_PATH, "test_preprocessed/"),
                              ground_truth_dir=os.path.join(config.DATA_ROOT_PATH, "test_generated/"))

dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=config.BATCH_SIZE_TEST, shuffle=True)

net = Spatial_RNN(config).to(config.device)
optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
loss_function = torch.nn.MSELoss()
max_test_acc = 0.

Loss_Curve = []


def visualization(net, input_img, out_name):
    net.eval()
    output_img = net(input_img)
    output_img = output_img.squeeze().permute(2, 0, 1).detach().numpy()
    plt.imsave(out_name, output_img)


for epoch in range(config.EPOCH):
    tprint("[EPOCH: %s/%s]" % (epoch + 1, config.EPOCH))
    losses = 0
    # train mode ON
    net.train()

    train_acc = []
    for step, (img, target) in enumerate(dataloader_train):
        if config.device == "cuda":
            img, target = [item.cuda() for item in (img, target)]
        # print(img.shape)
        new_img = net(img)

        loss = loss_function(target.float(), new_img.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        tprint("Processed %.2f%% samples...\r" % (step / dataloader_train.__len__() * 100), end="")
    tprint("Loss: %.2f" % (losses / (step + 1)))
    tprint("Processed 100% samples.")
    Loss_Curve.append(losses)

    # test
    net.eval()
    test_loss = 0
    for step, (img, target) in enumerate(dataloader_test):
        if config.device == "cuda":
            img, target = [item.cuda() for item in (img, target)]

        new_img = net(img)

        loss = loss_function(target.float(), new_img.float())

        test_loss += loss.item()
        tprint("Processed %.2f%% samples...\r" % (step / dataloader_test.__len__() * 100), end="")
    tprint("Loss: %.2f" % (test_loss / (step + 1)))

    visualization(net, dataset_test.img_list[0, :, :, :].unsqueeze(0), "./out.jpg")

    # if np.mean(test_acc) > max_test_acc:
    #     # save
    #     max_test_acc = np.mean(test_acc)
    #     net.cpu()
    #     if not os.path.exists(os.path.join(config.MODEL_PATH)):
    #         os.mkdir(os.path.join(config.MODEL_PATH))
    #     torch.save(net.state_dict(), os.path.join(config.MODEL_PATH, config.MODEL_NAME))
    #     if config.device == "cuda":
    #         net.cuda()
    #
    #     tprint("Save best model!\n")
