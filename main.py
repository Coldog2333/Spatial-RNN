import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional
import argparse

from data_provider import train_dataset
from network import Spatial_RNN
from utils import tprint
from conf import config_general

# --- config ---
config = config_general()

# --- dump ---


# ------------
dataset_train = train_dataset(data_dir="./preprocessed_data", ground_truth_dir="./generated_data")

dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, num_workers=1)

net = Spatial_RNN().to("cpu")
optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
loss_function = torch.nn.MSELoss()
max_test_acc = 0.

Loss_Curve = []

for epoch in range(config.EPOCH):
    tprint("[EPOCH: %s/%s]" % (epoch, config.EPOCH))
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
    test_acc = []
    for step, (img, target) in enumerate(dataloader_train):
        break
        if config.device == "cuda":
            img, target = [item.cuda() for item in (img, target)]

        new_img = net(img)

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
