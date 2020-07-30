import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import config_general

class DeepCNN(torch.nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.pooling3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.pooling4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=32+32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.conv8 = nn.Conv2d(in_channels=32+32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.conv9 = nn.Conv2d(in_channels=32+16, out_channels=64, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pooling1(x1)

        x2 = F.relu(self.conv2(x2))
        x3 = self.pooling2(x2)

        x3 = F.relu(self.conv3(x3))
        x4 = self.pooling3(x3)

        x4 = F.relu(self.conv4(x4))
        x5 = self.pooling4(x4)

        x5 = F.relu(self.conv5(x5))

        x6 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.relu(self.conv6(torch.cat([x6, x4], dim=1)))

        x7 = F.interpolate(x6, scale_factor=2, mode="bilinear", align_corners=False)
        x7 = F.relu(self.conv7(torch.cat([x7, x3], dim=1)))

        x8 = F.interpolate(x7, scale_factor=2, mode="bilinear", align_corners=False)
        x8 = F.relu(self.conv8(torch.cat([x8, x2], dim=1)))

        x9 = F.interpolate(x8, scale_factor=2, mode="bilinear", align_corners=False)
        x9 = F.relu(self.conv9(torch.cat([x9, x1], dim=1)))

        return x9


class Spatial_RNN(nn.Module):
    def __init__(self, config):
        super(Spatial_RNN, self).__init__()
        self.config = config
        self.DCNN = DeepCNN()
        self.LRNN_in_conv = nn.Conv2d(in_channels=15, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.LRNN_out_conv = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), padding=(1, 1))

    def LRNN(self, X, P, direction):
        # X: [-1, 16, 96, 96]
        # P: [-1, 16, 96, 96]
        batch_size = X.shape[0]
        n = X.shape[2]
        batch_H = []
        h = torch.zeros(size=(16, 96)).to(self.config.device)
        for i in range(batch_size):
            H = []
            for k in range(n):
                if direction == "l2r":
                    h = (1 - P[i, :, :, k]) * X[i, :, :, k] + P[i, :, :, k] * h
                elif direction == "r2l":
                    h = (1 - P[i, :, :, n - k - 1]) * X[i, :, :, n - k - 1] + P[i, :, :, n - k - 1] * h
                elif direction == "t2b":
                    h = (1 - P[i, :, k, :]) * X[i, :, k, :] + P[i, :, k, :] * h
                elif direction == "b2t":
                    h = (1 - P[i, :, n - k - 1, :]) * X[i, :, n - k - 1, :] + P[i, :, n - k - 1, :] * h
                else:
                    raise AssertionError("Unknown direction: %s." % direction)
                H.append(h)
            batch_H.append(torch.stack(H, dim=1))
        batch_H = torch.stack(batch_H, dim=0)
        return batch_H

    def forward(self, img):
        feature_map = self.DCNN(img)
        LRNN_input = self.LRNN_in_conv(img)

        directions = ["l2r", "r2l", "t2b", "b2t"]
        LRNN_output = []
        for i in range(4):
            LRNN_output.append(self.LRNN(LRNN_input, feature_map[:, i*16:(i+1)*16, :, :], direction=directions[i]))
        LRNN_output = torch.stack(LRNN_output, dim=1)
        LRNN_output = torch.max(LRNN_output, dim=1)[0]

        output = self.LRNN_out_conv(LRNN_output)
        return output


if __name__ == "__main__":
    config = config_general()
    net = DeepCNN()
    fake_img = torch.rand((2, 15, 96, 96))
    dcnn_out = net(fake_img)
    # print(dcnn_out.shape)

    spatial_rnn = Spatial_RNN(config=config)
    print(spatial_rnn(fake_img).shape)