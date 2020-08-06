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

        x6 = F.interpolate(x5, mode="bilinear", align_corners=False, size=x4.shape[-2:])
        x6 = F.relu(self.conv6(torch.cat([x6, x4], dim=1)))

        x7 = F.interpolate(x6, mode="bilinear", align_corners=False, size=x3.shape[-2:])
        x7 = F.relu(self.conv7(torch.cat([x7, x3], dim=1)))

        x8 = F.interpolate(x7, mode="bilinear", align_corners=False, size=x2.shape[-2:])
        x8 = F.relu(self.conv8(torch.cat([x8, x2], dim=1)))

        x9 = F.interpolate(x8, mode="bilinear", align_corners=False, size=x1.shape[-2:])
        x9 = self.conv9(torch.cat([x9, x1], dim=1))

        return x9


class Spatial_RNN(nn.Module):
    def __init__(self, config):
        super(Spatial_RNN, self).__init__()
        self.config = config
        self.DCNN = DeepCNN()
        self.LRNN_in_conv = nn.Conv2d(in_channels=15, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.LRNN_out_conv = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), padding=(1, 1))

    def LRNN_operation(self, X, P, direction):
        # X: [-1, 16, 96, 96]
        # P: [-1, 16, 96, 96]
        batch_size = X.shape[0]
        n = X.shape[2] if "l" in direction else X.shape[3]
        m = X.shape[3] if "l" in direction else X.shape[2]
        batch_H = []
        h = torch.zeros(size=(batch_size, 16, n)).to(self.config.device)
        for k in range(m):
            if direction == "l2r":
                h = (1 - P[:, :, :, k]) * X[:, :, :, k] + P[:, :, :, k] * h
            elif direction == "r2l":
                h = (1 - P[:, :, :, m - k - 1]) * X[:, :, :, m - k - 1] + P[:, :, :, m - k - 1] * h
            elif direction == "t2b":
                h = (1 - P[:, :, k, :]) * X[:, :, k, :] + P[:, :, k, :] * h
            elif direction == "b2t":
                h = (1 - P[:, :, m - k - 1, :]) * X[:, :, m - k - 1, :] + P[:, :, m - k - 1, :] * h
            else:
                raise AssertionError("Unknown direction: %s." % direction)
            batch_H.append(h)
        if "l" in direction:
            batch_H = torch.stack(batch_H, dim=3)
        else:
            batch_H = torch.stack(batch_H, dim=2)

        # for i in range(batch_size):
        #     H = []
        #     for k in range(m):
        #         if direction == "l2r":
        #             h = (1 - P[i, :, :, k]) * X[i, :, :, k] + P[i, :, :, k] * h
        #         elif direction == "r2l":
        #             h = (1 - P[i, :, :, m - k - 1]) * X[i, :, :, m - k - 1] + P[i, :, :, m - k - 1] * h
        #         elif direction == "t2b":
        #             h = (1 - P[i, :, k, :]) * X[i, :, k, :] + P[i, :, k, :] * h
        #         elif direction == "b2t":
        #             h = (1 - P[i, :, m - k - 1, :]) * X[i, :, m - k - 1, :] + P[i, :, m - k - 1, :] * h
        #         else:
        #             raise AssertionError("Unknown direction: %s." % direction)
        #         H.append(h)
        #     if "l" in direction:
        #         batch_H.append(torch.stack(H, dim=2))
        #     else:
        #         batch_H.append(torch.stack(H, dim=1))
        return batch_H

    def LRNN_layer(self, LRNN_input, feature_map):
        directions = ["l2r", "r2l", "t2b", "b2t"]
        LRNN_output = []

        # 2nd-order
        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:,  0:16, :, :], direction="l2r"))
        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:,  0:16, :, :], direction="r2l"))
        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:, 16:32, :, :], direction="t2b"))
        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:, 16:32, :, :], direction="b2t"))

        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:, 32:48, :, :], direction="l2r"))
        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:, 32:48, :, :], direction="r2l"))
        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:, 48:64, :, :], direction="t2b"))
        LRNN_output.append(self.LRNN_operation(LRNN_input, feature_map[:, 48:64, :, :], direction="b2t"))

        LRNN_output = torch.stack(LRNN_output, dim=1)
        LRNN_output = torch.max(LRNN_output, dim=1)[0]
        return LRNN_output

    def forward(self, img):
        feature_map = torch.tanh(self.DCNN(img))
        LRNN_input = self.LRNN_in_conv(img)

        LRNN_output = self.LRNN_layer(LRNN_input, feature_map)

        output = torch.sigmoid(self.LRNN_out_conv(LRNN_output))
        return output


if __name__ == "__main__":
    config = config_general()
    net = DeepCNN()
    fake_img = torch.rand((2, 15, 128, 256))
    dcnn_out = net(fake_img)
    # print(dcnn_out.shape)

    spatial_rnn = Spatial_RNN(config=config)
    print(spatial_rnn(fake_img).shape)