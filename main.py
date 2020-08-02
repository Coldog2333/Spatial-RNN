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
def dump(config):
    pass

# ------------
class Pipeline():
    def __init__(self, network_f, optimizer_f, loss_function_f, data_paths, config):
        self.traindata_path, self.testdata_path, self.devdata_path = data_paths
        self.network_f, self.optimizer_f, self.loss_function, self.config = network_f, optimizer_f, loss_function_f, config

        self.set_random_seed()

    def set_random_seed(self):
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.device == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

    def prepare_data(self):
        tprint("preparing data...")

        self.dataset_train = cv_dataset(data_dir=self.traindata_path["input"],
                                      ground_truth_dir=self.traindata_path["ground_truth"])
        self.dataset_test = cv_dataset(data_dir=self.testdata_path["input"],
                                      ground_truth_dir=self.testdata_path["ground_truth"])
        self.dataset_dev = None

        self.dataloader_train = torch.utils.data.DataLoader(dataset=self.dataset_train, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test, batch_size=config.BATCH_SIZE_TEST, shuffle=True)
        self.dataloader_dev = None

    def build_network(self):
        self.network = self.network_f(self.config).to(self.config.device)
        print(self.network)
        self.optimizer = self.optimizer_f(self.network.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)

    def train(self):
        Loss_Curve = []
        min_test_loss = 1e7
        for epoch in range(config.EPOCH):
            tprint("[EPOCH: %s/%s]" % (epoch + 1, config.EPOCH))
            losses = 0
            # train mode ON
            self.network.train()

            for step, (img, target) in enumerate(self.dataloader_train):
                if config.device == "cuda":
                    img, target = [item.cuda() for item in (img, target)]
                # print(img.shape)
                new_img = self.network(img)

                loss = self.loss_function(target.float(), new_img.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                tprint("Processed %.2f%% samples...\r" % (step / self.dataloader_train.__len__() * 100), end="")
            tprint("Loss: %.2f" % (losses / (step + 1)))
            tprint("Processed 100% samples.")
            Loss_Curve.append(losses)

            # test
            test_loss = self.evaluate(mode="test", epoch=epoch)
            tprint("Loss: %.2f" % (test_loss / (step + 1)))

            if step == 0:
                plt.imsave("./in.jpg", self.dataset_test.img_list[0, :3, :, :].permute(1, 2, 0).detach().cpu().numpy())
                plt.imsave("./ground_truth.jpg",
                           self.dataset_test.target_img_list[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy())
            self.visualization("./out_epoch%s.jpg" % (epoch + 1),
                          self.dataset_test.img_list[0, :, :, :].unsqueeze(0).to(config.device))

            if test_loss < min_test_loss:
                # save
                min_test_loss = test_loss
                self.network.cpu()
                if not os.path.exists(os.path.join(config.MODEL_PATH)):
                    os.mkdir(os.path.join(config.MODEL_PATH))
                torch.save(self.network.state_dict(), os.path.join(config.MODEL_PATH, config.MODEL_NAME))
                if config.device == "cuda":
                    self.network.cuda()

                tprint("Save best model!\n")

    def evaluate(self, mode="test", epoch=-1):
        self.network.eval()
        dataloader = self.dataloader_test if mode == "test" else self.dataloader_dev
        eval_loss = 0
        for step, (img, target) in enumerate(dataloader):
            if config.device == "cuda":
                img, target = [item.cuda() for item in (img, target)]

            new_img = self.network(img)

            loss = self.loss_function(target.float(), new_img.float())

            eval_loss += loss.item()
            tprint("Processed %.2f%% samples...\r" % (step / dataloader.__len__() * 100), end="")
        return eval_loss


    def save_model(self):
        self.network.cpu()
        if not os.path.exists(os.path.join(self.config.MODEL_PATH)):
            os.mkdir(os.path.join(self.config.MODEL_PATH))
        torch.save(self.network.state_dict(), os.path.join(self.config.MODEL_PATH, self.config.MODEL_NAME))
        if torch.cuda.is_available():
            self.network.cuda()
        tprint("Save best model!\n")

    def load_model(self):
        self.network.cpu()
        self.network.load_state_dict(torch.load(os.path.join(self.config.MODEL_PATH, self.config.MODEL_NAME)))
        if torch.cuda.is_available():
            self.network.cuda()
        tprint("Loaded model parameters!\n")

    def visualization(self, out_name, input_img):
        self.network.eval()
        output_img = self.network(input_img)
        output_img = output_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        plt.imsave(out_name, output_img)


if __name__ == "__main__":
    config = config_general()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config.update(args=args)

    traindata_path = {"input": os.path.join(config.DATA_ROOT_PATH, "train_preprocessed/"),
                      "ground_truth": os.path.join(config.DATA_ROOT_PATH, "train_generated/")}
    testdata_path = {"input": os.path.join(config.DATA_ROOT_PATH, "test_preprocessed/"),
                      "ground_truth": os.path.join(config.DATA_ROOT_PATH, "test_generated/")}
    devdata_path = "yes"

    dump(config)

    pipeline = Pipeline(network_f=Spatial_RNN,
                            optimizer_f=torch.optim.Adam,
                            loss_function_f=torch.nn.MSELoss(),
                            # loss_function_f=Distance_penalized_loss(config),
                            data_paths=[traindata_path, testdata_path, devdata_path],
                            config=config)

    pipeline.prepare_data()
    pipeline.build_network()
    pipeline.train()