import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional
import argparse


# --- config ---
config = conf.config_smn()

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", default="ubuntu", type=str, help="The Name of the Corpus.")
parser.add_argument("-n", "--name", default=config.MODEL_NAME, type=str, help="Model Name")
parser.add_argument("-e", "--epoch", default=config.EPOCH, type=int, help="epoch num")
parser.add_argument("-t", "--turn", default=config.MAX_TURN, type=int, help="max turn")
parser.add_argument("-d", "--device", default=config.device_index, type=str, help="device No. e.g. \"0,1,2,3\"")
parser.add_argument("-r", "--root", default=config.DATA_ROOT_PATH, type=str, help="root path of data.")
args = parser.parse_args()

config.update(args=args)

corpus = config.corpus

# --- dump ---
dump_string = "-" * 15 + " dump begin " + "-" * 15 + "\n"
with open("./run.py") as f:
    dump_string += f.read() + "-" * 15 + " dump finished " + "-" * 15 + "\n\n" + "-" * 15 + " dump begin " + "-" * 15 + "\n"
with open("./network.py") as f:
    dump_string += f.read() + "-" * 15 + " dump finished " + "-" * 15 + "\n\n" + "-" * 15 + " dump conf  " + "-" * 15 + "\n"
for key in config.__dict__.keys():
    dump_string += "  %s : %s\n" % (key, config.__dict__[key])
dump_string += "-" * 15 + " dump finished " + "-" * 15 + "\n\n"
print(dump_string)
# ------------
if config.USE_BERT:
    bert_tokenizer, bert_model, vocab_id_map, id_vocab_map = utils.init_bert(config.BERT_PATH)
    w2v_matrix = None
else:
    w2v_matrix, vocab_id_map, id_vocab_map = utils.init_w2v_matrix(os.path.join(config.DATA_ROOT_PATH, config.corpus, "%s.200d.word2vec" % config.corpus))
    bert_tokenizer, bert_model = None, None
# w2v_matrix = np.random.rand(36132, 200)  # for debug

dataset_train = dataset_dialogue(file=os.path.join(config.DATA_ROOT_PATH, config.corpus, "train.txt"),
                                 vocab_id_map=vocab_id_map,
                                 answer_num=config.ANSWER_NUM_TRAIN,
                                 max_turn=config.MAX_TURN,
                                 max_seq_len=config.MAX_SEQ_LEN,
                                 shuffle_answer=True)
dataset_test = dataset_dialogue(file=os.path.join(config.DATA_ROOT_PATH, config.corpus, "test.txt"),
                                vocab_id_map=vocab_id_map,
                                answer_num=config.ANSWER_NUM_TEST,
                                max_turn=config.MAX_TURN,
                                max_seq_len=config.MAX_SEQ_LEN,
                                shuffle_answer=True)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, num_workers=16)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=config.BATCH_SIZE_TEST, shuffle=True, num_workers=16)

net = SMN(config=config, w2v_matrix=w2v_matrix, bert_tokenizer=bert_tokenizer, bert_model=bert_model).to(config.device)
if len(config.device_index.split(',')) > 1:
    net = torch.nn.DataParallel(net, device_ids=[int(index) for index in config.device_index.split(',')])
optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
loss_function = torch.nn.BCELoss()
max_test_acc = 0

def train_step(net, loss_function, optimizer, c, r, y, c_l, r_l, turn):
    logits = net(c, r, c_l, r_l, turn)

    loss = loss_function(logits.float(), target=y.squeeze().float())  # for BCELoss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scores = logits  # for BCELoss

    pred = torch.argmax(scores, dim=-1)
    y = torch.argmax(y, dim=-1)
    acc = torch.sum(pred == y).float() / config.BATCH_SIZE_TRAIN
    return loss, acc

#
# def test(net, dataloader_test):
#     net.eval()
#     test_acc = []
#     for step, (c, r, y, c_l, r_l, turn) in enumerate(dataloader_test):
#         if config.device == "cuda":
#             c, r, y = [item.cuda() for item in (c, r, y)]
#
#         logits = net(c, r, c_l, r_l, turn)
#
#         scores = logits   # for BCELoss
#
#         pred = torch.argmax(scores, dim=-1)
#         y = torch.argmax(y, dim=-1)
#         acc = torch.sum(pred == y).float() / config.BATCH_SIZE_TEST
#         # acc = torch.sum(torch.logical_not(torch.logical_xor(pred.bool(), y.bool())).float()) / config.BATCH_SIZE_TEST
#
#         tprint("test step:%s, acc=%.2f%%" % (step, acc.item() * 100))
#         test_acc.append(acc.item())
#
#     tprint("Total acc=%.2f%%" % (np.mean(test_acc) * 100))
#     net.train()
#     return test_acc


Loss_Curve = []
# Gradual_Learning = [1, 5, 10] + [config.MAX_TURN] * (config.EPOCH - 3)
for epoch in range(config.EPOCH):
    tprint("[EPOCH: %s/%s]" % (epoch, config.EPOCH))
    losses = 0
    # train mode ON
    net.train()

    train_acc = []
    for step, (c, r, y, c_l, r_l, turn) in enumerate(dataloader_train):
        if config.device == "cuda":
            c, r, y = [item.cuda() for item in (c, r, y)]

        # integrated process #
        # turn = -torch.nn.functional.threshold(-turn, -Gradual_Learning[epoch], -Gradual_Learning[epoch])
        loss, acc = train_step(net, loss_function, optimizer, c, r, y, c_l, r_l, turn)

        losses += loss.item()
        # tprint("train step:%s, acc=%.2f%%" % (step, acc.item() * 100))
        train_acc.append(acc.item())
        tprint("Processed %.2f%% samples...\r" % (step / dataloader_train.__len__() * 100), end="")
    # tprint("Loss: %.2f" % (losses / step))
    tprint("Processed 100% samples.")
    tprint("Total acc=%.2f%%" % (np.mean(train_acc) * 100))
    Loss_Curve.append(losses)

    # test
    net.eval()
    # Mark_scores = []
    # Mark_preds = []
    # Mark_labels = []
    test_acc = []
    for step, (c, r, y, c_l, r_l, turn) in enumerate(dataloader_test):
        if config.device == "cuda":
            c, r, y = [item.cuda() for item in (c, r, y)]

        logits = net(c, r, c_l, r_l, turn)
        scores = logits   # for BCELoss

        pred = torch.argmax(scores, dim=-1)
        y = torch.argmax(y, dim=-1)
        acc = torch.sum(pred == y).float() / config.BATCH_SIZE_TEST

        tprint("test step:%s, acc=%.2f%%" % (step, acc.item() * 100))
        test_acc.append(acc.item())

        # for i in range(config.BATCH_SIZE_TEST):
        #     for j in range(config.ANSWER_NUM_TEST):
        #         Mark_scores.append(scores.view(-1)[i * config.ANSWER_NUM_TEST + j].item())
        #         Mark_preds.append(pred.view(-1)[i].item())
        #         Mark_labels.append(y.view(-1)[i].item())
    tprint("Total acc=%.2f%%" % (np.mean(test_acc) * 100))
    # test_acc = test(net, dataloader_test, loss_function)

    if np.mean(test_acc) > max_test_acc:
        # save
        max_test_acc = np.mean(test_acc)
        net.cpu()
        if not os.path.exists(os.path.join(config.MODEL_PATH)):
            os.mkdir(os.path.join(config.MODEL_PATH))
        torch.save(net.state_dict(), os.path.join(config.MODEL_PATH, config.MODEL_NAME))
        if config.device == "cuda":
            net.cuda()
        # Best = [losses / step, np.mean(train_acc) * 100, np.mean(test_acc) * 100]
        # save score_file
        # with open(os.path.join(config.MODEL_PATH, config.MODEL_NAME[:-4] + "_scorefile.txt"), "w") as writer:
        #     for i in range(len(Mark_labels)):
        #         writer.write("%s,%s,%s\n" % (Mark_scores[i], Mark_preds[i], Mark_labels[i]))

        tprint("Save best model!\n")
