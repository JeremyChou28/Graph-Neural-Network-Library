import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import GraphSAGE
from dataset import CoraData
from sampling import multihop_sampling
from config import DEVICE
from collections import namedtuple

INPUT_DIM = 1433
HIDDEN_DIM = [128, 7]
NUM_NEIGHBORS_LIST = [20, 20]
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 16
EPOCHS = 30
NUM_BATCH_PER_EPOCH = 10
LEARNING_RATE = 0.001
print(DEVICE)
Data = namedtuple(
    'Data',
    ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])

data = CoraData(data_root='../../GCN/Cora', rebuild=False).data
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

train_index = np.where(data.train_mask)[0]
train_label = data.y
test_index = np.where(data.test_mask)[0]
model = GraphSAGE(input_dim=INPUT_DIM,
                  hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
# print(model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)


def train():
    model.train()
    acc_list = []
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index,
                                               size=(BTACH_SIZE, ))
            batch_src_label = torch.from_numpy(
                train_label[batch_src_index]).long().to(DEVICE)
            batch_sampling_result = multihop_sampling(batch_src_index,
                                                      NUM_NEIGHBORS_LIST,
                                                      data.adjacency_dict)
            batch_sampling_x = [
                torch.from_numpy(x[idx]).float().to(DEVICE)
                for idx in batch_sampling_result
            ]
            batch_train_logits = model(batch_sampling_x)
            loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(
                e, batch, loss.item()))
        acc = test()
        acc_list.append(acc)
    return acc_list


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling(test_index,
                                                 NUM_NEIGHBORS_LIST,
                                                 data.adjacency_dict)
        test_x = [
            torch.from_numpy(x[idx]).float().to(DEVICE)
            for idx in test_sampling_result
        ]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)
        return accuarcy


if __name__ == '__main__':
    his = train()
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(his)),
             his,
             c=np.array([255, 71, 90]) / 255.,
             label='test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc=0)
    plt.title('acc')
    plt.savefig("../assets/acc.png")
    plt.show()
