# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/12/24 17:32
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import GCN, ChebNet, GAT
from metrics import MAE, MAPE, RMSE
from data_loader import get_loader
from visualize_dataset import show_pred

seed = 2020
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# data
train_loader, test_loader = get_loader('PEMS04')

gcn = GCN(in_c=6, hid_c=6, out_c=1)  # 输入层节点数，隐藏层节点数，输出层节点数
chebnet = ChebNet(in_c=6, hid_c=6, out_c=1,
                  K=1)  # 输入层节点数，隐藏层节点数，输出层节点数，切比雪夫多项式阶数
gat = GAT(in_c=6, hid_c=6, out_c=1, n_heads=6)  # 输入层节点数，隐藏层节点数，输出层节点数，注意力头数
devicenum = 2
device = torch.device(
    "cuda:{}".format(devicenum) if torch.cuda.is_available() else "cpu")
models = [chebnet.to(device), gcn.to(device), gat.to(device)]

all_predict_values = []
checkpoint_files = ['chebnet.pth', 'gcn.pth', 'gat.pth']
criterion = nn.MSELoss().to(device)
for i in range(len(models)):
    model = models[i]
    torch.load(model, '../checkpoints/PEMS04/' + checkpoint_files[i])  # 保存整个网络
    total_loss = 0.0
    num = 0
    all_predict_value = 0
    all_y_true = 0
    for data in test_loader:
        data['graph'], data['flow_x'], data['flow_y'] = data['graph'].to(
            device), data['flow_x'].to(device), data['flow_y'].to(device)
        predict_value = model(data, devicenum)
        if num == 0:
            all_predict_value = predict_value
            all_y_true = data["flow_y"]
        else:
            all_predict_value = torch.cat(
                [all_predict_value, predict_value], dim=0)
            all_y_true = torch.cat([all_y_true, data["flow_y"]], dim=0)
        loss = criterion(predict_value, data["flow_y"])
        num += 1
    epoch_mae = MAE(all_y_true.cpu(), all_predict_value.cpu())
    epoch_rmse = RMSE(all_y_true.cpu(), all_predict_value.cpu())
    epoch_mape = MAPE(all_y_true.cpu(), all_predict_value.cpu())
    print(
        "Test Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}"
            .format(10 * total_loss / (len(test_loader.dataset) / 64),
                    epoch_mae, epoch_rmse, epoch_mape))

    all_predict_values.append(all_predict_value.cpu())
show_pred(test_loader, all_y_true.cpu(), all_predict_values)
