# @description:pytorch原生复现GCN模型
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/10/21 10:23
import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

Data = namedtuple(
    'Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = [
        "ind.cora.{}".format(name) for name in
        ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    ]

    def __init__(self, data_root="cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}/raw
                缓存数据路径: {data_root}/processed_cora.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [
            self.read_data(osp.join(self.data_root, "raw", name))
            for name in self.filenames
        ]

        # 测试test_index的形状（1000，），如果那里不明白可以测试输出一下矩阵形状
        print('test_index', test_index.shape)

        train_index = np.arange(y.shape[0])  # [0,...139] 140个元素
        print('train_index', train_index.shape)

        val_index = np.arange(y.shape[0],
                              y.shape[0] + 500)  # [140 - 640] 500个元素
        print('val_index', val_index.shape)

        sorted_test_index = sorted(test_index)  # #test_index就是随机选取的下标,排下序
        #         print('test_index',sorted_test_index)

        x = np.concatenate((allx, tx), axis=0)  # 1708 +1000 =2708 特征向量
        y = np.concatenate((ally, ty),
                           axis=0).argmax(axis=1)  # 把最大值的下标重新作为一个数组 标签向量

        x[test_index] = x[
            sorted_test_index]  # 打乱顺序,单纯给test_index 的数据排个序,不清楚具体效果
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]  # 2078

        train_mask = np.zeros(num_nodes, dtype=bool)  # 生成零向量
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[train_index] = True  # 前140个元素为训练集
        val_mask[val_index] = True  # 140 -639 500个
        test_mask[test_index] = True  # 1708-2708 1000个元素

        # 下面两句是我测试用的，本来代码没有
        # 是为了知道使用掩码后，y_train_mask 的形状，由输出来说是（140，）
        # 这就是后面划分训练集的方法
        y_train_mask = y[train_mask]
        print('y_train_mask', y_train_mask.shape)

        # 构建邻接矩阵
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x,
                    y=y,
                    adjacency=adjacency,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask)

    def maybe_download(self):
        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data("{}/{}".format(self.download_url, name),
                                   save_path)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        print('num_nodesaaaaaaaaaaaa', num_nodes)
        for src, dst in adj_dict.items(
        ):  # 格式为 {index：[index_of_neighbor_nodes]}
            edge_index.extend([src, v] for v in dst)  #
            edge_index.extend([v, src] for v in dst)

        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(
            edge_index)))  # 以轮到的元素为k,每个k对应一个数组，和k相同放进数组，不
        # 同再生成一个k,sorted()是以第一个元素大小排序

        edge_index = np.asarray(edge_index)

        # 稀疏矩阵 存储非0值 节省空间
        adjacency = sp.coo_matrix(
            (np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            # print(type(out))
            # print(out)
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = os.path.split(url)[-1]

        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())

        return True

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()  # 返回稀疏矩阵的coo_matrix形式


# # 这样可以单独测试Process_data函数
# a = CoraData('Cora')
# a.process_data()


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        完整GCN函数
        f = sigma(D^-1/2 A D^-1/2 * H * W)
        卷积是D^-1/2 A D^-1/2 * H * W
        adjacency = D^-1/2 A D^-1/2 已经经过归一化，标准化的拉普拉斯矩阵

        这样就把傅里叶变化和拉普拉斯矩阵结合起来了.

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # 定义GCN层的 W 权重形状
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        # 定义GCN层的 b 权重矩阵
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 这里才是声明初始化 nn.Module 类里面的W,b参数
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)  # 矩阵相乘 m由matrix缩写
        output = torch.sparse.mm(adjacency, support)  # sparse 稀疏的
        if self.use_bias:
            output += self.bias  # bias 偏置，偏见
        return output

    # 一般是为了打印类实例的信息而重写的内置函数
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)  # (N,1433)->(N,16)
        self.gcn2 = GraphConvolution(16, 7)  # (N,16)->(N,7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))  # (N,1433)->(N,16),经过relu函数
        logits = self.gcn2(adjacency, h)  # (N,16)->(N,7)
        return logits


# 超参数定义
LEARNING_RATE = 0.1  # 学习率
WEIGHT_DACAY = 5e-4  # 正则化系数 weight_dacay
EPOCHS = 200  # 完整遍历训练集的次数
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 设备
# 如果当前显卡忙于其他工作，可以设置为 DEVICE = "cpu"，使用cpu运行

# 加载数据，并转换为torch.Tensor
dataset = CoraData('Cora').data

node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

tensor_x = tensor_from_numpy(node_feature, DEVICE)  # (2708,1433)
tensor_y = tensor_from_numpy(dataset.y, DEVICE)  # (2708,)

tensor_train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)  # 前140个为True
tensor_val_mask = tensor_from_numpy(dataset.val_mask,
                                    DEVICE)  # 140 - 639  500个
tensor_test_mask = tensor_from_numpy(dataset.test_mask,
                                     DEVICE)  # 1708 - 2707 1000个

normalize_adjacency = CoraData.normalization(
    dataset.adjacency)  # 规范化邻接矩阵 计算 L=D^-0.5 * (A+I) * D^-0.5

num_nodes, input_dim = node_feature.shape  # 2708,1433

# 原始创建coo_matrix((data, (row, col)), shape=(4, 4)) indices为index复数 https://blog.csdn.net/chao2016/article/details/80344828?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522160509865819724838529777%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=160509865819724838529777&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-2-80344828.pc_first_rank_v2_rank_v28&utm_term=%E7%A8%80%E7%96%8F%E7%9F%A9%E9%98%B5%E7%9A%84coo_matrix&spm=1018.2118.3001.4449
indices = torch.from_numpy(
    np.asarray([normalize_adjacency.row,
                normalize_adjacency.col]).astype('int64')).long()

values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))

tensor_adjacency = torch.sparse.FloatTensor(indices, values,
                                            (num_nodes, num_nodes)).to(DEVICE)

# 根据三元组 构造 稀疏矩阵向量,构造出来的张量是 (2708,2708)

# 模型定义：Model, Loss, Optimizer
model = GcnNet(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)  # criterion评判标准
optimizer = optim.Adam(
    model.parameters(),
    # optimizer 优化程序 ，使用Adam 优化方法，权重衰减https://blog.csdn.net/program_developer/article/details/80867468
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DACAY)


# 训练主体函数
def train():
    loss_history = []
    val_acc_history = []
    model.train()

    train_y = tensor_y[tensor_train_mask]  # shape=（140，）不是（2708，）了
    # 共进行200次训练
    for epoch in range(EPOCHS):
        logits = model(tensor_adjacency,
                       tensor_x)  # 前向传播，认为因为声明了 model.train()，不用forward了
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督 (140,)

        loss = criterion(train_mask_logits, train_y)  # 计算损失值
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新

        train_acc, _, _ = test(tensor_train_mask)  # 计算当前模型训练集上的准确率  调用test函数
        val_acc, _, _ = test(tensor_val_mask)  # 计算当前模型在验证集上的准确率

        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print(
            "Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
                epoch, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, val_acc_history


# 测试函数
def test(mask):
    model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

    with torch.no_grad():  # 显著减少显存占用
        logits = model(tensor_adjacency, tensor_x)  # (N,16)->(N,7) N节点数
        test_mask_logits = logits[mask]  # 矩阵形状和mask一样

        predict_y = test_mask_logits.max(1)[1]  # 返回每一行的最大值中索引（返回最大元素在各行的列索引）
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu(
    ).numpy()


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)),
             loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = fig.add_subplot(111, sharex=ax1,
                          frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)),
             val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


loss, val_acc = train()
test_acc, test_logits, test_label = test(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())
plot_loss_with_acc(loss, val_acc)
