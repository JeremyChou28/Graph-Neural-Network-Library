# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/10/20 11:01
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# 归一化特征
# 按行求均值
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


# 归一化邻接矩阵
# AD^{-1/2}.TD^{-1/2}
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def load_data(path, dataset_str):
    # step 1: 读取 x, y, tx, ty, allx, ally, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # step 2: 读取测试集索引
    test_idx_reorder = parse_index_file(os.path.join(path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    #
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # 获取整个图的所有节点特征
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # features = preprocess_features(features) 根据自己需要归一化特征
    features = features.toarray()

    # 获取整个图的邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # adj = preprocess_adj(adj) 根据自己需要归一化邻接矩阵
    adj = adj.toarray()

    # 获取所有节点标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    # 划分训练集、验证集、测试集索引
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    idx_test = test_idx_range.tolist()

    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == "__main__":
    '''
    adj, features, labels, idx_train, idx_val, idx_test = load_data('./Cora/raw', 'cora')

    print('--------adj--------')
    print(adj.shape)
    print(adj)
    print('--------features--------')
    print(features.shape)
    print(features)
    print('--------labels--------')
    print(labels.shape)
    print(labels)
    print('--------idx_train--------')
    print(idx_train)
    print('--------idx_val--------')
    print(idx_val)
    print('--------idx_test--------')
    print(idx_test)
    '''
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='./', name='Cora')  # if root='./', Planetoid will use local dataset
    # Cora节点的特征数量
    print(dataset.num_features)
    # Cora节点类别数量
    print(dataset.num_classes)

    # 得到数据集信息
    print(dataset.data)
    print(type(dataset))

    # 输出为
    # Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
    # data = dataset.data
    # data.y[data.train_mask]  # 得到训练集的标签
