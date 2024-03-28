'''
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2023-02-06 20:15:00
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        L*X*\theta
        :param input_dim: 节点输入特征维度
        :param output_dim: 输出特征维度
        :param use_bias: 是否偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)  # torch.mm是指两个二维的张量相乘
        output = torch.sparse.mm(
            adjacency, support)  # torch.sparse.mm(a,b)是指a为稀疏矩阵，b为稀疏矩阵或密集矩阵
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GCN(nn.Module):

    def __init__(self, input_dim=1433):
        """
        两层GCN模型
        :param input_dim: 输入维度
        """
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 128)
        self.gcn2 = GraphConvolution(128, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = F.softmax(self.gcn2(adjacency, h), dim=-1)
        return logits


if __name__ == '__main__':
    net = GCN().cuda()
    print(net)