import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from config import DEVICE


class AggregatorLSTM(nn.Module):
    """LSTM邻域聚合函数"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.weight_ih_l0 = nn.Parameter(
            torch.Tensor(4 * hidden_dim, input_dim))
        self.weight_hh_l0 = nn.Parameter(
            torch.Tensor(4 * hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.lstm.all_weights)
        init.kaiming_uniform_(self.lstm.weight_ih_l0)
        init.kaiming_uniform_(self.lstm.weight_hh_l0)

    def forward(self, x):
        # x: (batch_size, num_neighbor_nodes, input_dim)

        out, (ht, ct) = self.lstm(x)  # (batch_size, num_nodes, hidden_dim)
        return out


class NeighborAggregator(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 use_bias=False,
                 aggr_method="mean"):
        """
        聚合节点邻居
        :param input_dim: 输入特征的维度
        :param output_dim: 输出特征的维度
        :param use_bias: 是否使用偏置
        :param aggr_method: 邻居聚合算子形式
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        elif self.aggr_method == "LSTM":
            self.aggr_lstm = AggregatorLSTM(
                self.input_dim, neighbor_feature.shape[-1]).to(DEVICE)
            lstm_out = self.aggr_lstm(neighbor_feature)
            aggr_neighbor, _ = torch.max(lstm_out, dim=1)
            # print(neighbor_feature.shape[-1])
            # print('aggr_neighbor shape:', aggr_neighbor.shape)
            # print('neighbor_feature shape:', neighbor_feature.shape)
        else:
            raise ValueError(
                "Unknown aggr type, expected sum, max, or mean, but got {}".
                format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class SAGEGCN(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """
        单层的SAGE，主要是将节点聚合后的邻域信息和节点当前的隐藏表示相结合：拼接或求和。
        :param input_dim: 输入维度
        :param hidden_dim: 输出维度
        :param activation: 激活函数
        :param aggr_neighbor_method: 邻居特征聚合的方法
        :param aggr_hidden_method: 节点特征更新方法
        """
        super(SAGEGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max", "LSTM"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim,
                                             hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(
                self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSAGE(nn.Module):
    """搭建多个单层网络"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_neighbors_list,
                 aggre_method="mean"):
        """
        :param input_dim: 输入维度
        :param hidden_dim: 输出维度
        :param num_neighbors_list: 采样邻居个数
        """
        super(GraphSAGE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(
            SAGEGCN(input_dim,
                    hidden_dim[0],
                    aggr_neighbor_method=aggre_method))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(
                SAGEGCN(hidden_dim[index],
                        hidden_dim[index + 1],
                        aggr_neighbor_method=aggre_method))
        self.gcn.append(
            SAGEGCN(hidden_dim[-2],
                    hidden_dim[-1],
                    activation=None,
                    aggr_neighbor_method=aggre_method))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                # print('neighbor_node_features shape:',
                #       neighbor_node_features.shape)
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list)
