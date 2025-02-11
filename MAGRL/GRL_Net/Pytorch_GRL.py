import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

# from GRL_Net.network_test import RL_indice


# 该函数用来将环境中的observation转换成pytorch可以接收的float32的Tensor数据类型
# 注意：根据observation的数据结构特点的不同，需要对函数进行相应更改
def datatype_transmission(states, device):
    # print(type(states[1][i]))  # 输出 states[1][i] 的类型
    # print(states[1][i])  # 输出 states[1][i] 的内容
    features = torch.as_tensor(states[0], dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(states[1], dtype=torch.float32, device=device)
    # mask = torch.as_tensor(states[2][i], dtype=torch.float32, device=device)
    return features, adjacency


# 下述为pytorch框架下的GRL网络的主程序
class torch_GRL(nn.Module):
    # N为智能体数量，F为每个智能体的特征长度，A为可选择的动作数量(24)
    def __init__(self, N, F, A):
        super(torch_GRL, self).__init__()
        self.num_agents = N
        self.num_outputs = A

        # 定义编码器
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 128)

        # 定义图卷积网络
        # self.GraphConv = GraphConv(40, 32)
        self.GraphConv = GCNConv(128, 128)
        self.GraphConv_Dense = nn.Linear(128, 128)

        # 定义策略层
        self.policy_1 = nn.Linear(256, 128)
        self.policy_2 = nn.Linear(128, 64)
        self.policy_output = nn.Linear(64, A)

        # GPU设置
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def forward(self, observation):
    # def forward(self, observation):
        # 这里数据类型是numpy.ndarray，需要转换为Tensor数据类型
        # observation为状态观测矩阵，包括X_in, A_in_Dense和RL_indice三部分
        # X_in为节点特征矩阵，A_in_Dense为稠密邻接矩阵（NxN）(原始输入)
        # A_in_Sparse为稀疏邻接矩阵COO（2xnum），RL_indice为强化学习索引

        # X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device, i)
        X_in, A_in_Dense = datatype_transmission(observation, self.device)

    # 计算X_in解码后的结果
        # print("X_in.shape:", X_in.shape)
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # 计算图卷积网络后的结果
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)  # 将observation的邻接矩阵转换成稀疏矩阵
        # print("A_in_Sparse.shape:", A_in_Sparse.shape)
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)
        # print("X_graph.shape:", X_graph.shape)

        # 特征按列聚合
        F_concat = torch.cat((X_graph, X), 1)
        # print("F_concat:", F_concat)
        # print("F_concat.shape:", F_concat.shape)

        # 计算策略层的结果
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_output(X_policy) # 为行向量
        # print("X_policy.shape:", X_policy.shape)
        # print("X_policy:", X_policy)

        # 重新规定RL_indice的维度
        # mask = torch.reshape(RL_indice, (self.num_agents, 1)) # 横向量转为列向量，mask前面是HV，后面是AV所有形式为[0;0;0;.....1;1;1;1....]
        # mask = RL_indice.reshape(-1,1) # 横向量转为列向量
        # print("mask:", mask)

        # 计算网络最终输出
        # print("X_policy:", X_policy)
        # output = torch.mul(X_policy, mask) # 只输出自动驾驶车辆的动作，人类驾驶车辆按照指定的动作继续行驶，对于MADQN我只需要输出自车的动作序列即可
        output = X_policy
        # print("output:", output)
        # print("output.shape:", output.shape)

        return output
