import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

USE_CUDA = torch.cuda.is_available()


# 无图
class ActorNetwork(nn.Module):
    def __init__(self, N, F, A):
        super(ActorNetwork, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        # 可以参考torch_GRL的结构做一个类似的全连接结构，但不需要图卷积(根据需要)
        # 这里简化，只进行全连接层构建Actor
        self.fc1 = nn.Linear(F, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_output = nn.Linear(128, A)

        if USE_CUDA:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, observation):
        # observation为(get_step的返回)
        X_in, A_in_Dense, RL_indice = self.datatype_transmission(observation)
        # 对RL_indice位置上的agent进行决策，Actor也可只对这些有控制权的agent输出动作概率
        # 简化处理：Actor对所有agent输出动作概率，然后用mask选取相应的agent
        x = F.relu(self.fc1(X_in))
        x = F.relu(self.fc2(x))
        logits = self.policy_output(x)
        # mask用于过滤掉HV(agent)的输出
        # mask = RL_indice.unsqueeze(1)  # (N,1)
        return logits

    def datatype_transmission(self, states):
        features = torch.as_tensor(states[0], dtype=torch.float32, device=self.device)
        adjacency = torch.as_tensor(states[1], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(states[2], dtype=torch.float32, device=self.device)
        return features, adjacency, mask