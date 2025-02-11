import torch
from GRL_Library.agent import SAC_agent    # SAC网络的主要功能
from GRL_Net.Pytorch_GRL_SAC import ActorNetwork  # SAC算法的策略网络
from GRL_Net.Pytorch_GRL import torch_GRL  # 导入编写的pytorch下的网络
from GRL_Library.common import replay_buffer, explorer  # SAC网络的经验回放和？？？
import copy
import numpy as np

def Create_SAC(num_HVs, num_AVs, param):
    N = num_HVs + num_AVs  # 总的车辆
    F = 6 + param['n_lanes']  # 环境的特征维度
    A = 24  # 离散动作数量，根据您的环境确定
    # 创建网络
    actor_nets = [ActorNetwork(N, F, A) for _ in range(num_AVs)]
    # 为每个自动驾驶车辆（AV）创建一个策略网络（即演员网络），该网络的输入是车辆数量和交通环境的特征，输出是一个离散的动作
    critic_nets_1 = [torch_GRL(N, F, A) for _ in range(num_AVs)]
    critic_nets_2 = [torch_GRL(N, F, A) for _ in range(num_AVs)]
    # 创建两个价值网络（Q值网络），它们用于估计每个状态 - 动作对的价值。
    # 在SAC中，使用两个Q值网络来进行双重Q学习（DoubleQ - Learning）以减少过估计偏差。
    target_critic_nets_1 = [copy.deepcopy(net) for net in critic_nets_1]
    target_critic_nets_2 = [copy.deepcopy(net) for net in critic_nets_2]
    # 目标Q值网络是Q - learning中常见的目标网络（Target Network），
    # 通过软更新（soft update）来避免Q值的震荡。
    # 这里使用copy.deepcopy来复制critic_nets_1和critic_nets_2，使它们成为独立的网络

    actor_optimizers = [torch.optim.Adam(actor_nets[i].parameters(), lr=3e-4) for i in range(num_AVs)]
    critic_optimizers_1 = [torch.optim.Adam(critic_nets_1[i].parameters(), lr=3e-4) for i in range(num_AVs)]
    critic_optimizers_2 = [torch.optim.Adam(critic_nets_2[i].parameters(), lr=3e-4) for i in range(num_AVs)]
    # 为每个自动驾驶车辆的策略网络/目标网络创建一个Adam优化器

    alpha = 0.2
    # 是SAC中的温度参数，用于平衡奖励和熵。通常初始值设置为0.2
    log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
    # 是 alpha 的对数形式，这是为了便于梯度下降优化。它作为可学习参数进行优化
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    # 创建用于优化 log_alpha 的Adam优化器

    replay_buffer_0 = replay_buffer.ReplayBuffer(size=10**6)
    # 创建一个经验回放缓冲区，用于存储从环境中收集到的状态、动作、奖励和下一个状态等信息
    gamma = 0.99 # 折扣因子，用于计算未来奖励的折扣
    tau = 0.005  # 软更新的步长，目标Q网络的更新速率
    batch_size = 64
    warmup_step = 10000
    target_entropy = -np.log(1.0/A)*0.98  # 根据经验或需要设定目标熵，离散动作一般设为接近 -log(1/A)

    GRL_SAC = SAC_agent.DiscreteSAC(
        actor_nets=actor_nets,
        critic_nets_1=critic_nets_1,
        critic_nets_2=critic_nets_2,
        target_critic_nets_1=target_critic_nets_1,
        target_critic_nets_2=target_critic_nets_2,
        actor_optimizers=actor_optimizers,
        critic_optimizers_1=critic_optimizers_1,
        critic_optimizers_2=critic_optimizers_2,
        alpha_optimizer=alpha_optimizer,
        log_alpha=log_alpha,
        replay_buffer=replay_buffer_0,
        gamma=gamma, tau=tau, batch_size=batch_size,
        warmup_step=warmup_step,
        target_entropy=target_entropy,
        model_name="SAC_model"
    )

    return None, GRL_SAC
