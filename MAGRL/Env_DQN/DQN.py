# 初始化DQN类
import torch
from GRL_Library.agent import DQN_agent  # 导入编写的GRL_agent
from GRL_Net.Pytorch_GRL import torch_GRL  # 导入编写的pytorch下的网络
from GRL_Library.common import replay_buffer, explorer  # 导入编写的GRL库


def Create_DQN(num_HVs, num_AVs, param):
    # 初始化GRL网络
    # N = num_HVs + num_AVs
    # F = 6 + param['n_lanes']
    N = num_AVs
    F = 6
    A = 21

    # 为每一个智能体提供一个Q网络
    Nets = [torch_GRL(N, F, A) for _ in range(num_AVs)]
    # 为每一个智能体提供优化器
    optimizers = [torch.optim.Adam(Nets[i].parameters(), lr = 0.00075) for i in range(num_AVs)]

    # 定义replay_buffer
    replay_buffers = [replay_buffer.ReplayBuffer(size=10 ** 6) for _ in range(num_AVs)]
    # 定义折扣因子
    gamma = 0.99
    # 定义智能体策略参数
    explorer_0 = explorer.LinearDecayEpsilonGreedy(start_epsilon=0.6, end_epsilon=0.1, decay_step=400000)

    # 初始化DQN类
    warmup = 1000  # 设置warmup步长
    GRL_DQN = DQN_agent.DQN(
        Nets,  # 模型采用的网络
        optimizers,  # 模型采用的优化器
        explorer_0,  # 策略探索模型
        replay_buffers,  # 经验池
        gamma,  # 折扣率
        batch_size=64,  # 定义batch_size
        warmup_step=warmup,  # 定义开始更新的步长
        update_interval=50,  # 当前网络更新步长间隔
        target_update_interval=2000,  # 目标网络更新步长间隔
        target_update_method='soft',  # 目标网络更新方法
        soft_update_tau=0.1,  # 若soft_update，定义更新权重
        n_steps=1,  # multi-steps learning学习步长
        model_name="DQN_model"  # 模型命名
    ) # 就是外包，没多大差别


    return GRL_DQN
