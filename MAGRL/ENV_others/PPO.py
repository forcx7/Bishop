import torch
from GRL_Library.agent import PPO_agent
from GRL_Net.Pytorch_GRL_PPO import Graph_Actor_Model, Graph_Critic_Model, NonGraph_Actor_Model, NonGraph_Critic_Model

def Create_PPO(num_HVs, num_AVs, Graph):
    # Initialize GRL model
    N = num_HVs + num_AVs
    F = 6
    A = 21
    lr = 0.001
    assert isinstance(Graph, bool)  # 断言，false就报错
    if Graph:
        GRL_actors = [Graph_Actor_Model(N, F, A, lr) for _ in range(num_AVs)]
        GRL_critics = [Graph_Critic_Model(N, F, A, lr) for _ in range(num_AVs)]
    else:
        GRL_actors = [NonGraph_Actor_Model(N, F, A, lr) for _ in range(num_AVs)]
        GRL_critics = [NonGraph_Critic_Model(N, F, A, lr) for _ in range(num_AVs)]

    # Discount factor
    gamma = 0.9
    # GAE factor 广义优势估计（GAE）的因子
    GAE_lambda = 0.95
    # Policy clip factor PPO算法的策略剪切因子
    policy_clip = 0.2

    # Initialize GRL agent，封装GRL_PPO其实就是GRL_agent文件
    GRL_PPO = PPO_agent.PPO(
        GRL_actors,  # actor model
        GRL_critics,  # critic model
        gamma,  # discount factor
        GAE_lambda,  # GAE factor
        policy_clip,  # policy clip factor
        batch_size=32,  # batch_size < update_interval
        n_epochs=5,  # update times for one batch
        update_interval=100,  # update interval
        model_name="DQN_model"  # model name
    )

    return GRL_PPO

