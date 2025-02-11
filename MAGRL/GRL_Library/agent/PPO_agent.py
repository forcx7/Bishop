"""
    This function is used to define the PPO agent
"""

import torch as T
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import collections
from config import config

# CUDA configuration
USE_CUDA = T.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
#     if USE_CUDA else autograd.Variable(*args, **kwargs)
# autograd.set_detect_anomaly(True)


class PPOMemory(object):
    """
        Define PPOMemory class as replay buffer

        Parameter description:
        --------
        state: current state
    """

    def __init__(self, batch_size):
        self.states = []
        self.probs = []  # Action probability
        self.vals = []  # Value
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batch(self):
        """
           <batch sampling function>
           Used to implement empirical sampling of PPOMemory
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size) # 每个批次的开始位置，batch_size = 32
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices) # 打乱索引
        batches = [indices[i:i + self.batch_size] for i in batch_start] # 打乱索引后的小批次

        return self.states, \
               self.actions, \
               self.probs, \
               self.vals, \
               np.asarray(self.rewards), \
               np.asarray(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        """
           <data storage function>
           Used to store the data of the agent interaction process

           Parameters:
           --------
           state: current state
           action: current action
           probs: action probability
           vals: value of the action
           reward: the reward for performing the action
           done: whether the current round is completed or not
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """
           <data clear function>
           Used to clear the interaction data already stored and free memory
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPO(object):
    """
        Define the PPO class (Proximal Policy Optimization)

        Parameter description:
        --------
        actor_model: actor network
        actor_optimizer: actor optimizer
        critic_model: value network
        critic_optimizer: critic optimizer
        gamma: discount factor
        GAE_lambda: GAE (generalized advantage estimator) coefficient
        policy_clip: policy clipping coefficient
        batch_size: sample size
        n_epochs: number of updates per batch
        update_interval: model update step interval
        model_name: model name (used to save and read)
    """

    def __init__(self,
                 actor_models,
                 critic_models,
                 gamma,
                 GAE_lambda,
                 policy_clip,
                 batch_size,
                 n_epochs,
                 update_interval,
                 model_name):

        self.actor_models = actor_models
        self.critic_models = critic_models
        self.gamma = gamma
        self.GAE_lambda = GAE_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_interval = update_interval
        self.model_name = model_name

        # GPU configuration
        if USE_CUDA:
            GPU_num = T.cuda.current_device()
            self.device = T.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        # Replay buffer
        self.memories = [PPOMemory(self.batch_size) for _ in range(config.num_av)]

        # Record data
        self.loss_record = collections.deque(maxlen=100)

    def store_transition(self, state, action, probs, vals, reward, done, i):
        """
           <Experience storage function>
           Used to store the experience data during the agent learning process

           Parameters:
           --------
           state: the state of the current moment
           action: current moment action
           probs: probability of current action
           vals: the value of the current action
           reward: the reward obtained after performing the current action
           done: whether to terminate or not
        """
        self.memories[i].store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation, j):
        """
          <Action selection function>
          Generate agent's action based on environment observation

          Parameters:
          --------
          observation: the environment observation of the smart body
       """
        # actor 负责策略生成，而 critic 负责评估当前策略的好坏
        dist = self.actor_models[j](observation)  # 输出概率分布
        value = self.critic_models[j](observation) #评估当前状态的价值，为每个状态预测一个标量值，表示从该状态开始可能获得的总回报
        action = dist.sample() # 随机选择动作（动作索引）

        # 给出选定动作的对数概率，通常用于后续的策略梯度计算，移除可能存在的单一维度，确保结果是一个标准的张量
        probs = T.squeeze(dist.log_prob(action))
        action = T.squeeze(action) # 确保动作的维度没有多余的维度
        value = T.squeeze(value) # 确保动作的维度没有多余的维度

        return action, probs, value

    def learn(self, exist_AV):
        """
           <policy update function>
           Used to implement the agent's learning process
        """
        num_av = config.num_av
        for j in range(num_av):
            # ------计算损失------ #
            # 判断该av是否存在
            if f't_{j}' in exist_AV:
                # ------Training according to the specific value of n_epochs(5)------ #
                for _ in range(self.n_epochs): # 为什么要循环？？？
                    # 提取出多个transition
                    # 多出来的batches是分批次的索引
                    state_arr, action_arr, old_prob_arr, vals_arr, \
                    reward_arr, dones_arr, batches = \
                        self.memories[j].generate_batch()

                    values = vals_arr

                    # ------（Advantage）Training for each epoch------ #
                    # 优势函数计算，其衡量的是某个动作的相对价值，有助于调整策略
                    # 当前时间步的优势是基于当前的奖励、下一时间步的价值、当前的价值和是否终止状态进行加权计算的
                    # advantage = T.zeros(len(reward_arr), len(action_arr[1])).to(self.device) # 零矩阵（X * 12）
                    returns = np.zeros_like(reward_arr)  # 初始化returns
                    for t in reversed(range(len(reward_arr))):
                        if t == len(reward_arr) - 1:
                            returns[t] = reward_arr[t]
                        else:
                            returns[t] = reward_arr[t] + self.gamma * returns[t + 1]  # 计算returns
                    row_sums = [row.sum().item() for row in values]
                    row_sums = T.stack([T.tensor([sum_value]) for sum_value in row_sums])
                    # row_sums = T.stack(row_sums)
                    advantage = T.tensor(returns).to(self.device) - row_sums.T.to(self.device)
                    advantage = advantage.squeeze(0)  # 将形状从 (1, 99) 改为 (99,)
                    # advantage = torch.stack(advantage)

                    # values堆叠，，从list转为tensor，要求维度相同，X * 12
                    # values = T.stack(values)

                    # 训练小批次batch，如果有100个transition的话就有4个batch，每个batch里有32个索引，最后一个batch有4个
                    # Note: do loss.backward() in a loop, to avoid gradient compounding
                    for batch in batches:
                        # Initialize the loss matrix
                        actor_loss_matrix = []
                        critic_loss_matrix = []

                        # 训练 each index in the batch，i是索引
                        # Calculate actor_loss and update the actor network
                        for i in batch:
                            # 取出每个样本的probs和actions，并detach防止计算梯度
                            old_probs = old_prob_arr[i].detach()
                            actions = action_arr[i].detach()
                            # 新的概率，得清楚这些东西维度可都是12！！！
                            dist = self.actor_models[j](state_arr[i])
                            if actions.numel() != dist.probs.shape[0]:
                                continue
                            new_probs = dist.log_prob(actions) # 取对数
                            prob_ratio = new_probs.exp() / old_probs.exp() # prob前后比值，可动态
                            # ------PPO1------#
                            weighted_probs = advantage[i].detach() * prob_ratio
                            # ------PPO2------#
                            weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                             1 + self.policy_clip) * advantage[i].detach()
                            # ----------------#
                            # 取最小值确保策略在合理范围内更新，防止过度优化
                            # 取最小值的负数作为演员的损失（因为我们要最大化目标，所以损失是负数）
                            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                            actor_loss_matrix.append(actor_loss)

                        actor_loss_matrix = T.stack(actor_loss_matrix) # 堆叠成张量
                        actor_loss_mean = T.mean(actor_loss_matrix) # 计算平均
                        self.actor_models[j].optimizer.zero_grad() # 梯度清零
                        actor_loss_mean.backward() # 反向传播
                        self.actor_models[j].optimizer.step() # 更新actor参数


                        # Calculate critic_loss and update the critic network
                        for i in batch:
                            # 新的value
                            critic_value = self.critic_models[j](state_arr[i])
                            critic_value = T.squeeze(critic_value)
                            # 优势函数（advantage）+状态值函数（value） = Q
                            returns = advantage[i] + values[i] # 维度为 X * 12，折扣回报
                            returns = returns.detach()
                            # 计算平滑L1损失，即差的绝对值
                            if returns.numel() != critic_value.numel():
                                continue
                            critic_loss = F.smooth_l1_loss(returns, critic_value)
                            critic_loss_matrix.append(critic_loss)

                        critic_loss_matrix = T.stack(critic_loss_matrix)
                        critic_loss_mean = 0.5 * T.mean(critic_loss_matrix)
                        self.critic_models[j].optimizer.zero_grad()
                        critic_loss_mean.backward()
                        self.critic_models[j].optimizer.step()

                        # Save loss
                        self.loss_record.append(float((actor_loss_mean + critic_loss_mean).detach().cpu().numpy()))

                # Buffer clear，每次learn完之后清空
                # self.memories[j].clear_memory()

    def get_statistics(self):
        """
           <training data acquisition function>
           Used to get the relevant data during training
        """
        loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        return [loss_statistics]

    def save_model(self, save_path):
        """
           <Model saving function>
           Used to save the trained model
        """
        save_path_actor = save_path + "/" + self.model_name + "_actor" + ".pt"
        save_path_critic = save_path + "/" + self.model_name + "_critic" + ".pt"
        T.save(self.actor_models, save_path_actor)
        T.save(self.critic_models, save_path_critic)

    def load_model(self, load_path):
        """
           <Model reading function>
           Used to read the trained model
        """
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic = load_path + "/" + self.model_name + "_critic" + ".pt"
        self.actor_model = T.load(load_path_actor)
        self.critic_model = T.load(load_path_critic)
