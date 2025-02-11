import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym

# 定义Q网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)  # 输出Q值对应每个动作
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放（Replay Buffer）类
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.size = 0
    
    def push(self, experience):
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[random.randint(0, self.capacity-1)] = experience
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return self.size

# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
    
    def act(self, state):
        # 探索还是利用（ε-greedy）
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从回放缓冲区采样
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # 计算目标Q值
        next_q_values = self.target_network(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # 计算当前Q值
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_target_network()
        
        # 更新epsilon值（探索度衰减）
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# 训练DQN智能体
def train_dqn(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# 创建环境和智能体
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

# 训练智能体
train_dqn(env, agent, episodes=1000)



import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义一个简单的神经网络作为策略模型
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)  # 输出为一个概率分布
        return x

# REINFORCE算法
def reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99):
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        log_probs = []
        rewards = []

        # 在当前策略下与环境交互
        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            probs = policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # 存储log probability和奖励
            log_probs.append(dist.log_prob(action))
            rewards.append(0 if done else 1)  # 假设每个时间步的奖励为1，终止时为0

            state, _, done, _, _ = env.step(action.item())

        # 计算回报，使用蒙特卡洛方法
        episode_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            episode_returns.insert(0, R)

        # 更新策略
        optimizer.zero_grad()
        loss = 0
        for log_prob, R in zip(log_probs, episode_returns):
            loss -= log_prob * R  # 负号是因为我们在进行梯度下降
        loss.backward()
        optimizer.step()

        # 每隔100个回合打印一次结果
        if episode % 100 == 0:
            print(f'Episode {episode}/{num_episodes}, Loss: {loss.item()}')

# 创建环境和模型
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# 训练模型
reinforce(env, policy, optimizer)

