import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import collections
from GRL_Library.common.prioritized_replay_buffer import PrioritizedReplayBuffer
from config import config

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)

class DiscreteSAC:
    def __init__(self, actor_nets, critic_nets_1, critic_nets_2, target_critic_nets_1, target_critic_nets_2,
                 actor_optimizers, critic_optimizers_1, critic_optimizers_2,
                 alpha_optimizer, log_alpha, replay_buffer,
                 gamma, tau, batch_size, warmup_step, target_entropy,
                 model_name):

        self.actor_nets = actor_nets
        self.critic_nets_1 = critic_nets_1
        self.critic_nets_2 = critic_nets_2
        self.target_critic_nets_1 = target_critic_nets_1
        self.target_critic_nets_2 = target_critic_nets_2

        self.actor_optimizers = actor_optimizers
        self.critic_optimizers_1 = critic_optimizers_1
        self.critic_optimizers_2 = critic_optimizers_2

        self.alpha_optimizer = alpha_optimizer
        self.log_alpha = log_alpha
        self.alpha = self.log_alpha.exp()

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_step = warmup_step
        self.target_entropy = target_entropy
        self.model_name = model_name

        self.time_counter = 0
        self.loss_record = collections.deque(maxlen=100)
        self.q_record = collections.deque(maxlen=100)

        if USE_CUDA:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def sample_memory(self):
        samples, data_sample = self.replay_buffer.sample(self.batch_size, n_steps=1)
        return samples, data_sample

    def choose_action(self, observation, i):
        # 对第i个AV选择动作
        # 从Actor网络输出logits后进行softmax采样
        with torch.no_grad():
            logits = self.actor_nets[i - config.num_hv](observation)  # 返回(N, A)的logits
            # 只取对应的i-th agent条目: 
            # 假设i-th agent对应network输出的第i条数据（请根据实际定义修改索引）
            # 若N为总车辆数，该处需要根据i定位到AV在状态特征矩阵中的位置
            # 简化假设：RL_indice对AV在后半部分，即[i - num_hv]对应索引在states中是num_hv + (i - num_hv) = i位置
            # 实际需要严格对应一下，假设i-th AV在输出中的位置为(num_hv + (i-num_hv))=i（与DQN中一致）
            # 如不一致，请根据实际情况进行索引变换。
            agent_index = i  # 简化处理，需要依据实际情况进行修正
            agent_logits = logits[agent_index].unsqueeze(0)  # (1,A)
            probs = F.softmax(agent_logits, dim=1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
            return action

    def learn(self):
        if (self.time_counter <= self.warmup_step):
            self.time_counter += 1
            return

        samples, data_sample = self.sample_memory()
        num_av = config.num_av

        for i in range(num_av):
            states, actions, rewards, next_states, dones = self._batch_data(data_sample)
            # 转为GPU
            states = [torch.FloatTensor(s).to(self.device) for s in states]
            # states[0]=features (N,F); states[1]=Adj (N,N); states[2]=mask (N,)
            features, adjacency, masks = states

            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_features = torch.FloatTensor([d[3][0] for d in data_sample]).to(self.device) # next_features
            next_adj = torch.FloatTensor([d[3][1] for d in data_sample]).to(self.device)
            next_mask = torch.FloatTensor([d[3][2] for d in data_sample]).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # 这里的states和next_states是batch，需要堆叠处理，实际中需要保证get_step返回的结构是可批处理的
            # 简化假设get_step返回的结构是固定的N,可以直接拼接，实际需根    据代码逻辑进行处理

            # Critic更新
            with torch.no_grad():
                # 下个状态下的policy分布
                # next_state输入到actor_nets[i]
                next_state_obs = [next_features, next_adj, next_mask]
                logits_next = self.actor_nets[i](next_state_obs)
                probs_next = F.softmax(logits_next, dim=1)  # (N,A)
                log_probs_next = torch.log(probs_next + 1e-8)

                # 利用target critics计算Q_next
                Q1_next = self.target_critic_nets_1[i](next_state_obs) # (N,A)
                Q2_next = self.target_critic_nets_2[i](next_state_obs) # (N,A)
                Q_next = torch.min(Q1_next, Q2_next)
                
                # V(next_state) = sum_a pi(a|s) [Q(s,a)-alpha*log_pi(a|s)]
                V_next = (probs_next * (Q_next - self.alpha * log_probs_next)).sum(dim=1, keepdim=True)
                Q_target = rewards + (1 - dones)*self.gamma*V_next

            # 当前Q值
            Q1 = self.critic_nets_1[i]([features, adjacency, masks]) # (N,A)
            Q2 = self.critic_nets_2[i]([features, adjacency, masks]) # (N,A)

            # actions为从buffer中采样，一批(N,)，Q根据actions选取对应Q值
            Q1_a = Q1.gather(1, actions.unsqueeze(1))
            Q2_a = Q2.gather(1, actions.unsqueeze(1))

            critic_loss_1 = F.mse_loss(Q1_a, Q_target)
            critic_loss_2 = F.mse_loss(Q2_a, Q_target)

            self.critic_optimizers_1[i].zero_grad()
            critic_loss_1.backward()
            self.critic_optimizers_1[i].step()

            self.critic_optimizers_2[i].zero_grad()
            critic_loss_2.backward()
            self.critic_optimizers_2[i].step()

            # Actor更新
            # 重新计算当前状态下的策略分布
            logits = self.actor_nets[i]([features, adjacency, masks])
            probs = F.softmax(logits, dim=1)  # (N,A)
            log_probs = torch.log(probs + 1e-8)

            Q1_current = self.critic_nets_1[i]([features, adjacency, masks])
            Q2_current = self.critic_nets_2[i]([features, adjacency, masks])
            Q_current = torch.min(Q1_current, Q2_current)

            actor_loss = (probs*(self.alpha*log_probs - Q_current)).sum(dim=1).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # 自动调节alpha
            with torch.no_grad():
                entropy = - (probs*log_probs).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            # soft update
            self.soft_update(self.target_critic_nets_1[i], self.critic_nets_1[i], self.tau)
            self.soft_update(self.target_critic_nets_2[i], self.critic_nets_2[i], self.tau)

            self.loss_record.append(float((critic_loss_1.item() + critic_loss_2.item())/2.0))
            self.q_record.append(Q1_a.mean().item())

        self.time_counter += 1

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau)*target_param.data + tau*source_param.data)

    def _batch_data(self, data_sample):
        # data_sample格式: [(state, action, reward, next_state, done), ...]批大小为self.batch_size
        # 这里需要将state, next_state解包，并组成tensor
        states = [d[0] for d in data_sample]
        actions = [d[1] for d in data_sample]
        rewards = [d[2] for d in data_sample]
        next_states = [d[3] for d in data_sample]
        dones = [d[4] for d in data_sample]

        # states是 (features, adjacency, mask)的tuple
        # 将它们stack起来
        # 假设get_step返回定长N，不变，可以直接stack
        features = np.stack([s[0] for s in states])
        adjacency = np.stack([s[1] for s in states])
        masks = np.stack([s[2] for s in states])
        # 同理next_states
        # 如果next_state格式相同：
        # 已在learn中分开处理，这里直接返回即可
        # 返回格式统一为list
        return [features, adjacency, masks], actions, rewards, next_states, dones

    def get_statistics(self):
        loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        q_statistics = np.mean(np.absolute(self.q_record)) if self.q_record else np.nan
        return [loss_statistics, q_statistics]

    def save_model(self, save_path):
        # 保存actor和critic
        torch.save(self.actor_nets, save_path + "/" + self.model_name + "_actor.pt")
        torch.save(self.critic_nets_1, save_path + "/" + self.model_name + "_critic_1.pt")
        torch.save(self.critic_nets_2, save_path + "/" + self.model_name + "_critic_2.pt")

    def load_model(self, load_path):
        self.actor_nets = torch.load(load_path + "/" + self.model_name + "_actor.pt")
        self.critic_nets_1 = torch.load(load_path + "/" + self.model_name + "_critic_1.pt")
        self.critic_nets_2 = torch.load(load_path + "/" + self.model_name + "_critic_2.pt")