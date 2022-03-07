#代码概述
'''
1.创建了一个DQN模型网络 DQNModel
2.提供了存储历史经验(state, action ,reward, next_state, done)的类，ReplayMemory
3.创建了DQN代理进行仿真操作 DQNAgent
4.提供了ε-greedy方法，进行动作选取
5.提供了奖励期望计算方法
'''
import random
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNModel(nn.Module):
    """
    Deep Q-Network Neural Network
    """
    def __init__(self, n_actions, env_dims, n_hidden1=14, n_hidden2=28):
        """
        Neural Network Architecture, same for model and target_model
        """
        super().__init__()
        # 1 Input Layer, 2 fully connected hidden layers, 1 output layer
        #设置网络全连接层
        """
        nn.Linear(in_features, out_features, bias=True)
        :param in_features: size of each input sample
        :param out_feature: size of each output sample
        :bias: 偏移量 默认为true
        note：一个输入为[batch_size, in_features]的张量变为[batch_size, out_features]的张量
        """
        self.fc1 = nn.Linear(in_features = env_dims, 
                             out_features = n_hidden1)
        self.fc2 = nn.Linear(in_features = n_hidden1, 
                             out_features = n_hidden2)
        self.out = nn.Linear(in_features = n_hidden2, 
                             out_features = n_actions)
        
    def forward(self, t):
        """
        Feedforward(前馈，正反馈) state through neural network,非线性化
        """
        # reLu activation functions for both layers, pass state through network
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


#存储经验
class ReplayMemory():
    """
    Experience Replay to store the experiences of the agent
    """
    def __init__(self, capacity):
        # Capacity of the Experience Replay
        self.capacity = capacity    
        # Initialize Experience Repaly
        self.memory = [None] * self.capacity 
        self.memory_counter = 0
         
    def store(self, experience):
        """
        Save experience to Experience Replay
        """
        self.memory[self.memory_counter % self.capacity] = experience
        self.memory_counter +=1

    def sample(self, batch_size):
        """
        Sample experience from Experience Replay 获得batch_size数量的样例
        """
        if self.memory_counter < self.capacity:
            # Sample from memory_counter available experiences
            # return [] size == batch_size
            return random.sample(self.memory[:self.memory_counter], batch_size)
        else:
            # Sample from whole replay memory
            return random.sample(self.memory, batch_size) 
    
    def sample_possible(self, batch_size):
        """
        Check if sampling from memory is possible
        """
        return self.memory_counter >= batch_size


class EpsilonGreedy():
    """
    Epsilon Greedy strategy
    の-Greedy策略
    """
    def __init__(self, start, end, decay):
        """
        Initialize Epsilon greedy strategy variables
        """
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        """
        Calculate current exploration rate via exponential decay
        """
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class DDQNAgent():
    """
    Double-DQN RL Agent
    :param env 环境
    :param model 模型
    :param target_model 目标模型
    :param lr 学习率
    """
    def __init__(self, env, model, target_model, lr, buffer_sz, epsilon, epsilon_decay, 
    min_epsilon, gamma, target_update_iter, start_learning):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.lr = lr                            #学习率
        self.buffer_sz = buffer_sz              #采样大小
        self.epsilon = epsilon                  #greedy policy 参数
        self.epsilon_decay = epsilon_decay      #greedy policy 参数
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.target_update_iter = target_update_iter
        self.start_learning = start_learning

        self.optimizer = optim.Adam(params=model.parameters(), lr=self.lr)
        self.strategy = EpsilonGreedy(self.epsilon , self.min_epsilon, self.epsilon_decay)
        self.memory = ReplayMemory(self.buffer_sz)  #生成训练经验样例
        # create a experience tuple
        '''
        具名元组：由于元组不能对内部数据进行明明，所以引入具名元组
        exp: self.experience.state
        '''
        self.experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))
        self.num_actions = self.env.action_space.n

        # copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict()) 
        # 将目标_模型设置为评估模式。该网络仅用于推理。
        target_model.eval()

        self.current_step = 0

    def train(self, epochs, batch_sz):

        # training loop
        ep_rewards = [0.0]
        ep_steps = [0]

        # tracking of simulation 记录仿真结果
        produced_parts = []
        ep_rewards_mean = []
        #执行很多轮，每一轮都从一个新状态开始迭代
        for epoch in range(epochs):

            #重新生成初始状态
            state = self.env.reset()
            state = torch.FloatTensor([state]) # convert state to tensor

            while True:
                # select action according to e-greedy strategy
                action = self._select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                ep_rewards[-1] += reward    #当前轮将获得的系统奖励
                ep_steps[-1] += 1           #当前轮执行步骤
                
                # convert to tensors       
                next_state = torch.FloatTensor([next_state])
                reward = torch.FloatTensor([reward])
                action = torch.tensor([action], dtype=torch.int64)
                done = torch.tensor([done], dtype=torch.int32)
                #存储一个元组
                self.memory.store(self.experience(state, action, next_state, reward, done))

                # training of DQN model
                # 如果已经积累了足够数量的经验
                if self.memory.sample_possible(batch_sz):
                    # choose random experience from Replay Memory
                    experiences = self.memory.sample(batch_sz)
                    # separate states, actions, rewards and next_states from the minibatch
                    # zip 按列提取元素
                    states, actions, next_states, rewards, dones = zip(*experiences)
                    states = torch.cat(states)
                    actions = torch.cat(actions)
                    rewards = torch.cat(rewards)
                    next_states = torch.cat(next_states)
                    dones = torch.cat(dones)

                    # Input states of minibatch into model --> Get current Q-Value estimation of model
                    # 输入当前批量状态进入模型->得到当前模型Q值评估结果
                    # 将动作张量进行扩张，
                    index = actions.unsqueeze(-1) # 将动作张量转换为带有索引列表的张量
                    # ① 记录列向量索引 (0,0),(1,0),(2,0)... ②将dim=1换为索引值 (0,a),(1,b),(2,c)... ③ squeeze:[[][][]] -> []
                    current_q_values = self.model(states).gather(dim=1, index=index).squeeze() # squeeze to remove 1 axis

                    # DDQN
                    max_next_q_values_model_indices = self.model(next_states).argmax(1).detach()
                    index_ddqn = max_next_q_values_model_indices.unsqueeze(-1)
                    # Gather Q-Values of target_model for corresponding actions
                    next_q_values_from_target_of_model_indices = self.target_model(next_states).gather(dim=1,index=index_ddqn).squeeze() # squeeze to remove 1 axis
                    # Update target Q_values with Q-values of target_model based on max Q-values of model
                    # Q(state) = rs + γQ(next_state)
                    target_q_values = (next_q_values_from_target_of_model_indices*self.gamma)+rewards*(1-dones)
                    
                    # Calculate loss
                    loss = F.mse_loss(current_q_values, target_q_values)

                    # Set the gradients to zero before starting to do backpropragation with loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    # clip the gradients  修剪梯度
                    # 梯度在反向传播过程中会出现梯度小时/爆炸。所以最简单粗暴的方法是设定阈值，使得梯度不会超过阈值
                    clip=1
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    
                    # update params
                    self.optimizer.step()

                state = next_state
                # Logging and update of target_model
                if done:

                    ep_rewards.append(0.0)
                    ep_steps.append(0)
                    #???
                    produced_parts.append(len(self.env.system.sink_store.items))
                    #最近100次仿真的奖励
                    ep_rewards_mean.append(self._get_mean_reward(ep_rewards))

                    print('epoch:', epoch)
                    break

            # Copy weights from model to target_model every "target_update" setps
            # 策略网络迭代一定轮数后更新到目标网络中
            if epoch % self.target_update_iter == 0 and epoch != 0:
                self.target_model.load_state_dict(self.model.state_dict())

        ep_rewards = ep_rewards[:-1]

        return ep_rewards, produced_parts, ep_rewards_mean

    def _select_action(self, state):
        """
        Select action depending on exploration strategie (ε-greedy)
        """  
        self.exploration_rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step +=1

        #探索新的动作
        if self.exploration_rate > random.random():
            action = self.env.action_space.sample()
            return action  # agent explores
        else:
            # 找到当前状态的最佳动作
            # Turn off gradient tracking since we’re currently using the model for inference and not training.
            with torch.no_grad():
                #输入：状态， 输出：动作概率分布
                action = self.model(state).argmax(dim=1).item()
                return action #  agent exploits

    def _get_mean_reward(self, ep_rewards):
        """ mean reward over 100 episodes"""
        if len(ep_rewards) <= 100:
            return np.mean(ep_rewards[-len(ep_rewards):-1])
        else:
            return np.mean(ep_rewards[-100:-1])