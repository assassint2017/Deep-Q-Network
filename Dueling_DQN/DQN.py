"""

构建DQN
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from Dueling_Qnet import eval_net
from Dueling_Qnet import target_net

batch_size = 32
leaing_rate = 1e-2

memory_pool_capacity = 2000  # 经验池的大小
target_net_update = 100  # 目标网络每多少步更新一次

num_state = 4  # 一个observation由几个数字构成
num_action = 2  # 有哪几种可能选择的动作

gamma = 0.9  # 未来奖励的折扣因子

# e-greedy选取动作策略
epsilon_min = 0.5  # 最开始的e值
epsilon_max = 0.9  # 最大的e值

module_dir = './module/'  # 保存训练好的模型的路径


class DQN(object):
    def __init__(self):

        # 记忆库的容量
        self.memory_pool_capacity = memory_pool_capacity

        # 构建记忆库
        self.memory_pool = np.zeros((memory_pool_capacity, num_state * 2 + 3))

        # 用于记录记忆库当前的大小
        self.memory_pool_size = 0

        # 用于记录网络已经更新了多少步
        self.update_step = 0

        # 定义优化器和损失函数
        self.optimizer = torch.optim.Adam(eval_net.parameters(), leaing_rate)
        self.loss_func = nn.MSELoss()

        # 对于epsilon的数值，在训练初期应该是相对较低，这样有利于进行探索
        # 随着不断的训练，应该逐渐提升epsilon的数值，因为这时候网络的预测结果正在变得越来越精确，越可以进行贪心选择
        self.epsilon = epsilon_min

    def choose_action(self, observation):
        """

        :param observation: 输入状态
        :return: 按照e-greedy策略进行选择动作
        """
        if self.memory_pool_size >= self.memory_pool_capacity:
            self.epsilon += 2e-5

        if self.epsilon > epsilon_max:
            self.epsilon = epsilon_max

        if random.uniform(0, 1) < self.epsilon:  # 执行贪心
            observation = torch.FloatTensor(observation).unsqueeze(dim=0)
            with torch.no_grad():
                action_q_vecotr = eval_net(observation)
            action = int(torch.max(action_q_vecotr, dim=1)[1])

        else:  # 随机选取动作
            action = random.randint(0, num_action - 1)

        return action

    def store_memory(self, s, a, r, s_, done):
        """

        :param s: 当前状态 ndarray
        :param a: 在当前状态下执行的动作 int
        :param r: 在当前状态下执行动作后得到的奖励 float
        :param s_: 在当前状态下执行的动作后进行的下一状态 ndarray
        :param done: 在当前状态下执行动作后有没有导致本次episode结束的标志位
        """
        if done is True:
            done = 0
        else:
            done = 1

        memory = np.hstack((s, a, r, s_, done))

        # 用新的记忆来覆盖老的记忆
        index = self.memory_pool_size % memory_pool_capacity

        self.memory_pool[index, :] = memory
        self.memory_pool_size += 1

    def update_parameter(self):

        # 每间隔一定的时间才更新一次目标网络
        if self.update_step % target_net_update is 0:
            target_net.load_state_dict(eval_net.state_dict())

        # 每次都要对eval网络进行更新
        # 随机从记忆库中抽取mini batch个数据进行训练网络
        sample_index = np.random.choice(memory_pool_capacity, batch_size)
        memory_batch = self.memory_pool[sample_index, :]

        observation_batch = torch.FloatTensor(memory_batch[:, :num_state])
        action_batch = torch.LongTensor(memory_batch[:, num_state]).unsqueeze(dim=1)
        reward_batch = torch.FloatTensor(memory_batch[:, num_state + 1]).unsqueeze(dim=1)
        next_observation_batch = torch.FloatTensor(memory_batch[:, num_state + 2:-1])
        done_batch = torch.FloatTensor(memory_batch[:, -1]).unsqueeze(dim=1)

        # 估计的q值 q_eval
        q_eval = eval_net(observation_batch).gather(1, action_batch)

        # 现实中的q值 q_target，这里由于不用对target网络进行更新，因此仅使用farward模型即可
        # 这里许多乘一个done_batch的判断位信息，因为如果本回合已经结束，这时候的q_target应该应该只包括当前状态的奖励
        # 而不包括下一状态的最大Q值
        with torch.no_grad():
            q_target = reward_batch + gamma * target_net(next_observation_batch).max(1)[0].unsqueeze(dim=1) * done_batch
        # 根据现实和估计的q值之间的差距计算loss并进行反向传播
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_step += 1

    def save(self, score):
        """

        :param score: 根据本轮episode获得的reward
        :return: 保存模型
        """
        torch.save(eval_net.state_dict(), os.path.join(module_dir, 'best_reward_{:.1f}.pth'.format(score)))


dqn = DQN()
