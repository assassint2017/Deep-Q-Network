"""

训练脚本

进行max_episode场游戏，并最终保存在一次episode中获得reward最大的模型
"""

import os
from time import time

import gym
from DQN import dqn

max_episode = 250  # 最大迭代的次数

env = gym.make('CartPole-v0')  # 选择游戏
env = env.unwrapped

max_reward = 0

os.system('rm ./module/best*')
start_time = time()
for episode in range(max_episode):

    print('--------------------------')
    print(episode, 'episode beginning')

    episode_reward = 0  # 本次迭代的奖励值

    # 初始化环境，并拿到初始的状态
    observation = env.reset()

    while True:
        env.render()

        # step函数会产生四个返回值
        # observation (object): 执行完上一个动作之后进入的状态（下一状态）
        # reward(float): 对于刚刚执行的动作的奖励值
        # done(boolean): 标志本次episode是否结束
        # info(dict): 对于算法，貌似没有什么太大作用

        # 选取动作
        action = dqn.choose_action(observation)

        # 执行上一步中选取好的动作，得到环境的反馈
        observation_, reward, done, info = env.step(action)

        # 环境直接返回的reward不容易进行训练，我们可以使用自己编写的reward进行训练
        # 基于价值估计的强化学习算法，如何设计reward也是非常重要的
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # 这里的reawrd的意义在于小车偏离中心点越多，奖励越小，杆子偏离角度越大，奖励越小
        reward = r1 + r2

        # 存储记忆，也就是存放 s,a,r,s' 序列
        dqn.store_memory(observation, action, reward, observation_)

        # 记录本次episode的总奖励值
        episode_reward += reward

        # 这时，如果记忆库存储达到上限，则开始更新神经网络参数
        if dqn.memory_pool_size >= dqn.memory_pool_capacity:
            dqn.update_parameter()

        # 如果本episode结束，重新对环境进行初始化
        if done is True:
            print(episode, 'episode finished')
            print('get reward:', episode_reward)
            if episode_reward > max_reward:
                max_reward = episode_reward
                os.system('rm ./module/best*')
                dqn.save(max_reward)
            print('already use {:.2f} min'.format((time() - start_time) / 60))
            break

        # 将下一状态当做当前状态
        observation = observation_

print('training finished!')
env.close()
