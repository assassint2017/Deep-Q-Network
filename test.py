"""

测试脚本
测试max_episode场游戏的平均得分

这时使用已经训练好的网络，使用完全贪心的策略进行动作的选取
"""

import gym
import torch

from Qnet import eval_net

max_episode = 5  # 进行5场游戏的测试

env = gym.make('CartPole-v0')  # 选择游戏
env = env.unwrapped

module_dir = './module/best_reward_628.0.pth'
eval_net.load_state_dict(torch.load(module_dir))

reward_list = []

for episode in range(max_episode):

    print('--------------------------')
    print(episode, 'episode beginning')

    episode_reward = 0  # 本次迭代的奖励值

    # 初始化环境，并拿到初始的状态
    observation = env.reset()

    while True:
        env.render()

        # 选取动作
        observation = torch.FloatTensor(observation).unsqueeze(dim=0)
        with torch.no_grad():
            q_vector = eval_net(observation)
            action = int(torch.max(q_vector, dim=1)[1])

        # 执行上一步中选取好的动作，得到环境的反馈
        observation_, reward, done, info = env.step(action)

        # 记录本次episode的总奖励值
        episode_reward += reward

        # 如果本episode结束，重新对环境进行初始化
        if done is True:
            print(episode, 'episode finished')
            print('get reward:', episode_reward)
            reward_list.append(episode_reward)
            break

        # 将下一状态当做当前状态
        observation = observation_

mean_reward = sum(reward_list) / max_episode

print('------------------')
print('testing finished!')
print('mean reward:', mean_reward)
env.close()
