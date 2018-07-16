"""

OpenAI的官网
http://gym.openai.com/docs/
安装gym只需要使用下面的命令
pip install gym
如果是要安装全套的环境，则需要加上all
pip install  gym[all]
"""

import gym
from gym import envs

# 查看当前所有可使用游戏的代码
for item in envs.registry.all():
    print(item)

# 选择游戏，当偏离15度或者移动超过2.4个单位，游戏结束
env = gym.make('CartPole-v0')

# 一个environment包含两个空间：action_space 和observation_space
print(env.action_space)
# Discrete(2)
print(env.observation_space)
# Box(4,)
# Discrete代表非负数，2意思就是动作的取值为0,1，对于CartPole这个环境，0代表小车向左，1代表向右
# Box就是返回数组的维度，因此，一个observation代表着4个数字，也就是一个状态由四个数字决定

# 0 小车的位置
# 1 小车的速度
# 2 木棒的角度:-41.8°	41.8°
# 3 木棒的速度

# 返回observation的最值
print(env.observation_space.high)
# [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low)
# [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
