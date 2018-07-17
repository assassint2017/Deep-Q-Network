"""

这里就是和原始的DQN的区别之处了
原始的DQN输入一个状态，输出的是对应动作的Q值
而dueling DQN有两个输出，一个是对于输入状态的描述值(单一值)另外一个是对应每种动作的优势值(维度和动作值相同)
最终，每种动作的Q值为：状态值+每种动作的优势值(通过广播加法的形式)

创建Q网络，这里就使用简单的全连接网络即可
"""

import torch.nn as nn

num_state = 4
num_action = 2


class QNet(nn.Module):
    def __init__(self, num_states, num_actions):
        """

        :param num_states: 输入的状态数量
        :param num_actions: 输出的动作数量
        """
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(num_states, 50),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU()
        )

        self.fcV = nn.Linear(50, 1)
        self.fcA = nn.Linear(50, num_actions)

    def forward(self, state):
        """

        :param state: 输入一个状态
        :return: 返回每个可选择的动作对应的Q值
        """
        outputs = self.fc1(state)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)

        advantage = self.fcA(outputs)
        state_value = self.fcV(outputs)

        q_vector = state_value + advantage

        return q_vector


def weight_initialization(module):
    if isinstance(module, nn.Linear):
        nn.init.normal(module.weight.data, std=0.1)
        nn.init.constant(module.bias.data, 0)


eval_net = QNet(num_state, num_action)
target_net = QNet(num_state, num_action)

eval_net.apply(weight_initialization)
target_net.apply(weight_initialization)


# # 测试GPU CPU速度代码
# # 发现GPU比CPU稍慢，所以放弃使用GPU
# import torch
# from time import time
#
# eval_net = eval_net.cuda()
# start_time = time()
# data = torch.randn((32, 4)).cuda()
# res = eval_net(data)
# print(time() - start_time)
