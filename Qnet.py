"""

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

        self.fc4 = nn.Linear(50, num_actions)

    def forward(self, state):
        """

        :param state: 输入一个状态
        :return: 返回每个可选择的动作对应的Q值
        """
        outputs = self.fc1(state)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
        q_vector = self.fc4(outputs)

        return q_vector


def weight_initialization(module):
    if isinstance(module, nn.Linear):
        nn.init.normal(module.weight.data, std=0.1)
        nn.init.constant(module.bias.data, 0)


eval_net = QNet(num_state, num_action)
target_net = QNet(num_state, num_action)

eval_net.apply(weight_initialization)
target_net.apply(weight_initialization)
