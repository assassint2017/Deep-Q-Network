# 强化学习记录

## 在强化学习中的一些常见名词

* Agent：智能体
* Environment：环境
* Obervation：观察（对环境做出观察）
* State：状态
* Action：动作
* Reward：奖励
* Policy：策略，状态State到动作Action的过程就称之为一个策略
* off-policy：离线学习，可以从过往的经验中学习
* on-policy：在线学习，现学现卖
* MDP（Markov Decision Process）马尔科夫决策过程：未来的变化只跟当前的状态相关，和过去没有关系

## 强化学习的一般的流程：
* Agent对Environment进行Obervation得到当前state，并根据当前的State基于一定的policy做出Action，得到Reward，进入到下一State，如此循环往复  
* Agent所做的每一轮决策，称为一个episode，跟美剧里的“集”单位一样，这里指的应该是完成一次实验，比如寻宝的话就是要么走入陷阱摔死了，要么就是寻到了宝藏，总之就是一次实验结束了

## 强化学习分类
|算法名称|modle base|module free|policy based（基于概率）|value based（基于价值）|回合更新|单步更新|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Q-learning|n|y|n|y|n|y|
|sarsa|n|y|n|y|n|y|
|DQN|n|y|n|y|n|y|
|policy gradients|n|y|y|n|y|n|
|actor-critic|n|y|y|y|n|y|

* modle base相对比modle free就是多了一个对真实世界建模的环节，之后不管是在虚拟环境还是真实环境中学习，方法都是modle free中的方法
* 由于单步更新的策略往往更加高效，而且很多时候问题并不存在回合，因此现在的方法大多都在向单步更新靠拢


## Q-learning（off-policy）
* Q-learning的Q代表的是Quality
* Q-table：以state为行、action为列
* Bellman Equation：贝尔曼方程，更新Q-table的依据
* Q(s, a)，价值函数，描述的是在当前的状态下做出的动作的长期奖励期望

## sarsa（on-policy）
和Q-learning非常相似的一种学习方式，同样使用Q表，只是Q-target的值选取原则不同，总体来说还是非常类似的

## dqn（off-policy）
经验池，fix Q target，目的都是为了消除数据之间的相关性，为使用神经网络做的准备

DQN原始论文：https://arxiv.org/abs/1312.5602
Double DQN：https://arxiv.org/abs/1509.06461
Prioritized Experience Replay DQN：https://arxiv.org/abs/1511.05952
Dueling DQN：https://arxiv.org/abs/1511.06581

## policy gradient
输入状态，输出对应动作的概率，相对于Q-learning，可以处理连续控制的问题

## actor-critic
结合了基于动作概率和基于动作估计的两种算法，取长朴短
