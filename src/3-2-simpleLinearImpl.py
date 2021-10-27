import numpy as np
import torch
import torch.utils.data as Data
from torch import nn

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# for X, y in data_iter:
#     print(X, y)
#     break


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


# net = LinearNet(num_inputs)
# print(net)

# 使用nn.Seqyebtial
# 写法一
# net = nn.Sequential(nn.Linear(num_inputs, 1))

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

# 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))

print(net)
print(net[0])

# 打印可学习参数
# for param in net.parameters():
# print(param)

# 初始化模型参数
from torch.nn import init  # NOQA: E402

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

# 定义损失函数
loss = nn.MSELoss()
