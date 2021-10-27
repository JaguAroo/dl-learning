import torch
import numpy as np
import include.d2l_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)),
                 dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def net(X):
    return d2l.softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


num_epochs, lr = 5, 0.1
d2l.train_ch3(net, train_iter, test_iter, d2l.cross_entropy, num_epochs,
              batch_size, [W, b], lr)
