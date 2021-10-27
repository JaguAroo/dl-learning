import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
import pandas as pd
import include.d2l_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('data/kaggle_house/train.csv')
test_data = pd.read_csv('data/kaggle_house/test.csv')
print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 所有79个特征 1~79 按样本相连
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)
