# -*- coding = utf08 -*-
# Time : 2021/2/22 12:02
# Author : SWhite-Horse
# File : Dataset_division.py
# Software: PyCharm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]


dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # 关键句，自动构造数据集


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 其它激活函数只需替换这行即可，但是Rule在最后一个仍用sigmoid，因为其间断性

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for i, data in enumerate(dataset, 0):
        input, label = data
        y_val = model(input)
        loss = criterion(y_val, label)
        print(epoch+1, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
