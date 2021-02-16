# -*- coding = utf08 -*-
# Time : 2021/2/16 12:07
# Author : SWhite-Horse
# File : Logistic_Regression.py
# Software: PyCharm

import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])  # 张量的形式定义数据
y_data = torch.tensor([[0.0], [0.0], [1.0]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y = f.sigmoid(self.linear(x))
        return y


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)  # 二分类的损失函数，参数为是否求平均，大部分情况下无所谓
optimzer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD的方法为参数优化，第一个参数得到所有的linear里的权重w，第二个参数定学习率


for epoch in range(1000):  # 看用法
    y_val = model(x_data)
    loss = criterion(y_val, y_data)
    print(epoch+1, loss.item())

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()  # == updata（）
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x = np.linspace(0, 10, 201)
x_t = torch.Tensor(x).view((201, 1))  # 类似于reshape，可以变成一个矩阵
y_t = model(x_t)
y = y_t.data.numpy()  # 得到了一个数组
print(x, '\n', y)
# 绘图可以看出，2.5是一个分界点，原因是我们的数据集导致的平均通过时间就是2.5
plt. plot(x, y)
plt. plot([0,10], [0.5, 0.5], c='r')
plt. xlabel('Hours')
plt. ylabel('Probability of Pass')
plt. grid()
plt. show()
