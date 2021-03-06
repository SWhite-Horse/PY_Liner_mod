# -*- coding = utf08 -*-
# Time : 2021/2/16 16:23
# Author : SWhite-Horse
# File : Mul_Dim_Log_Regression.py
# Software: PyCharm
import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)  # 1、文件名（当前文件夹）；2、分隔符；3、数据类型
x_data = torch.from_numpy(xy[:, :-1])  # 除过最后一列的所有行列
y_data = torch.from_numpy(xy[:, [-1]])  # 矩阵形式 x*1


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

criterion = torch.nn.BCELoss(size_average=True)  # 二分类的损失函数，参数为是否求平均，大部分情况下无所谓
optimzer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD的方法为参数优化，第一个参数得到所有的linear里的权重w，第二个参数定学习率


for epoch in range(1000):  # 看用法
    y_val = model(x_data)
    loss = criterion(y_val, y_data)
    print(epoch+1, loss.item())

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()  # == updata（）

# x = np.linspace(0, 10, 201)
# x_t = torch.Tensor(x).view((201, 1))  # 类似于reshape，可以变成一个矩阵
# y_t = model(x_t)
# y = y_t.data.numpy()  # 得到了一个数组
# print(x, '\n', y)
# # 绘图可以看出，2.5是一个分界点，原因是我们的数据集导致的平均通过时间就是2.5
# plt. plot(x, y)
# plt. plot([0, 10], [0.5, 0.5], c='r')
# plt. xlabel('Hours')
# plt. ylabel('Probability of Pass')
# plt. grid()
# plt. show()
