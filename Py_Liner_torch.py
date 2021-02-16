# -*- coding = utf08 -*-
# Time : 2021/2/14 10:58
# Author : SWhite-Horse
# File : Py_Liner_torch.py
# Software: PyCharm
import torch
import matplotlib.pyplot as plt
import numpy as np

x_data = torch.tensor([[1.0], [2.0], [3.0]])  # 张量的形式定义数据
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):  # 构造函数
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):  # 为重载，具体看表达式，linear是继承于Module的一个Class
        y = self.linear(x)
        return y


model = LinearModel()  # class的Call函数的应用，可以作为函数形式使用

criterion = torch.nn.MSELoss(size_average=False)  # 损失函数，参数为是否求平均，大部分情况下无所谓
optimzer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD的方法为参数优化，第一个参数得到所有的linear里的权重w，第二个参数定学习率

epoch_list = []
loss_list = []
for epoch in range(1000):  # 看用法
    y_val = model(x_data)
    loss = criterion(y_val, y_data)
    loss_list.append(loss.item())
    epoch_list.append(epoch)
    print(epoch+1, loss.item())

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()  # == updata（）
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])  # 注意为张量形式
y_test = model(x_test)
print('y_val = ', y_test.item())

plt. plot(epoch_list, loss_list)
plt. title("SGD")
plt. ylabel("Loss")
plt. xlabel('Epoch')
plt. show()
