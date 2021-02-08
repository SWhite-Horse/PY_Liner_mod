# -*- coding = utf08 -*-
# Time : 2021/2/6 23:14
# Author : SWhite-Horse
# File : Liner_cau.py
# Software: PyCharm
# y=x^2-3x+2

import torch
import matplotlib.pyplot as plt
import numpy as np

x_data = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
y_data = [6.0, 2.0, 0, 0, 2.0, 6.0]

w1 = torch.tensor([0.0])  # 要float类型
w2 = torch.tensor([-4.0])  # 要float类型
b = torch.tensor([1.0])  # 要float类型

# w = torch.tensor([item.cpu().detach().numpy() for item in w]).cuda()
w1.requires_grad = True  # 需要梯度
w2.requires_grad = True
b.requires_grad = True


def forward(x):
    return w1 * x**2 + w2 * x + b


def loss(x, y):
    y_temp = forward(x)
    return (y - y_temp)**2


print(("predict(before training)", 5.0, forward(5).item()))

for epoch in range(10000):
    loss_temp = 0
    for x_val, y_val in zip(x_data, y_data):
        loss_temp = loss(x_val, y_val)
        loss_temp.backward()
        print("'\tgrad:", x_val, y_val, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.0001 * w1.grad.item()  # item() 函数有个问题，converted 那个，解决方案在收藏夹
        w2.data = w2.data - 0.0001 * w2.grad.item()
        b.data = b.data - 0.0001 * b.grad.item()
        w1.grad.data.zero_()  # 梯度清零，必须，zero_ ！！！
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress", epoch+1, " ", loss_temp.item())  # 问题：0.001精度足够接近，但是无法完全拟合
print("predict(after training)", 5.0, forward(5).item())


