#-*- coding = utf08 -*-
#@Time : 2021/2/3 23:14
#@Author : SWhite-Horse
#@File : Liner_cau.py
#@Software: PyCharm

#@ 思路： 求损失（MSE）最小时的 W

import matplotlib.pyplot as plt
import numpy as np
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w



def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2



w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)
min = w_list[0]
min_val = mse_list[0]

for i in range(1, len(w_list), 1):
    min = w_list[i] if mse_list[i] < min_val else min
    min_val = min_val if mse_list[i] > min_val else mse_list[i]
print('\n', "合理值：", min, "误差", min_val)

plt. plot(w_list, mse_list)
plt. ylabel("Loss")
plt. xlabel('w')
plt.annotate("Perfect", (min, min_val), xycoords='data',
             xytext=(min+0.5, min_val+1),
             arrowprops=dict(arrowstyle='->'))
plt. show()





