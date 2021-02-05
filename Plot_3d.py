#-*- coding = utf08 -*-
#@Time : 2021/2/5 17:26
#@Author : SWhite-Horse
#@File : Plot_3d.py
#@Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

# 原始数据输入 #
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [4.0, 7.0, 10.0, 13.0, 16.0]


def pre_val(array):  # 矩阵最值函数 #
    min_mse = array[0][0]
    col_val, row_val = array.shape
    for i in range(col_val):
        for j in range(row_val):
            if array[i][j] <= min_mse:
                min_mse, col_pre, row_pre = array[i][j], i, j
            else:
                pass
    return min_mse, col_pre, row_pre  # 注意返回的时最小损失，列、行相应坐标 #


def back(x, w, b):  # 表达式 #
    return w * x + b


def loss(x, y, w, b):  # 一个数据点的损失 #
    y_pre = back(x, w, b)
    return (y - y_pre)**2


def mse_sum(w, b):  # 损失求和 #
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        loss_sum += loss(x_val, y_val, w, b)
    mse_val = loss_sum / 3
    return mse_val


w_list = np.linspace(0, 6, 61)  # 创建参数估计范围点阵
b_list = np.linspace(-4, 6, 101)

w_mesh, b_mesh = np.meshgrid(w_list, b_list, indexing='ij')
# 得到损失点阵、及最值点
mse_mesh = np.array(mse_sum(w_mesh, b_mesh))  # 注意这里是一个list，所以转化为array #
mse_pre, col, row = pre_val(mse_sum(w_mesh, b_mesh))
w_pre = w_mesh[col][row]
b_pre = b_mesh[col][row]
print(w_mesh, b_mesh, mse_mesh)

pic = plt.figure()  # 创建绘图窗口 #
sub = pic.add_subplot(111, projection='3d')  # 设置3d绘图 #
sub.plot_surface(w_mesh, b_mesh, mse_mesh)  # 要求使用 array 参数绘图 #
sub.set_xlabel(r'$w$')   # 标签 #
sub.set_ylabel(r'$b$')
sub.set_zlabel(r'$mse$')
sub.scatter(w_pre, b_pre, mse_pre, c='y')  # 突出显示最值点 #
sub.text(w_pre, b_pre, mse_pre, "w=%d,b=%d,mse=%d'\n'Prefect"%(w_pre, b_pre, mse_pre), c='r')  # 标注最值点 #
plt.show()

