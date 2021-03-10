# -*- coding = utf08 -*-
# Time : 2021/2/22 16:48
# Author : SWhite-Horse
# File : Minist.py
# Software: PyCharm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim


batch_size = 64
transform = transforms.Compose([      # 这个函数是将图像转为张量
    transforms.ToTensor(),            # 转化为张量表，C(层） * Widen * High
    transforms.Normalize((0.1307, ), (0.3081, ))   # 标准化（否则可能导致梯度爆炸），第一个参数为均值，二为标准差
])

train_dataset = datasets.MNIST(root='../dataset/mnist', train=True,
                               transform=transforms, download=True)  # True 为训练集
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False,
                              transform=transforms, download=True)  # transform就是处理函数，上面定义的
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 线性全链接需要一个输入是一个线性向量，这个参数就是用总输入大小（一个batch）（N * 784） 除以784（-1的作用）
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)


model = Model()

criterion = torch.nn.CrossEntropyLoss(size_average=True)  # 交叉熵损失，Y^ = Log[softmax], Loss = -Y * log[Y^]
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)  # 最后一个是冲量大小

train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, 32, shuffle=True)
test_loader = DataLoader(test_dataset, 32, shuffle=False)


def train(epoch):
    running_loss = 0.0
    for batch_index, (inputs, target) in enumerate(train_loader, 0):  # 枚举函数返回第一个为下标，第二个为数据
        optimizer.zero_grad()  # 优化器清零
        y_val = model(inputs)   # 与下一步为 forword 前馈
        loss = criterion(y_val, target)

        loss.backward()  # 反馈
        optimizer.step()  # 优化

        running_loss += loss.item()
        if batch_index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # with 语句，以下代码就不会计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            ''' model返回值是一个矩阵，每一行有10个数值，
                最大的表示第几类，max函数会返回（每行最大值，最大值下标）
                dim = 1 ,表示延第二个维度寻找，也就是行'''
            total += labels.size(0)  # 由于labels是一个 N * 1，存储了标签，size就是一个元组（N，1), size(0)就是 N。此式计算总的图片个数
            correct += (predicted == labels).sum().item()  # 累加正确判断个数
    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
