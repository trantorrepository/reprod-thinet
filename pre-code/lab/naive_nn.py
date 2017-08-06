#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-08-05 Sydney <theodoruszq@gmail.com>

"""
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # [Python类中super()和__init__()的区别](http://www.linuxidc.com/Linux/2016-10/136300.htm)
        # 建立了两个卷积层，self.conv1, self.conv2，注意，这些层都是不包含激活函数的
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120) # An affine operation: y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 注意，2D卷积层的输入data维数是 batchsize*channel*height*width
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square, you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.01)
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
net.zero_grad() # 对所有的参数的梯度缓冲区进行归零
out.backward(torch.randn(1, 10)) # 使用随机的梯度进行反向传播


for i in range(num_iterations):
    optimizer.zero_grad()   # zero the gradient buffers，如果不归0的话，gradients会累加
    output = net(input)     # 这里就体现出来动态建图了，你还可以传入其他的参数来改变网络的结构
    loss = criterion(output, target)
    loss.backward()         # Get grad, i.e. assign value to Variable.grad
    optimizer.step()        # Does the update. i.e. Variable.data -= learning_rate*Variable.grad


































