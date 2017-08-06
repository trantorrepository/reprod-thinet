#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-08-05 Sydney <theodoruszq@gmail.com>

"""
"""

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad = True)
y = x + 2
print(y.creator) # y 是作为一个操作的结果创建的因此y有一个creator, 但是 x 没有

z = y * y * 3
out = z.mean()
print(out)

# Now backprop
out.backward() # == out.backward(torch.Tensor([1.0]))
print(x.grad)


