#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-08-05 Sydney <theodoruszq@gmail.com>

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

# NOTE: when `pin_memory` = True, you can ONLY use CUDA devices to compute

def loader(path, batch_size=32, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader
    return data.DataLoader(
    # http://pytorch.org/docs/master/torchvision/datasets.html
        datasets.ImageFolder(path,
    # http://pytorch.org/docs/master/torchvision/transforms.html#torchvision-transforms
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=32, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


#train_d = loader("../dataset/kaggle_train")
#test_d  = test_loader("../dataset/kaggle_test")
#for item in test_d:
#    print(item)
