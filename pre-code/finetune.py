#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-08-05 Sydney <theodoruszq@gmail.com>

"""
"""

import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.optim as optim

import cv2
import sys
import numpy as np
import argparse
import os

import dataset


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()
        model = models.vgg16(pretrained = True)
        self.features = model.features  # Pretrained model divides itself to features(Map) and classifier(FC)

        for param in self.features.parameters():
            param.requires_grad = False

        # Origin classifier
        # Sequential (
        # (0): Linear (25088 -> 4096)
        # (1): ReLU (inplace)
        # (2): Dropout (p = 0.5)
        # (3): Linear (4096 -> 4096)
        # (4): ReLU (inplace)
        # (5): Dropout (p = 0.5)
        # (6): Linear (4096 -> 1000))
        self.classifier = nn.Sequential(
                nn.Dropout(),               # Default p = 0.5
                nn.Linear(25088, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        # https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader  = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        #self.prunner = FilterPr
        self.model.train()  #??

    def test(self):
        self.model.eval()  #??
        correct = 0
        total   = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if torch.cuda.is_available:
                batch = batch.cuda()
            output = model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy:", float(correct) / total)
        self.model.train()

    def train(self, optimizer = None, epoches = 10):
        if optimizer == None:
            optimizer = optim.SGD(model.classifier.parameters(), 
                                lr = 0.0001, momentum = 0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning...")

    def train_epoch(self, optimizer = None, rank_filters = False):
        if torch.cuda.is_available is True:
            for batch, label in self.train_data_loader:
                self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)
        else:
            for batch, label in self.train_data_loader:
                self.train_batch(optimizer, batch, label, rank_filters)


    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        inp = Variable(batch)
        self.criterion(self.model(inp), Variable(label)).backward()
        optimizer.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune",  dest="prune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path",  type=str, default="test")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.train:
        model = ModifiedVGG16Model().cuda()
    
    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches = 20)





