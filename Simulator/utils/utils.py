import pygame as pg
import numpy as np
from numpy import floor
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from perlin_numpy import generate_perlin_noise_2d

from planning.A_star import solve


import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn



input_size = 42
num_classes = 6

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

        self.conv0 = nn.Conv2d(1, 20, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(20, 40, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(40, 80, 3, stride=1, padding=1)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(82, 150)
        self.linear2 = nn.Linear(150, 100)
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 25)
        self.linear5 = nn.Linear(25, num_classes)

        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x, g):
        # print(xb.shape)
        out = self.conv0(x)
        out = self.act(out)
        # print(out.shape)
        out = self.conv1(out)
        out = self.act(out)
        # print(out.shape)
        out = self.conv2(out)
        out = self.act(out)
        # print(out.shape)

        out = self.adaptivepool(out)
        # print(out.shape)
        out = torch.cat((out, g), 1)
        # print(out.shape)

        out = self.flat(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.linear4(out)
        out = self.act(out)
        out = self.linear5(out)
        return(out)
    
    def training_step(self, batch):
        images, goals, labels = batch
        out = self(images, goals) ## Generate predictions
        loss = self.loss_fn(out, labels) ## Calculate the loss
        return(loss)
    
    def validation_step(self, batch):
        images, goals, labels = batch
        out = self(images, goals)
        # labels = F.one_hot(labels, num_classes)
        # out = torch.argmax(out, dim=1) 
        # out = torch.argmax(out, dim=1) 
        # print(out.shape, labels.shape)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def epoch_end(self, epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
        




