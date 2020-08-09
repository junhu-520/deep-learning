#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Authors:hujun06
Date:2020-06-15
training the model
"""

from torchvision.transforms import transforms
from torchvision import models
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
from utils.data_deal import CustomDataset, get_labels
import time
import torch.optim as optim
import config as cfg
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from utils.utils import AverageMeter, accuracy
from utils.progress_bar import ProgressBar
import torch.nn as nn
import torch



train_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=100, interpolation=PIL.Image.BILINEAR),
    transforms.CenterCrop(size=(cfg.input_size, cfg.input_size)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载训练集和验证集
labels_train = get_labels(cfg.train_file_path)
labels_val = get_labels(cfg.valid_file_path)

# 为训练集创建加载程序
train_data = CustomDataset(cfg.data_dir, labels_train, transform=train_transformations, size=cfg.input_size)
train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.n_worker, drop_last=True)

#验证集
val_data = CustomDataset(cfg.data_dir, labels_val, transform=test_transformations, size=cfg.input_size)
val_loader = DataLoader(val_data, batch_size=cfg.val_batch_size, shuffle=True, num_workers=cfg.n_worker)

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.fc = nn.Linear(num_ftrs, cfg.num_classes)
if cfg.load_model:
    print("from   " + cfg.model_path + "   load model")
    model = torch.load(cfg.model_path)
    print("load model ok--------")
model.cuda()

from torchsummary import summary
summary(model, input_size=(3, 128, 128))

# 定义优化器和损失函数
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.RMSprop(model.parameters(), lr=cfg.learning_rate, alpha=0.99, eps=1e-08)

def save_models(epoch, save_path):
    """
    :param epoch:
    :return:
    """
    torch.save(model, save_path)
    print("Chekcpoint saved")


def evaluate(val_loader, model, criterion, epoch):
    """
    :param val_loader:
    :param model:
    :param criterion:
    :param epoch:
    :return:
    """
    #2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    #progress bar
    val_progressor = ProgressBar(mode="Val  ",
                                 epoch=epoch,
                                 total_epoch=cfg.epoch,
                                 model_name=cfg.model_name,
                                 total=len(val_loader))
    #2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            val_progressor.current = i
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            target = target.squeeze()
            #print("target:", target)
            #2.2.1 compute output
            output = model(input)
            out = torch.argmax(F.softmax(output, dim=1), 1)
#            print("out:", out)
            loss = criterion(output, target)
            #2.2.2 measure accuracy and record loss
            precision1, precision2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
    return [losses.avg, top1.avg]


def train(epochs, cuda_avail=True):
    """
    :param epochs:
    :return:
    """
    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 若GPU可用，将图像和标签移往GPU
            if cuda_avail:
                images = Variable(images).cuda()
                labels = Variable(torch.from_numpy(np.array(labels)).long()).cuda()
            # 用来自测试集的图像预测类
#            print(images.shape)
            outputs = model(images)
            labels = labels.squeeze()
#            print("labels:", labels)
#            out =  torch.argmax(F.softmax(outputs, dim=1), 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()                           # 计算梯度
            optimizer.step()                          # 反向传播
            running_loss += loss.item()
            if i % 1000 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
        if epoch % cfg.checkpoint_interval == 0:
            save_models(epochs, cfg.model_save_dir + '/' + cfg.model_name + "_{}.pkl".format(epoch))
#        print('the val nums:', len(val_loader))
        valid_loss = evaluate(val_loader, model, criterion, epoch)
    print('Finished Training! Total cost time: ', time.time() - start)

if __name__ == "__main__":
    train(cfg.epoch)
