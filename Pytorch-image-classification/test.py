#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Authors:hujun
Date:2020-07-16
"""


from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils.data_deal import CustomDataset, get_labels
import config as cfg
from utils.utils import AverageMeter, accuracy
from utils.progress_bar import ProgressBar
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

def evaluate(val_loader, model, criterion):
    """
    evaluate the model
    """
    #2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    #progress bar
    val_progressor = ProgressBar(mode="Val  ", epoch=0, total_epoch=0, model_name=cfg.model_name, total=len(val_loader))
    #2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    tru_labels_all = []
    pred_prob_all = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            val_progressor.current = i
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            target = target.squeeze()
            #the output of model
            output = model(input)
            #argmax
            preds = torch.argmax(F.softmax(output, dim=1), 1)
            tru_labels_all.append(target.cpu().numpy())
            pred_prob_all.append(preds.cpu().numpy())
            loss = criterion(output, target)
            precision1, precision2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
    labels = np.concatenate(tru_labels_all)
    probs = np.concatenate(pred_prob_all)
    return [losses.avg, top1.avg, labels, probs]

if __name__ == "__main__":
    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # val
    labels_val = get_labels(cfg.test_file_path)
    val_data = CustomDataset(cfg.data_dir, labels_val, transform=test_transformations, size=cfg.input_size)
    val_loader = DataLoader(val_data, batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.n_worker)
    criterion = nn.CrossEntropyLoss().cuda()
    model = torch.load(cfg.test_weigth_path)
    [losses, top1, labels, probs] = evaluate(val_loader, model, criterion, 0)
    from sklearn.metrics import classification_report
    print(classification_report(labels, probs))
