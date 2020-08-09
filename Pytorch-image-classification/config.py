#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Authors:hujun06
Date:2020-06-15
"""

#config
epoch = 100

#the config of model
num_classes = 81
learning_rate = 0.002
batch_size = 256
val_batch_size = 256
input_size=(128, 128)
n_worker = 2
checkpoint_interval = 2


##training file path
file_dir = 'xxx'
train_file_path = file_dir +'/train.txt'
valid_file_path = file_dir + '/valid.txt'
test_file_path =  file_dir +'/test.txt'
data_dir = 'xxx'

#if load the pretrained model weight
load_model = True
model_path = './checkpoints/xxx.pkl'
test_weigth_path = './checkpoints/xxx.pkl'

##save_dir and the save_name
model_save_dir = "checkpoints"
model_name = 'model_name'
