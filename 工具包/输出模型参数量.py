import timm
import torch
from torch import nn



'''
写一个main文件
模型的创建
优化器的创建
学习率的初始化
损失函数的创建

训练函数
验证函数
输出准确率
'''




model = None
sum([m.numel() for m in model.parameters()])