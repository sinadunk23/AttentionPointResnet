#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

  
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)

        return x1,x2


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        return x2,x1
    


class PointResNet(nn.Module):
    def __init__(self, args):
        super(PointResNet, self).__init__()
        self.args = args
        dims = [64, 64, 64, 128, args.emb_dims]
        self.conv1 = nn.Conv1d(3, dims[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims[0], dims[1], kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(dims[1], dims[2], kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(dims[2], dims[3], kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(dims[3], dims[4], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(dims[0])
        self.bn2 = nn.BatchNorm1d(dims[1])
        self.bn3 = nn.BatchNorm1d(dims[2])
        self.bn4 = nn.BatchNorm1d(dims[3])
        self.bn5 = nn.BatchNorm1d(dims[4])

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + x1
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = x3 + x2
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)

        return x1,x2

class AttentionPointResNet(nn.Module):
    def __init__(self, args):
        super(AttentionPointResNet, self).__init__()
        self.args = args
        dims = [64, 64, 64, 128, args.emb_dims]
        self.conv1 = nn.Conv1d(3, dims[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims[0], dims[1], kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(dims[1], dims[2], kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(dims[2], dims[3], kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(dims[3], dims[4], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(dims[0])
        self.bn2 = nn.BatchNorm1d(dims[1])
        self.bn3 = nn.BatchNorm1d(dims[2])
        self.bn4 = nn.BatchNorm1d(dims[3])
        self.bn5 = nn.BatchNorm1d(dims[4])

        self.projq1 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.projk1 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.projv1 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm1d(64)
        self.projq2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.projk2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.projv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn7 = nn.BatchNorm1d(64)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + F.relu(self.bn6(attention(self.projq1(x1), self.projk1(x1), self.projv1(x1))))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = x3 + F.relu(self.bn7(attention(self.projq2(x2), self.projk2(x2), self.projv2(x2))))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)

        return x1,x2
    

def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)
  
class SelfAttentionModule(nn.Module):
    def __init__(self, in_dim = 64, hidden_dim = 64, dropout = None):
        super(SelfAttentionModule, self).__init__()
        self.projq = nn.Conv1d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.projk = nn.Conv1d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.projv = nn.Conv1d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(hidden_dim)
    def forward(self, x):
        return F.relu(self.bn(attention(self.projq(x), self.projk(x), self.projv(x), self.dropout)))
    
class AttentionPointResNetV2(nn.Module):
    def __init__(self, args):
        super(AttentionPointResNetV2, self).__init__()
        self.args = args
        dims = [64, 64, 64, 128, args.emb_dims]
        self.conv1 = nn.Conv1d(3, dims[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims[0], dims[1], kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(dims[1], dims[2], kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(dims[2], dims[3], kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(dims[3], dims[4], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(dims[0])
        self.bn2 = nn.BatchNorm1d(dims[1])
        self.bn3 = nn.BatchNorm1d(dims[2])
        self.bn4 = nn.BatchNorm1d(dims[3])
        self.bn5 = nn.BatchNorm1d(dims[4])

        self.attn1 = SelfAttentionModule(dims[0], dims[1])
        self.attn2 = SelfAttentionModule(dims[1], dims[2])
        self.attn3 = SelfAttentionModule(dims[2], dims[3])

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + self.attn1(x1)
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = x3 + self.attn2(x2)
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x4 = x4 + self.attn3(x3)
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        return x1,x2