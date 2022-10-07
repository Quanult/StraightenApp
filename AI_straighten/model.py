# encoding: utf-8

import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from AI_straighten.grid_sample import grid_sample
from torch.autograd import Variable
from AI_straighten.tps_grid_gen import TPSGridGen 
import AI_straighten.config as config
class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 61 * 61, 320)
        self.fc2 = nn.Linear(320, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(config.NUM_CLASSIFY)

    def forward(self, x):
        return F.log_softmax(self.cnn(x), dim=1)

class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)

class STNClsNet(nn.Module):

    def __init__(self):
        super(STNClsNet, self).__init__()
        
        r1 = config.SPAN_RANGE
        r2 = config.SPAN_RANGE
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (config.GRID_SIZE - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (config.GRID_SIZE- 1)),
        )))
        # print(target_control_points)
        Y, X = target_control_points.split(1, dim = 1)
        # print(Y, X)
        target_control_points = torch.cat([X, Y], dim = 1)
        # print(target_control_points)
        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }[config.BOUNDE_GRID_TYPE]
        self.loc_net = GridLocNet(config.GRID_SIZE,config.GRID_SIZE, target_control_points)

        self.tps = TPSGridGen(256, 256, target_control_points)

        self.cls_net = ClsNet()

    def stn(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 2)
        # transformed_x = grid_sample(x, grid)
        return grid# transformed_x
    
    def forward(self, x):
        grid = self.stn(x)
        transformed_x = grid_sample(x, grid)
        # logit = self.cls_net(transformed_x)
        return transformed_x

def get_model(arg):
    if arg == 'no_stn':
        print('create model without STN')
        # model = ClsNet()
    else:
        print('create model with STN')
        model = STNClsNet()
    return model

