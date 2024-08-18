import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG9
class Net(nn.Module):
    def __init__(self, chunksize):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256, track_running_stats=False)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256, track_running_stats=False)
        
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256, track_running_stats=False)
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(512, track_running_stats=False)
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512, 10)
        
        self.chunksize = chunksize
        self.param_list = [self.conv1, self.bn1, self.conv2, self.bn2,
                           self.conv3, self.bn3, self.conv4, self.bn4,
                           self.conv5, self.bn5, self.conv6, self.bn6, 
                           self.conv7, self.bn7, self.conv8, self.bn8, self.fc]
                           # self.fc1, self.fc2, self.fc3]
        self.total_param, self.coef_size = self._count_param()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    @torch.no_grad()
    def compute_grad(self, global_model):
        for self_nn, glb_nn in zip(self.param_list, global_model.param_list):
            self_nn.weight.data -= glb_nn.weight.data
            if self_nn.bias is not None:
                self_nn.bias.data -= glb_nn.bias.data
    
    @torch.no_grad()
    def _count_param(self):
        total_param = 0
        for param in self.param_list:
            total_param += param.weight.numel()
            if param.bias is not None:
                total_param += param.bias.numel()

        if total_param % self.chunksize:
            coef_size = ((total_param // self.chunksize) + 1) * self.chunksize
        else:
            coef_size = total_param
        return total_param, coef_size
    
    @torch.no_grad()
    def _partition(self):
        torch.manual_seed(-1)
        all_param = torch.randn(self.coef_size, dtype=self.param_list[0].weight.dtype, device=self.param_list[0].weight.device)
        counter = 0
        for param in self.param_list:
            start = counter
            end = param.weight.numel() + start
            all_param[start:end] = torch.reshape(param.weight, (-1,))
            counter = end 
            if param.bias is not None:
                start = counter
                end = param.bias.numel() + start
                all_param[start:end] = torch.reshape(param.bias, (-1,))
                counter = end 

        return torch.reshape(all_param, (-1, self.chunksize)).detach()
    
    @torch.no_grad()
    def param_transform(self, global_model, seed=0):
        self.compute_grad(global_model)
        param = self._partition()
        torch.manual_seed(seed)  # this is important
        coef = torch.randn((self.coef_size // self.chunksize, self.chunksize), dtype=param.dtype, device=param.device) # block_num, chunk_num
        projection = torch.sum(coef * param, dim=(1,), keepdim=True).detach() # block_num, 1
        return projection
    
    @torch.no_grad()
    def shape_recovery(self, agg_param):
        counter = 0
        for param in self.param_list:
            start = counter
            end = param.weight.numel() + start
            block_param = agg_param[start:end]
            param.weight.data += torch.reshape(block_param, param.weight.shape)
            counter = end 
            if param.bias is not None:
                start = counter
                end = param.bias.numel() + start
                block_param = agg_param[start:end]
                param.bias.data += torch.reshape(block_param, param.bias.shape)
                counter = end 

    @torch.no_grad()
    def aggregate(self, projection_list, seed_list):
        param_size = (self.total_param % self.chunksize) + self.total_param
        coef_list = []
        for (seed, proj) in zip(seed_list, projection_list):
            torch.manual_seed(seed)
            coef = torch.randn((self.coef_size // self.chunksize, self.chunksize), dtype=proj.dtype, device=proj.device)
            coef_list.append(coef)
        coef_tensor = torch.permute(torch.stack(coef_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, chunk_num --> block_num, user_num, chunk_num
        projection_tensor = torch.permute(torch.stack(projection_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, 1 --> block_num, user_num, 1
        agg_param = torch.pinverse(coef_tensor) @ projection_tensor # (block_num, chunk_num, user_num) @ (block_num, user_num, 1) --> block_num, chunk_num, 1
        self.shape_recovery(torch.reshape(agg_param.detach(), (-1,)))