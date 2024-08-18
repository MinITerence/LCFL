""" Models for MNIST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetCNN(nn.Module):
    def __init__(self, chunksize=10):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.chunksize = chunksize
        self.param_list = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.total_param, self.coef_size = self._count_param()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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