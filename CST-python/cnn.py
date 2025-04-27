import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




###EcgConv1d该网络来自于FL-SPL代码中，输入大小为（X, 1, 128）后缀为h5df
class EcgConv2d(nn.Module):
    def __init__(self):
        super(EcgConv2d, self).__init__()        
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  # 32 x 16
        self.linear3 = nn.Linear(32 * 16, 128)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(128, 5)
        self.softmax4 = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 16)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.softmax4(x)
        return x            
    



class EcgConv2d_fl(nn.Module):
    def __init__(self, chunksize=1):
        super(EcgConv2d_fl, self).__init__()        
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  # 32 x 16
        self.linear3 = nn.Linear(32 * 16, 128)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(128, 5)
        self.softmax4 = nn.Softmax(dim=1)
        

        self.chunksize = chunksize
        self.param_list = [self.conv1, self.conv2, self.linear3, self.linear4]
        self.total_param, self.coef_size = self._count_param()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 16)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.softmax4(x)
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

        a = torch.stack(coef_list, dim=0)
        coef_tensor = a.permute(dims=(1, 0, 2))

        b = torch.stack(projection_list, dim=0)
        projection_tensor = b.permute(dims=(1, 0, 2))

        #coef_tensor = torch.permute(torch.stack(coef_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, chunk_num --> block_num, user_num, chunk_num
        #projection_tensor = torch.permute(torch.stack(projection_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, 1 --> block_num, user_num, 1
        agg_param = torch.pinverse(coef_tensor) @ projection_tensor # (block_num, chunk_num, user_num) @ (block_num, user_num, 1) --> block_num, chunk_num, 1
        self.shape_recovery(torch.reshape(agg_param.detach(), (-1,)))




class EcgConv3d(nn.Module):
    def __init__(self):
        super(EcgConv3d, self).__init__()        
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool1d(2)  # 32 x 16
        self.linear4 = nn.Linear(32 * 16, 128)
        self.relu4 = nn.LeakyReLU()
        self.linear5 = nn.Linear(128, 5)
        self.softmax5 = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 32 * 16)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.softmax5(x)
        return x              


class EcgConv3d_fl(nn.Module):
    def __init__(self, chunksize=1):
        super(EcgConv3d_fl, self).__init__()        
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool1d(2)  # 32 x 16
        self.linear4 = nn.Linear(32 * 16, 128)
        self.relu4 = nn.LeakyReLU()
        self.linear5 = nn.Linear(128, 5)
        self.softmax5 = nn.Softmax(dim=1)

        self.chunksize = chunksize
        self.param_list = [self.conv1, self.conv2, self.conv3, self.linear4, self.linear5]
        self.total_param, self.coef_size = self._count_param()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 32 * 16)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.softmax5(x)
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

        a = torch.stack(coef_list, dim=0)
        coef_tensor = a.permute(dims=(1, 0, 2))

        b = torch.stack(projection_list, dim=0)
        projection_tensor = b.permute(dims=(1, 0, 2))

        #coef_tensor = torch.permute(torch.stack(coef_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, chunk_num --> block_num, user_num, chunk_num
        #projection_tensor = torch.permute(torch.stack(projection_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, 1 --> block_num, user_num, 1
        agg_param = torch.pinverse(coef_tensor) @ projection_tensor # (block_num, chunk_num, user_num) @ (block_num, user_num, 1) --> block_num, chunk_num, 1
        self.shape_recovery(torch.reshape(agg_param.detach(), (-1,)))


class AlexNetEcgClassifier(nn.Module):
    '''input tensor size: (batch_size, 1, 3, 128)'''
    def __init__(self, num_classes=4, dropout_keep=0.5):
        super(AlexNetEcgClassifier, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # (N, 64, 1, 62)
            
            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # (N, 192, 1, 30)
            
            nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # (N, 256, 1, 15)
        )
        
        # 全连接层部分
        self.fc = nn.Sequential(
            nn.Linear(256 * 15, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256 * 10),
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(256 * 10, 256 * 5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_keep),
            nn.Linear(256 * 5, num_classes),
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        x = x.contiguous().view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        # 分类器
        x = self.classifier(x)
        
        return x





class AlexNetEcgClassifier_fl(nn.Module):
    '''input tensor size: (batch_size, 1, 3, 128)'''
    def __init__(self, chunksize = 1, num_classes=4, dropout_keep=0.5):
        super(AlexNetEcgClassifier_fl, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0))
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2))
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv3 = nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2))
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.linear4 = nn.Linear(256 * 15, 256 * 10)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout(0.3)
        self.bn4 = nn.BatchNorm1d(256 * 10)

        self.linear5 = nn.Linear(256 * 10, 256 * 5)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop5 = nn.Dropout(dropout_keep)

        self.linear6 =   nn.Linear(256 * 5, num_classes)

        self.chunksize = chunksize
        self.param_list = [self.conv1, self.conv2, self.conv3, self.linear4, self.linear5, self.linear6]
        self.total_param, self.coef_size = self._count_param()
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = x.contiguous().view(x.size(0), -1)
        
        # 全连接层
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        x = self.bn4(x)

        x = self.linear5(x)
        x = self.relu5(x)
        x = self.drop5(x)

        x = self.linear6(x)


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

        a = torch.stack(coef_list, dim=0)
        coef_tensor = a.permute(dims=(1, 0, 2))

        b = torch.stack(projection_list, dim=0)
        projection_tensor = b.permute(dims=(1, 0, 2))

        
        agg_param = torch.pinverse(coef_tensor) @ projection_tensor # (block_num, chunk_num, user_num) @ (block_num, user_num, 1) --> block_num, chunk_num, 1
        self.shape_recovery(torch.reshape(agg_param.detach(), (-1,)))

