from argparse import ArgumentParser
import os
import copy
import random as rn
import numpy as np
import torch
import torch.nn as nn
from cnn import EcgConv2d_fl, EcgConv3d_fl, AlexNetEcgClassifier_fl
from ecgdata_utils import DatasetSplit, get_dataset_mitdb
from torch.utils.data import DataLoader, Subset
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from dataload_manager import DataloadManager, MMDatasetGenerator, collate_mm_fn_padd
#新增评估指标
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, confusion_matrix



class Conv1dEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        n_filters: int,
        dropout: float=0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
        ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

   
class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64,
        d_head: int=4
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
        self,
        x: Tensor,
        val_a=None,
        val_b=None,
        a_len=None
    ):
        att = self.att_pool(self.att_fc1(x))
        # att = self.att_fc2(att).squeeze(-1)
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid)
        return x


class ECGClassifier_fl(nn.Module):
    def __init__(self, 
        num_classes: int,           # Number of classes 
        i_to_avf_input_dim: int,    # 6 lead ecg
        v1_to_v6_input_dim: int,    # v1-v6 ecg
        chunksize: int,
        d_hid: int=64,              # Hidden Layer size
        n_filters: int=32,          # number of filters
        en_att: bool=False,         # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6         # Head dim
        ):
        
        super(ECGClassifier_fl, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.chunksize = chunksize
        
        # Conv Encoder module
        self.i_to_avf_conv = Conv1dEncoder(input_dim=i_to_avf_input_dim, n_filters=n_filters, dropout=self.dropout_p)
        self.v1_to_v6_conv = Conv1dEncoder(input_dim=v1_to_v6_input_dim, n_filters=n_filters, dropout=self.dropout_p)

        # RNN module
        self.i_to_avf_rnn = nn.GRU(input_size=n_filters*4, hidden_size=d_hid, num_layers=1, batch_first=True, dropout=self.dropout_p, bidirectional=False)
        self.v1_to_v6_rnn = nn.GRU(input_size=n_filters*4, hidden_size=d_hid, num_layers=1, batch_first=True, dropout=self.dropout_p, bidirectional=False)

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(d_hid=d_hid, d_head=d_head)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            
        else:
            self.i_to_avf_proj = nn.Linear(d_hid, d_hid//2)
            self.v1_to_v6_proj = nn.Linear(d_hid, d_hid//2)
            
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        self.init_weight()
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.param_list = [self.i_to_avf_conv, self.v1_to_v6_conv, self.i_to_avf_rnn, self.v1_to_v6_rnn, self.classifier]
        self.total_param, self.coef_size = self._count_param()
        

    def forward(self, x_i_to_avf, x_v1_to_v6, l_a, l_b):
        # 1. Conv forward
        x_i_to_avf = self.i_to_avf_conv(x_i_to_avf)
        x_v1_to_v6 = self.v1_to_v6_conv(x_v1_to_v6)

        l_a = l_a // 8
        l_b = l_b // 8
        
        # 2. Rnn forward
        x_i_to_avf, _ = self.i_to_avf_rnn(x_i_to_avf)
        x_v1_to_v6, _ = self.v1_to_v6_rnn(x_v1_to_v6)

        # 3. Attention
        if self.en_att:
            # get attention output
            x_mm = torch.cat((x_i_to_avf, x_v1_to_v6), dim=1)
            x_mm = self.fuse_att(
                x_mm, 
                val_a=l_a, 
                val_b=l_b, 
                a_len=x_i_to_avf.shape[1]
            )
        else:
            # 4. Average pooling
            x_i_to_avf = torch.mean(x_i_to_avf, axis=1)
            x_v1_to_v6 = torch.mean(x_v1_to_v6, axis=1)
            # 6. MM embedding and predict
            x_mm = torch.cat((x_i_to_avf, x_v1_to_v6), dim=1)
        preds = self.classifier(x_mm)
        return preds, x_mm
    
    @torch.no_grad()
    def _count_param(self):
        total_param = sum(p.numel() for p in self.parameters())
        coef_size = ((total_param + self.chunksize - 1) // self.chunksize) * self.chunksize
        return total_param, coef_size
    
    @torch.no_grad()
    def _partition(self):
        all_param = torch.empty(self.coef_size, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        counter = 0

        for param in self.parameters():
            numel = param.numel()
            start = counter
            end = start + numel
            all_param[start:end] = param.view(-1)
            counter = end
    
        # # 确保我们已经使用了所有分配的空间
        # if counter != self.coef_size:
        #     raise ValueError(f"Not all elements in all_param were used. Used {counter}, expected {self.coef_size}")

        return all_param.view(-1, self.chunksize).detach()

    @torch.no_grad()
    def compute_grad(self, global_model):
        for local_param, global_param in zip(self.parameters(), global_model.parameters()):
            if local_param.requires_grad:
                local_param.grad = local_param.data - global_param.data

    @torch.no_grad()
    def param_transform(self, global_model, seed=0):
        self.compute_grad(global_model)
        param = self._partition()  # Assume this returns a tensor of shape [chunk_num, chunk_size]
        torch.manual_seed(seed)
        coef = torch.randn((self.coef_size // self.chunksize, self.chunksize), dtype=param.dtype, device=param.device) # block_num, chunk_num  # Ensure coef has the same shape as param
        #projection = (coef * param).sum(dim=-1, keepdim=True).detach()  # Dot product over the last dimension
        projection = torch.sum(coef * param, dim=(1,), keepdim=True).detach() # block_num, 1
        return projection
    
    
    @torch.no_grad()
    def shape_recovery(self, agg_param):
        counter = 0
        for module in self.param_list:
            # 遍历当前模块及其所有子模块
            for name, param in module.named_parameters():
                start = counter
                end = start + param.numel()
                block_param = agg_param[start:end]
                reshaped_block_param = torch.reshape(block_param, param.shape)
                param.data.copy_(reshaped_block_param)  # 使用 copy_ 方法来更新参数值
                counter = end
                #print(f"Recovered parameter '{name}' with numel={param.numel()}, new counter={counter}")


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

   
class Client:
    # === Parameter setting ====
    local_epoch = 1
    batch_size = 64   
    lr = 0.1#0.005   
    lr_decay = 0.996
    # =========================
    
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    
    
    def __init__(self, data_idx, client_id, dataloader):

        self.trainloader = dataloader 
        self.client_id = client_id
        self.num_data = len(data_idx)

        self.train_loss = [] 
        self.train_acc = []
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def local_update_upload(self, global_model, cr):
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)

        
        lr = self.lr
       
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
        
        

        for e in range(self.local_epoch):
            local_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            all_labels_train = []
            all_outputs_train = []

            for batch_idx, batch_data in enumerate(self.trainloader):
                optimizer.zero_grad()

                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                
                output, _ = local_model(x_a.float(), x_b.float(), l_a, l_b)

                # Convert one-hot encoded labels to class indices
                y_indices = torch.argmax(y, dim=1)  # Get the index of max value along dim=1 [ty-reference](1)

                loss = self.criterion(output, y.float())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * y.size(0)
                _, predicted = output.max(1)
                train_total += y.size(0)
                train_correct += predicted.eq(y_indices).sum().item()

                all_labels_train.extend(y_indices.cpu().numpy())
                all_outputs_train.extend(predicted.cpu().numpy())

            #scheduler.step()

            train_acc = train_correct / train_total if train_total > 0 else 0
            train_loss /= train_total

            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

        print(f'\rClient {self.client_id} finished local training at communication round {cr}. Loss: {train_loss:.4f}, Acc: {train_acc:.4f}', end='')

       
        
        return local_model.param_transform(global_model, seed=int(float(self.client_id))), int(float(self.client_id))

      
class Server:
    def __init__(self, chunksize, clients, testloader, csv_name):
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        self.model_global = ECGClassifier_fl(num_classes=5, i_to_avf_input_dim=feature_len_dict['i_to_avf'],  v1_to_v6_input_dim=feature_len_dict['v1_to_v6'],  
        en_att=args.att,  d_hid=args.hid_size,  att_name=args.att_name, chunksize=chunksize).to(self.device)
        self.clients = clients
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.total_client = len(clients)
        self.csv_name = csv_name
        
        # Note that only the global test loss and accuracy are recored in this demo
        self.his_loss = [] 
        self.his_acc = []
         #新增的评估指标
        self.his_precision = []
        self.his_recall = []
        self.his_f1 = []
        
        
    
    def _single_client(self, client_id, cr):
        """
        Perform local update to single client at one communication round.
        """
        self.model_global.to(self.device)
        local_model = copy.deepcopy(self.model_global) 

        projection, seed = self.clients[client_id].local_update_upload(local_model, cr)
        del local_model
        return projection, seed

    def single_cr(self, cr, frac=1):
        """
        Perform local update to all the clients one by one。
        """
        self.model_global.to(self.device)
        projection_list = []
        seed_list = []

        # 使用提供的 client_ids 集合
        client_ids = {'0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', 
                      '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0', '18.0', 
                      '19.0', '20.0', '21.0', '22.0', '23.0', '24.0', '25.0', '26.0', '27.0', 
                      '28.0', '29.0', '31.0', '34.0', '37.0'}

        # 将 client_ids 转换为整数列表，并根据 frac 抽样
        client_id_list = [int(float(client_id)) for client_id in client_ids if float(client_id).is_integer() and int(float(client_id)) < self.total_client]
        
        num_selected_clients = min(int(frac * len(client_id_list)), len(client_id_list))
        selected_int_ids = rn.sample(client_id_list, num_selected_clients)
        
        # 将选中的整数 ID 转换回原始的浮点数字符串格式
        selected_ids = {f"{client_id}.0" for client_id in selected_int_ids}
        #print("Selected client IDs:", selected_ids)

        for client_id_str in selected_ids:
            client_id = int(float(client_id_str))
            projection, seed = self._single_client(client_id, cr)
            projection_list.append(projection)
            seed_list.append(seed)
        
        
        self.model_global.aggregate(projection_list, seed_list) 


        
        self.model_global.eval()
        test_loss = 0
        correct = 0
        total = 0

        best_test_acc = 0 

        # 收集所有批次的预测和标签
        all_labels = []
        all_outputs = []
        

        with torch.no_grad():
            for batch_idx, (x_a, x_b, len_a, len_b, ys) in enumerate(self.testloader):
                x_a, x_b, ys = x_a.to(self.device), x_b.to(self.device), ys.to(self.device)
                labels = ys
                len_a, len_b = len_a.to(self.device), len_b.to(self.device)

                outputs, _ = self.model_global(x_a.float(), x_b.float(), len_a, len_b)

                # Convert one-hot encoded labels to class indices
                y_indices = torch.argmax(ys, dim=1)  # Get the index of max value along dim=1 [ty-reference](1)

                loss = self.criterion(outputs, ys.float())
                test_loss += loss.item() * ys.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(y_indices).sum().item()

                all_labels.extend(y_indices.cpu().numpy())
                all_outputs.extend(predicted.cpu().numpy())
                
        acc = correct / total if total > 0 else 0
        test_loss /= total
        precision_val = precision_score(all_labels, all_outputs, average='macro', zero_division=0)
        recall_val = recall_score(all_labels, all_outputs, average='macro', zero_division=0)
        f1_val = f1_score(all_labels, all_outputs, average='macro', zero_division=0)

        self.his_acc.append(acc)
        self.his_loss.append(test_loss)
        self.his_precision.append(precision_val)
        self.his_recall.append(recall_val)
        self.his_f1.append(f1_val)

        print(f"Comm round {cr+1}, global_model val_acc: {100*acc:.6f}, val_loss: {test_loss:.6f}")
        
        df = pd.DataFrame({  # save model training process into csv file
            'global_val_acc': self.his_acc,
            'global_val_loss': self.his_loss,
            'global_val_precision': self.his_precision,
            'global_val_recall': self.his_recall,
            'global_val_f1': self.his_f1
             })
        
        csv_ext = '.csv'
        
        df.to_csv(os.path.join('./LCFL/results/CST/Results/ptb-xl/fedmultimodal-output/result-chunksize/', self.csv_name + csv_ext))
        print("Global Training completed.")
       
        #save model that has best train accuracy
        if acc > best_test_acc:
            best_test_acc = acc
            torch.save(self.model_global.state_dict(), os.path.join('./LCFL/results/CST/Results/ptb-xl/fedmultimodal-output/result-chunksize/', '{}-best.pth'.format(self.csv_name)))    
        
        return acc, test_loss



def get_ptbxl():
    
    dm = DataloadManager(args)
    dm.get_client_ids()

    # number of clients, removing dev and test
    client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
    print(client_ids)
    num_of_clients = len(client_ids)
    print("Number of clients is {}".format(num_of_clients))

    # set dataloaders
    dataloader_dict = dict()
    user_groups = dict()
    
    for idx, client_id in enumerate(client_ids):
        
        i_avf_dict, v1_v6_dict = dm.load_ecg_feat(client_id=client_id)
        shuffle = False if client_id in ['dev', 'test'] else True
        #client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
        client_sim_dict = None
        dataloader_dict[client_id] = dm.set_dataloader(
            i_avf_dict, 
            v1_v6_dict, 
            shuffle=shuffle,
            client_sim_dict=client_sim_dict,
            default_feat_shape_a=np.array([1000, feature_len_dict["i_to_avf"]]),
            default_feat_shape_b=np.array([1000, feature_len_dict["v1_to_v6"]])
        )
        # 计算每个用户的分配数据量
        num_data_each_client = len(dataloader_dict[client_id]) *64  #batch_size
        print(f'The data number of each client {client_id} is {num_data_each_client}')

        # 直接在此处生成用户组
        all_idxs = list(range(num_data_each_client))
        np.random.seed(0)  # 确保每次运行时随机数生成器的行为一致
        np.random.shuffle(all_idxs)  # 随机打乱索引

        # 因为我们是为单个client_id分配数据，所以这里不需要再进行复杂的分配逻辑
        user_groups[client_id] = set(all_idxs[:num_data_each_client])


    # number of clients, removing dev and test
    client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
    print(client_ids)
    num_of_clients = len(client_ids)
    print("Number of clients is {}".format(num_of_clients))

    # 检查结果
    for client_id, group in user_groups.items():
        print(f"Client {client_id} has been assigned {len(group)} data points.")    

    # 加载测试数据集
    data_root='/data/ptb-xl/fedmultimodal-output/feature'
    i_to_avf_data_path_test = os.path.join(data_root, 'I_to_AVF', 'ptb-xl', 'dev.pkl')
    v1_to_v6_data_path_test = os.path.join(data_root, 'V1_to_V6', 'ptb-xl', 'dev.pkl')

    with open(i_to_avf_data_path_test, 'rb') as f3:
        i_avf_dict_dev = pickle.load(f3)
    with open(v1_to_v6_data_path_test, 'rb') as f4:
        v1_v6_dict_dev = pickle.load(f4)

    dataset_test = MMDatasetGenerator(
        i_avf_dict_dev, v1_v6_dict_dev,
        default_feat_shape_a=np.array([1000, 6]),
        default_feat_shape_b=np.array([1000, 6]),
        data_len=len(i_avf_dict_dev),
        dataset='ptb-xl'
    )
    
    val_dataloader = DataLoader(dataset_test, batch_size=64, num_workers=0, shuffle=False, collate_fn=collate_mm_fn_padd)


    return dataloader_dict, val_dataloader, user_groups
    


parser = ArgumentParser()
parser.add_argument("--usernum", default=33, type=int, help="number of users/clients in the FL training")
parser.add_argument("--frac", default=1, type=float, help="fraction of users for participating the FL training in each round")
parser.add_argument("--chunksize", default=30, type=int, help="equivalent to the compression ratio")
parser.add_argument("--cr", default=200, type=int, help="number of communication rounds to train")
parser.add_argument("--iid", default=False, action='store_true', help="iid data distribution")

parser.add_argument('--att', type=bool, default=False, help='self attention applied or not')
parser.add_argument("--en_att",dest='att',default=True, help="enable self-attention")  #action='store_true',
parser.add_argument('--hid_size',type=int, default=64, help='RNN hidden size dim')  #64    128
parser.add_argument('--att_name',type=str, default='multihead', help='attention name') #multihead    fuse_base
parser.add_argument("--dataset",  type=str, default="ptb-xl", help='data set name')

args = parser.parse_args()

## === For reproduction =====
seed = 2025
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
## ==========================


# define feature len mapping
feature_len_dict = {'i_to_avf': 6, 'v1_to_v6':  6}

# define num of class dict
num_class_dict = {'ptb-xl':  5}




def main():
    num_users = args.usernum
    frac = args.frac
    chunksize = args.chunksize 
    comm_round = args.cr
    iid = args.iid  
    
    
    
    clients = []

    #判断是否独立同分布，仅影响到User_groups对train_dataset进行各个客户端的划分
    train_dataloaders, val_dataloader, user_groups = get_ptbxl()

    

    # # 定义客户端ID集合
    client_ids = {'0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0', '18.0', '19.0', '20.0', '21.0', '22.0', '23.0', '24.0', '25.0', '26.0', '27.0', '28.0', '29.0', '31.0', '34.0', '37.0'}

   

    # 初始化客户端
    for client_id in client_ids:
        int_client_id = client_id  # 将浮点数转换为整数
        
        
        # 获取对应客户端的数据集
        Client.datasetloader = train_dataloaders[client_id]
        
        # 创建客户端实例
        
        clients.append(Client(user_groups[client_id], int_client_id, Client.datasetloader))

    

    csv_name = 'Attention-lr0.1-crossentropy-chunksize-{}-results'.format(chunksize)
    
    
    
    server = Server(chunksize, clients, val_dataloader, csv_name=csv_name)
    
    for cr in range(comm_round):
        server.single_cr(cr, frac)




if __name__ == "__main__":
    main()




























