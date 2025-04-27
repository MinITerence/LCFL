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
    
class ECGClassifier(nn.Module):
    def __init__(self, num_classes, i_to_avf_input_dim, v1_to_v6_input_dim, d_hid=64, n_filters=32, en_att=False, att_name='', d_head=6):
        super(ECGClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.param_list = []
        
        
        
        # Conv Encoder module
        self.i_to_avf_conv = Conv1dEncoder(input_dim=i_to_avf_input_dim, n_filters=n_filters, dropout=self.dropout_p)
        self.v1_to_v6_conv = Conv1dEncoder(input_dim=v1_to_v6_input_dim, n_filters=n_filters, dropout=self.dropout_p)

        # Add individual modules from Conv1dEncoders to param_list
        for module in [self.i_to_avf_conv, self.v1_to_v6_conv]:
            for name, submodule in module.named_modules():
                if isinstance(submodule, (nn.Conv1d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                    self.param_list.append(submodule)
                    
        # RNN module
        self.i_to_avf_rnn = nn.GRU(input_size=n_filters*4, hidden_size=d_hid, num_layers=1, batch_first=True, dropout=self.dropout_p, bidirectional=False)
        self.v1_to_v6_rnn = nn.GRU(input_size=n_filters*4, hidden_size=d_hid, num_layers=1, batch_first=True, dropout=self.dropout_p, bidirectional=False)
        self.param_list.extend([self.i_to_avf_rnn, self.v1_to_v6_rnn])

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(d_hid=d_hid, d_head=d_head)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            self.param_list.extend([self.fuse_att, self.classifier])
        else:
            self.i_to_avf_proj = nn.Linear(d_hid, d_hid//2)
            self.v1_to_v6_proj = nn.Linear(d_hid, d_hid//2)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            self.param_list.extend([self.i_to_avf_proj, self.v1_to_v6_proj, self.classifier])

        #self.total_param, self.coef_size = self._count_param()

        

        self.init_weight()
        

    def init_weight(self):
        for m in self.modules():  # Correctly iterate over all submodules
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
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
    

def get_dataset_ptbdb():
    

    i_to_avf_data_path = './LCFL/results/CST/Results/ptb-xl/fedmultimodal-output/cl_feature/I_to_AVF/ptb-xl/train.pkl'
    v1_to_v6_data_path = './LCFL/results/CST/Results/ptb-xl/fedmultimodal-output/cl_feature/V1_to_V6/ptb-xl/train.pkl'

    # 打开并读取pkl文件
    with open(i_to_avf_data_path, 'rb') as f1:
        i_avf_dict = pickle.load(f1)  # 将pkl文件中的数据加载到i_avf_dict字典中
    with open(v1_to_v6_data_path, 'rb') as f2:
        v1_v6_dict = pickle.load(f2)  # 将pkl文件中的数据加载到i_avf_dict字典中
    
    dataset_train = MMDatasetGenerator(i_avf_dict, v1_v6_dict, default_feat_shape_a=np.array([1000, 6]), default_feat_shape_b=np.array([1000, 6]), 
                                       data_len = len(i_avf_dict), dataset='ptb-xl-cl')
    train_dataloader = DataLoader(dataset_train, batch_size=64, num_workers=0, shuffle=False, collate_fn=collate_mm_fn_padd)
        
    i_to_avf_data_path_test = './LCFL/results/CST/Results/ptb-xl/fedmultimodal-output/cl_feature/I_to_AVF/ptb-xl/dev.pkl'
    v1_to_v6_data_path_test = './LCFL/results/CST/Results/ptb-xl/fedmultimodal-output/cl_feature/V1_to_V6/ptb-xl/dev.pkl'
    
    # 打开并读取pkl文件
    with open(i_to_avf_data_path_test, 'rb') as f3:
        i_avf_dict_dev = pickle.load(f3)  # 将pkl文件中的数据加载到i_avf_dict字典中
    with open(v1_to_v6_data_path_test, 'rb') as f4:
        v1_v6_dict_dev = pickle.load(f4)  # 将pkl文件中的数据加载到i_avf_dict字典中

    dataset_test = MMDatasetGenerator(i_avf_dict_dev, v1_v6_dict_dev, default_feat_shape_a=np.array([1000, 6]), default_feat_shape_b=np.array([1000, 6]), 
                                       data_len = len(i_avf_dict_dev), dataset='ptb-xl-cl')
        
    test_dataloader = DataLoader(dataset_test, batch_size=64, num_workers=0, shuffle=False, collate_fn=collate_mm_fn_padd)
        
    
    

    return train_dataloader, test_dataloader


parser = ArgumentParser()
parser.add_argument("--usernum", default=1, type=int, help="number of users/clients in the FL training")
parser.add_argument("--frac", default=1, type=float, help="fraction of users for participating the FL training in each round")
parser.add_argument("--chunksize", default=1, type=int, help="equivalent to the compression ratio")
parser.add_argument("--cr", default=200, type=int, help="number of communication rounds to train")
parser.add_argument("--iid", default=False, action='store_true', help="iid data distribution")

parser.add_argument('--att', type=bool, default=True, help='self attention applied or not')
parser.add_argument("--en_att",dest='att',default=False, help="enable self-attention")  #action='store_true',
parser.add_argument('--hid_size',type=int, default=64, help='RNN hidden size dim')  #64    128
parser.add_argument('--att_name',type=str, default='multihead', help='attention name') #multihead    fuse_base
parser.add_argument("--dataset",  type=str, default="ptb-xl-cl", help='data set name')

args = parser.parse_args()

## === For reproduction =====
seed = 2024
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
    
    
    train_dataloader, test_dataloader = get_dataset_ptbdb()
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    model = ECGClassifier(num_classes=5, i_to_avf_input_dim=feature_len_dict['i_to_avf'],  v1_to_v6_input_dim=feature_len_dict['v1_to_v6'],  
        en_att=args.att,  d_hid=args.hid_size,  att_name=args.att_name).to(device)
    

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005 , momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)

    local_epoch = 200
    best_val_acc = 0
    his_train_acc, his_train_loss, his_train_precision, his_train_recall, his_train_f1 = [], [], [], [], []
    his_val_acc, his_val_loss, his_val_precision, his_val_recall, his_val_f1 = [], [], [], [], []

    csv_ext = '.csv'
    csv_name = 'Attention-ptbxl-cl-results'
    


    for cr in range(local_epoch):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        all_labels_train = []
        all_outputs_train = []

        for batch_idx, (x_a, x_b, len_a, len_b, ys) in enumerate(train_dataloader):
            optimizer.zero_grad()

            x_a, x_b, ys = x_a.to(device), x_b.to(device), ys.to(device)
            len_a, len_b = len_a.to(device), len_b.to(device)

            outputs, _ = model(x_a.float(), x_b.float(), len_a, len_b)

            # Convert one-hot encoded labels to class indices
            y_indices = torch.argmax(ys, dim=1)  # Get the index of max value along dim=1 [ty-reference](1)

            loss = criterion(outputs, ys.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * ys.size(0)
            _, predicted = outputs.max(1)
            total += ys.size(0)
            correct += predicted.eq(y_indices).sum().item()

            all_labels_train.extend(y_indices.cpu().numpy())
            all_outputs_train.extend(predicted.cpu().numpy())

        scheduler.step()

        train_acc = correct / total if total > 0 else 0
        train_loss /= total
        precision_train = precision_score(all_labels_train, all_outputs_train, average='macro', zero_division=0)
        recall_train = recall_score(all_labels_train, all_outputs_train, average='macro', zero_division=0)
        f1_train = f1_score(all_labels_train, all_outputs_train, average='macro', zero_division=0)

        his_train_acc.append(train_acc)
        his_train_loss.append(train_loss)
        his_train_precision.append(precision_train)
        his_train_recall.append(recall_train)
        his_train_f1.append(f1_train)

        print(f"\nTraining epoch {cr+1}, train acc: {100*train_acc:.6f}, train loss: {train_loss:.6f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_labels_val = []
        all_outputs_val = []

        with torch.no_grad():
            for batch_idx, (x_a, x_b, len_a, len_b, ys) in enumerate(test_dataloader):
                x_a, x_b, ys = x_a.to(device), x_b.to(device), ys.to(device)
                len_a, len_b = len_a.to(device), len_b.to(device)

                outputs, _ = model(x_a.float(), x_b.float(), len_a, len_b)
                y_indices = torch.argmax(ys, dim=1)

                loss = criterion(outputs, ys.float())
                val_loss += loss.item() * ys.size(0)

                _, predicted = outputs.max(1)
                val_total += ys.size(0)
                val_correct += predicted.eq(y_indices).sum().item()

                all_labels_val.extend(y_indices.cpu().numpy())
                all_outputs_val.extend(predicted.cpu().numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss /= val_total
        precision_val = precision_score(all_labels_val, all_outputs_val, average='macro', zero_division=0)
        recall_val = recall_score(all_labels_val, all_outputs_val, average='macro', zero_division=0)
        f1_val = f1_score(all_labels_val, all_outputs_val, average='macro', zero_division=0)

        his_val_acc.append(val_acc)
        his_val_loss.append(val_loss)
        his_val_precision.append(precision_val)
        his_val_recall.append(recall_val)
        his_val_f1.append(f1_val)

        print(f"Comm round {cr+1}, val acc: {100*val_acc:.6f}, val loss: {val_loss:.6f}")


        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join('./FedMultimodal/fed_multimodal/result-cl/', f'{csv_name}-best.pth'))

        # 日志记录
        df = pd.DataFrame({
            'train_acc': his_train_acc,
            'train_loss': his_train_loss,
            'train_precision': his_train_precision,
            'train_recall': his_train_recall,
            'train_f1': his_train_f1,
            'val_acc': his_val_acc,
            'val_loss': his_val_loss,
            'val_precision': his_val_precision,
            'val_recall': his_val_recall,
            'val_f1': his_val_f1
        })
        df.to_csv(os.path.join('./FedMultimodal/fed_multimodal/result-cl/', csv_name + csv_ext))

        print("Training and validation completed.")
                




if __name__ == "__main__":
    main()


    