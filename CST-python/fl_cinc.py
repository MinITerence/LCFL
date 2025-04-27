from argparse import ArgumentParser
import os
import copy
import random as rn
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

#新增评估指标
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, classification_report
from torch.utils.data import RandomSampler
from base_model import load_crnn, load_crnn_f1
from ecg_fedavg import get_split_loader
from torch_ecg.models.loss import AsymmetricLoss
from fairseq_signals.data.ecg import ecg_utils
from core import get_cinc_score, get_score
from communication_func import *


class Client:
    # === Parameter setting ====
    local_epoch = 1
    batch_size = 64   
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    
    
    def __init__(self, data_idx, client_id, dataloader, local_model):

        self.trainloader = dataloader 
        self.client_id = client_id
        self.num_data = len(data_idx)

        self.train_loss = [] 
        self.train_acc = []
        self.criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")#nn.CrossEntropyLoss().to(self.device)#nn.BCEWithLogitsLoss().to(self.device)
        self.local_model = local_model

    def local_update_upload(self, global_model, cr):
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-08, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, epochs=args.scheduler_step, steps_per_epoch=len(self.trainloader),) 

        

        for e in range(self.local_epoch):
            local_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            all_labels_train = []
            all_outputs_train = []

            for idx, t in enumerate(self.trainloader):
                optimizer.zero_grad()

                source = t['net_input']['source'].to(self.device)
                padding_mask = t['net_input']['padding_mask'].to(self.device)
                labels = t['label'].to(self.device).float()

                outputs = local_model(source)
                loss = self.criterion(outputs, labels)

                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=local_model.parameters(), max_norm=args.max_norm)
                optimizer.step()
                
                train_loss += loss.item() #* labels.size(0)
                 # 如果是多分类任务，请确保输出是一个概率分布，并选择最大值作为预测类别
                _, predicted = torch.max(outputs.data, 1)  # 获取最高分对应的索引
                
                
                # _, predicted = output.max(1)
                # 注意这里假设标签是整数类型，如果标签是one-hot编码，则需要调整这部分逻辑
                y_indices = labels.argmax(dim=1) if len(labels.shape) > 1 else labels.long()
                
                
                train_total += labels.size(0)
                train_correct += predicted.eq(y_indices).sum().item()

                all_labels_train.extend(y_indices.cpu().numpy())
                all_outputs_train.extend(predicted.cpu().numpy())
                torch.cuda.empty_cache()
            scheduler.step()

            train_acc = train_correct / train_total if train_total > 0 else 0
            train_loss /= train_total

            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

        print(f'\rClient {self.client_id} finished local training at communication round {cr}. Loss: {train_loss:.4f}, Acc: {train_acc:.4f}', end='')

       
        
        return local_model.param_transform(global_model, seed=int(float(self.client_id))), int(float(self.client_id)), local_model

      
class Server:
    def __init__(self, model_global, chunksize, clients, testloader, csv_name):
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        #self.model_global = load_crnn_f1(args)
        self.clients = clients
        self.testloader = testloader
        self.criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp").to(self.device)#nn.CrossEntropyLoss().to(self.device)#nn.BCEWithLogitsLoss()
        
        self.total_client = len(clients)
        self.csv_name = csv_name
        self.model_global = model_global
        # Note that only the global test loss and accuracy are recored in this demo
        self.his_loss = [] 
        self.his_acc = []
         #新增的评估指标
        self.his_precision = []
        self.his_recall = []
        self.his_f1 = []
        self.his_cinc = []
        
        
        
    
    def _single_client(self, client_id, cr):
        """
        Perform local update to single client at one communication round.
        """
        self.model_global.to(self.device)
        local_model = copy.deepcopy(self.model_global) # use 'deepcopy' to generate model for distribution 

        projection, seed, local_model1 = self.clients[client_id].local_update_upload(local_model, cr)
        del local_model
        return projection, seed, local_model1

    def single_cr(self, client_weighted, cr, frac=1):
        """
        Perform local update to all the clients one by one。
        """
        #self.model_global = model_global.to(self.device)
        projection_list = []
        seed_list = []

        local_models = []

        
        # 使用提供的 client_ids 集合

        client_ids = {0, 1, 2, 3}
        # 将 client_ids 转换为整数列表，并根据 frac 抽样
        client_id_list = [int(float(client_id)) for client_id in client_ids if float(client_id).is_integer() and int(float(client_id)) < self.total_client]
        
        num_selected_clients = min(int(frac * len(client_id_list)), len(client_id_list))
        selected_int_ids = rn.sample(client_id_list, num_selected_clients)

        for client_id_str in selected_int_ids:
            client_id = client_id_str#int(float(client_id_str))
            projection, seed, local_model = self._single_client(client_id, cr)
            projection_list.append(projection)
            seed_list.append(seed)

            local_models.append(local_model)
        
        
        self.model_global.aggregate(projection_list, seed_list) 
        ################################运行fedavg算法###################################################################
        server_model, local_models = communication(args, load_crnn_f1(args).to(self.device), local_models, client_weighted)

        
        #self.model_global.eval()
        server_model.eval()
        total_loss = 0.0
        total_samples = 0
        best_test_acc = 0 
        # 收集所有批次的预测和标签
        all_labels, all_outputs = [], []
        preds, targets = [], []

        with torch.no_grad():
            for idx, t in enumerate(self.testloader):

                source = t['net_input']['source'].to(self.device) #print('source shape is {}'.format(source.shape))  torch.Size([64/X, 12, 2500])
                label = t['label'].to(self.device).float()        #print('label shape is {}'.format(labels.shape))    #torch.Size([64/X, 26])
                output = server_model(source)#self.model_global(source) #print('outputs shape is {}'.format(outputs.shape)) #torch.Size([64/X, 26])
                preds.extend( output.detach().cpu().numpy() )
                targets.extend(label.cpu().numpy())
                loss = self.criterion(output, label)
                total_loss += loss.item()

                #
                probs = torch.sigmoid(output)
                preds_binary = (probs > 0.5).float().cpu().numpy()
                all_outputs.extend(preds_binary)
                all_labels.extend(label.cpu().numpy())
                torch.cuda.empty_cache()

        classes, score_weights = ( ecg_utils.get_physionet_weights("./CHIL/ecg_federated/weights.csv") )
        sinus_rhythm_index = ecg_utils.get_sinus_rhythm_index(classes)
        logging_output = get_score(np.array(preds), np.array(targets), score_weights, sinus_rhythm_index)
        test_loss = float(total_loss) / len(self.testloader)
        cinc_score = get_cinc_score(logging_output)


        #
        preds_array = np.array(all_outputs)
        targets_array = np.array(all_labels)
        # 将预测结果和标签从one-hot编码转换为分类标签
        preds_labels = np.argmax(preds_array, axis=1)
        targets_labels = np.argmax(targets_array, axis=1)
        accuracy = accuracy_score(targets_labels, preds_labels)  
        precision_val = precision_score(targets_array, preds_array, average='samples')
        recall_val = recall_score(targets_array, preds_array, average='samples')
        f1_val = f1_score(targets_array, preds_array, average='samples')  
        print(f"Comm round {cr+1}, global_model cinc_score: {cinc_score:.4f}")
        print(f"Test Loss: {test_loss}")
        print(f"CINC Score: {cinc_score}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision_val}")
        print(f"Recall: {recall_val}")
        print(f"F1 Score: {f1_val}")

       
        
        self.his_acc.append(accuracy)
        self.his_loss.append(test_loss)
        self.his_precision.append(precision_val)
        self.his_recall.append(recall_val)
        self.his_f1.append(f1_val)
        self.his_cinc.append(cinc_score)

        print(f"Comm round {cr+1}, global_model val_acc: {100*accuracy:.4f}, val_loss: {test_loss:.4f}, cincsoce: {cinc_score:.4f}")
        
        df = pd.DataFrame({  # save model training process into csv file
            'global_val_acc': self.his_acc,
            'global_val_loss': self.his_loss,
            'global_val_precision': self.his_precision,
            'global_val_recall': self.his_recall,
            'global_val_f1': self.his_f1,
            'global_val_cinc': self.his_cinc
             })
        
        
        csv_ext = '.csv'
        result_dir = './CHIL/ecg_federated/Resultsss/'
        os.makedirs(result_dir, exist_ok=True)  # 确保目录存在
        df.to_csv(os.path.join(result_dir, self.csv_name + csv_ext))
        print("Global Training completed.")

       
       
        #save model that has best train accuracy
        if accuracy > best_test_acc:
            best_test_acc = accuracy 
            model_path = os.path.join(result_dir, f'{self.csv_name}-best.pth')
            torch.save(self.model_global.state_dict(), model_path)
        
        return accuracy, test_loss  


def get_cinc():
    

    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []

    for dataname in args.data_list :
        common_dir = f"{args.load_dir}/{dataname}/cinc"
        print(common_dir)
        train_loader, train_size = get_split_loader( args, os.path.join( common_dir, "train.tsv" ) , split=True)
        valid_loader, valid_size = get_split_loader( args, os.path.join( common_dir, "valid.tsv" ) , split=False)
        test_loader, test_size = get_split_loader( args, os.path.join( common_dir, "test.tsv" ) , split=False)

        train_loaders.append( train_loader )
        valid_loaders.append( valid_loader )
        test_loaders.append( test_loader )
        client_weights.append( train_size )

    client_weighted = [ c_weight / sum(client_weights) for c_weight in client_weights ]

    

    
    user_groups = dict()
    
    for client_id in range(len(args.data_list)) :
        # 计算每个用户的分配数据量
        num_data_each_client = client_weights[client_id]#len(train_loaders[client_id]) * args.batch_size
        print(f'The data number of each client {client_id} is {num_data_each_client}')

        # 直接在此处生成用户组
        all_idxs = list(range(num_data_each_client))
        
        np.random.seed(0)  # 确保每次运行时随机数生成器的行为一致
        np.random.shuffle(all_idxs)  # 随机打乱索引

        # 因为我们是为单个client_id分配数据，所以这里不需要再进行复杂的分配逻辑
        user_groups[client_id] = set(all_idxs[:num_data_each_client])

    return train_loaders, valid_loaders, test_loaders, client_weighted, user_groups 
    

parser = ArgumentParser()
parser.add_argument("--usernum", default=4, type=int, help="number of users/clients in the FL training")
parser.add_argument("--frac", default=1, type=float, help="fraction of users for participating the FL training in each round")
parser.add_argument("--chunksize", default=15, type=int, help="equivalent to the compression ratio")
parser.add_argument("--cr", default=200, type=int, help="number of communication rounds to train")

tp = lambda x:list(map(str, x.split('.')))
parser.add_argument('--data_list', type=tp, default="chapman_shaoxing.cpsc_2018.georgia.ningbo", help="data list") #default="chapman_shaoxing.cpsc_2018.georgia.ningbo",
parser.add_argument('--load_dir', type=str, default='/data/physionet.org/files/challenge-2021/1.0.3/federated_ecg_manifest', help="load dir")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")   
parser.add_argument('--num_workers', type=int, default=1, help="workers of dataloader")
parser.add_argument('--model_type', type=str, default="resnet", help="model type")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay")
parser.add_argument('--scheduler_step', type=int, default=1, help="scheduler step")
parser.add_argument('--max_norm', type=float, default=5.0, help="max norm")
parser.add_argument('--algorithm', type=str, default='fedavg', help="fedavg|fedprox|fedbn|fedadam|fedadagrad|fedyogi") 

args = parser.parse_args()

## === For reproduction =====
seed = 2024
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
## ==========================


def main():
   
    frac = args.frac
    chunksize = args.chunksize 
    comm_round = args.cr
        
    
    

    #判断是否独立同分布，仅影响到User_groups对train_dataset进行各个客户端的划分
    train_dataloaders, val_dataloader, test_dataloader, client_weighted, user_groups = get_cinc()

    
    clients = []
    #初始化客户端chapman_shaoxing.cpsc_2018.georgia.ningbo分别对应0,1,2,3
    for client_id in range(len(args.data_list)) :
        # 获取对应客户端的数据集
        Client.datasetloader = train_dataloaders[client_id]
        Client.local_model = load_crnn_f1(args)
        # 创建客户端实例
        clients.append(Client(user_groups[client_id], client_id, Client.datasetloader, Client.local_model))
    
    for client_id in range(len(args.data_list)) :
        
        server_model = load_crnn_f1(args)
        csv_name = 'cinc-chunksize-{}-client-id-{}-results'.format(chunksize, client_id)
        
        server = Server(server_model, chunksize, clients, val_dataloader[client_id], csv_name=csv_name)
        
        for cr in range(comm_round):
            server.single_cr(client_weighted, cr, frac)




if __name__ == "__main__":
    main()


    