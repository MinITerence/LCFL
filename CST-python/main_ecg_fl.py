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


parser = ArgumentParser()
parser.add_argument("--usernum", default=20, type=int, help="number of users/clients in the FL training")
parser.add_argument("--frac", default=1, type=float, help="fraction of users for participating the FL training in each round")
parser.add_argument("--chunksize", default=1, type=int, help="equivalent to the compression ratio")
parser.add_argument("--cr", default=100, type=int, help="number of communication rounds to train")
parser.add_argument("--iid", default=False, action='store_false', help="iid data distribution")
parser.add_argument("--alpha", default=0.1, type=int, help="Dirichlet hyper-parameter")
args = parser.parse_args()

## === For reproduction =====
seed = 2024
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
## ==========================


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的输出，形状为 (batch_size, num_classes)
        :param targets: 真实标签，形状为 (batch_size)
        :return: 计算的 focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Client:
    # === Parameter setting ====
    local_epoch = 1
    batch_size = 256   
    lr = 0.001   
    lr_decay = 0.996
    # =========================
    dataset = None
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    def __init__(self, data_idx, client_id):

        self.trainloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)  
        self.client_id = client_id
        self.num_data = len(data_idx)

        self.train_loss = [] 
        self.train_acc = []

    def local_update_upload(self, global_model, cr):
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        lr = self.lr * (self.lr_decay ** cr) # decay the learning rate
        #lr = self.lr
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
        local_model.train()


        

        for e in range(self.local_epoch):
            train_loss = 0
            train_correct = 0
            train_total = 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                local_model.zero_grad()
                output = local_model(images)
                loss = self.criterion(output, labels.long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            avg_train_loss = train_loss / len(self.trainloader)
            avg_train_acc = train_correct / train_total
            self.train_loss.append(avg_train_loss)
            self.train_acc.append(avg_train_acc)

        print(f'\rClient {self.client_id} finished local training at communication round {cr}. Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}', end='')

       
        print('\rClient %d finished local training at %d communication round' % (self.client_id, cr), end='')
        return local_model.param_transform(global_model, seed=self.client_id), self.client_id



      
class Server:
    def __init__(self, chunksize, clients, testset, csv_name):
        self.model_global = EcgConv2d_fl(chunksize)#AlexNetEcgClassifier_fl(chunksize, num_classes=4, dropout_keep=0.5)#EcgConv3d_fl(chunksize) # CNN to train
        self.clients = clients
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)  #batch_size=100
        self.criterion = nn.CrossEntropyLoss()#FocalLoss(alpha=0.25, gamma=2)
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        self.total_client = len(clients)
        self.csv_name = csv_name
        #self.log_name = log_name
        
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
        local_model = copy.deepcopy(self.model_global) # use 'deepcopy' to generate model for distribution 
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
        selected_id = rn.sample(list(range(self.total_client)), int(frac * self.total_client))
        for client_id in selected_id:
            projection, seed = self._single_client(client_id, cr)
            projection_list.append(projection)
            seed_list.append(seed)
        
        self.model_global.aggregate(projection_list, seed_list)
        
        
        # evaulate the global model
        # save the training info in current communication round
        self.model_global.eval()
        test_loss = 0
        correct = 0
        total = 0

        best_test_acc = 0 

        # 收集所有批次的预测和标签
        all_labels = []
        all_outputs = []
        

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model_global(images)
                loss = self.criterion(outputs, labels.long())

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy().argmax(axis=1))


        # 计算整体的评估指标
        acc = correct/total
        test_loss /= total
        

        #新增评估指标
        from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, confusion_matrix

       


        precision_train = precision_score(all_labels, all_outputs, average='macro', zero_division=0)
        recall_train = recall_score(all_labels, all_outputs, average='macro', zero_division=0)
        f1_train = f1_score(all_labels, all_outputs, average='macro', zero_division=0)
        
        self.his_acc.append(acc)
        self.his_loss.append(test_loss)
        self.his_precision.append(precision_train)
        self.his_recall.append(recall_train)
        self.his_f1.append(f1_train)
        
        # write the log info
        # with open(self.log_path, 'wb') as file:
        #     pickle.dump([self.his_acc, self.his_loss, self.his_precision, self.his_recall, self.his_f1], file)

        df = pd.DataFrame({  # save model training process into csv file
            'global_train_acc': self.his_acc,
            'global_train_loss': self.his_loss,
            'global_train_precision': self.his_precision,
            'global_train_recall': self.his_recall,
            'global_train_f1': self.his_f1
             })
        
        csv_ext = '.csv'
        #csv_name = 'Global-conv2-train-eval-Client-20'
        
        df.to_csv(os.path.join('./LCFL/results/CST/Results/MITBIH/alpha/', self.csv_name + csv_ext))
        
        print("\nComm round %d, global train acc: %.6f, global train loss: %.6f" % (cr, 100*acc, test_loss))

        #save model that has best train accuracy
        if acc > best_test_acc:
            best_test_acc = acc
            torch.save(self.model_global.state_dict(), os.path.join('./LCFL/results/CST/Results/MITBIH/alpha/', '{}-best.pth'.format(self.csv_name)))    
        
        return acc, test_loss







def main():
    num_users = args.usernum
    frac = args.frac
    chunksize = args.chunksize # 1 ~ 31
    comm_round = args.cr
    iid = args.iid  # iid: True; non-iid: False
    alpha = args.alpha
    
    # 创建 Focal Loss 损失函数
    #criterion = FocalLoss(alpha=0.25, gamma=2)

    
    clients = []

    #判断是否独立同分布，仅影响到User_groups对train_dataset进行各个客户端的划分
    train_dataset, test_dataset, user_groups = get_dataset_mitdb(iid, num_users, alpha)

    Client.dataset = train_dataset
    for user_idx in range(num_users):
        clients.append(Client(user_groups[user_idx], user_idx))
    
    print(clients)

    
    csv_name = 'nlrav-conv2-chunksize-{}-alpha-{}'.format(chunksize, alpha)
    
    
    
    server = Server(chunksize, clients, test_dataset, csv_name=csv_name)
    
    for cr in range(comm_round):
        server.single_cr(cr, frac)




if __name__ == "__main__":
    main()





















      














    