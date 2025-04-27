import torch
from ecgdata_utils import ECG
from cnn import EcgConv2d, EcgConv3d
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt



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




# 创建 Focal Loss 损失函数
criterion = FocalLoss(alpha=0.25, gamma=2)





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
epoch =100
lr = 0.001  
batch_size = 64
train_dataset = ECG(mode='train')
val_dataset = ECG(mode='val')
#test_dataset = ECG(mode='test')
train_loader = DataLoader(train_dataset, batch_size)
val_loader = DataLoader(val_dataset, batch_size)
#test_loader = DataLoader(test_dataset, batch_size)
x_train, y_train = next(iter(train_loader))
total_batch = len(train_loader)


model_ext = '.pth'
csv_ext = '.csv'

model_name = 'CL-conv2' 
csv_name = 'CL-conv2-acc-loss'
csv_accs_name = 'CL--conv2-val_best_acc'



def train(nrun, model):
    criterion = nn.CrossEntropyLoss()
      
    #optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = list()
    train_accs = list()

    val_losses = list()
    val_accs = list()

    # test_losses = list()
    # test_accs = list()

    best_val_acc = 0  # best test accuracy 

    for e in range(epoch):
        print("Epoch {} - ".format(e+1), end='')

        # train
        train_loss = 0.0
        correct, total = 0, 0
        for _, batch in enumerate(train_loader):
            x, label = batch  # get feature and label from a batch
            x, label = x.to(device), label.to(device)  # send to device
            optimizer.zero_grad()  # init all grads to zero
            output = model(x)  # forward propagation
            loss = criterion(output, label)  # calculate loss
            loss.backward()  # backward propagation
            optimizer.step()  # weight update

            train_loss += loss.item()
            correct += torch.sum(output.argmax(dim=1) == label).item()
            total += len(label)
        #train_losses.append(train_loss / len(train_loader))
        train_loss /= total
        train_losses.append(train_loss)
        train_accs.append(correct / total)
        print("loss: {:.4f}, acc: {:.2f}%".format(train_losses[-1], train_accs[-1]*100), end=' / ')


        
        # val
        with torch.no_grad():
            val_loss = 0.0
            correct, total = 0, 0
            for _, batch in enumerate(val_loader):
                x, label = batch
                x, label = x.to(device), label.to(device)
                output = model(x)
                loss = criterion(output, label)
                
                val_loss += loss.item()
                correct += torch.sum(output.argmax(dim=1) == label).item()
                total += len(label)
            
            val_loss /= total
            val_losses.append(val_loss)
            #val_losses.append(val_loss / len(val_loader))
            val_accs.append(correct / total)
        print("val_loss: {:.4f}, val_acc: {:.2f}%".format(val_losses[-1], val_accs[-1]*100))

        # save model that has best validation accuracy
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            torch.save(model.state_dict(), os.path.join('./LCFL/results/CST/Results/model-cl/', '-'.join([model_name, 'best']) + model_ext))
    
        # # save model for each 10 epochs
        # if (e + 1) % 50 == 0:
        #     torch.save(model.state_dict(), os.path.join('./model-lr/', '_'.join([model_name, str(nrun), str(e+1)]) + model_ext))
    
    return train_losses, train_accs, val_losses, val_accs


#Repeat for run times

best_val_accs = list()
run =1
for i in range(run):
    print('Run', i+1)
    
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ecgnet = EcgConv2d()  
    ecgnet.load_state_dict(torch.load('./LCFL/results/CST/Results/model-cl/init_weight_conv2.pth'))
    print('Successfully load init_weight.pth')


    train_losses, train_accs, val_losses, val_accs = train(i, ecgnet.to(device))  # train

    best_val_accs.append(max(val_accs))  # get best test accuracy
    best_val_acc_epoch = np.array(val_accs).argmax() + 1
    print('Best val accuracy {:.2f}% in epoch {}.'.format(best_val_accs[-1]*100, best_val_acc_epoch))
    print('-' * 100)

    df = pd.DataFrame({  # save model training process into csv file
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    df.to_csv(os.path.join('./LCFL/results/CST/Results/csv-cl/', '_'.join([csv_name, str(i+1)]) + csv_ext))

df = pd.DataFrame({'best_val_acc': best_val_accs})  # save best test accuracy of each run




df.to_csv(os.path.join('./LCFL/results/CST/Results/', csv_accs_name + csv_ext))


for i, a in enumerate(best_val_accs):
    print('Run {}: {:.2f}%'.format(i+1, a*100))


