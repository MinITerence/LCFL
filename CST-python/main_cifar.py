from argparse import ArgumentParser
import os
import copy
import random as rn
import numpy as np
import torch
import torch.nn as nn
from vgg import Net
from data_utils import DatasetSplit, get_dataset_cifar
from torch.utils.data import DataLoader
import pickle


# python main_cifar.py --usernum 100 --frac 0.32 --chunksize 10 --datadir ~/datasets --cr 200

parser = ArgumentParser()

parser.add_argument("--usernum", default=100, type=int, help="number of users/clients in the FL training")
parser.add_argument("--frac", default=0.32, type=float, help="fraction of users for participating the FL training in each round")
parser.add_argument("--chunksize", default=10, type=int, help="equivalent to the compression ratio")
parser.add_argument("--datadir", default='./', type=str, help="path of the dataset")
parser.add_argument("--cr", default=200, type=int, help="number of communication rounds to train")
parser.add_argument("--noniid", action='store_false', help="non iid data distribution")


args = parser.parse_args()



## === For reproduction =====
seed = 0
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
## ==========================


os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Specify the GPU to be used


class Client:
    # === Parameter setting ====
    local_epoch = 1
    batch_size = 64
    lr = 0.05
    lr_decay = 0.996
    # =========================
    dataset = None
    device = 'cuda'
    criterion = nn.CrossEntropyLoss()

    def __init__(self, data_idx, client_id):
        self.trainloader = DataLoader(DatasetSplit(self.dataset, data_idx), 
                                      batch_size=self.batch_size, shuffle=True)
        self.client_id = client_id
        self.num_data = len(data_idx)

    def local_update_upload(self, global_model, cr):
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        lr = self.lr * (self.lr_decay ** cr) # decay the learning rate
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
        local_model.train()
        for e in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                local_model.zero_grad()
                output = local_model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

        print('\rClient %d finished local training at %d communication round' % (self.client_id, cr), end='')
        return local_model.param_transform(global_model, seed=self.client_id), self.client_id

        
class Server:
    def __init__(self, chunksize, clients, testset, log_name):
        self.model_global = Net(chunksize) # DNN to train
        self.clients = clients
        self.testloader = DataLoader(testset, batch_size=100, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.device = 'cuda'
        self.total_client = len(clients)
        
        # Note that only the global test loss and accuracy are recored in this demo
        self.his_loss = [] 
        self.his_acc = []
        
        # path to save the info
        if not os.path.exists('./log/'):
            os.mkdir('./log')
        self.log_path = './log/' + log_name
    
    def _single_client(self, client_id, cr):
        """
        Perform local update to single client at one communication round.
        """
        self.model_global.to(self.device)
        local_model = copy.deepcopy(self.model_global) # use 'deepcopy' to generate model for distribution 
        projection, seed = self.clients[client_id].local_update_upload(local_model, cr)
        del local_model
        return projection, seed

    def single_cr(self, cr, frac=0.5):
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
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model_global(images)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100.*correct/total
        test_loss /= total
        self.his_acc.append(acc)
        self.his_loss.append(test_loss)
        
        # write the log info
        with open(self.log_path, 'wb') as file:
            pickle.dump([self.his_acc, self.his_loss], file)
        
        print("\nComm round %d, test acc: %.6f, test loss: %.6f" % (cr, acc, test_loss))
        return acc, test_loss


def main():
    num_users = args.usernum
    frac = args.frac
    chunksize = args.chunksize # 1 ~ 31
    comm_round = args.cr
    data_dir = args.datadir
    iid = args.noniid  # iid: True; non-iid: False
    clients = []
    train_dataset, test_dataset, user_groups = get_dataset_cifar(data_dir, iid, num_users, 50000//num_users)
    Client.dataset = train_dataset
    for user_idx in range(num_users):
        clients.append(Client(user_groups[user_idx], user_idx))
    
    log_name = 'CIFAR10_usernum={}_frac={}_chunksize={}_cr={}_iid={}'.format(num_users, frac, chunksize, comm_round, iid)
    server = Server(chunksize, clients, test_dataset, log_name=log_name)
    for cr in range(comm_round):
        server.single_cr(cr, frac)
        

if __name__ == "__main__":
    main()


    