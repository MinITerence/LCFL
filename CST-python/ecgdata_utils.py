from typing import Any, Callable, Optional, Tuple
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def get_dataset_mnist(data_dir, iid, num_users):
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    if iid:
        user_groups = mnist_iid(train_dataset, num_users)
    else:
        user_groups = mnist_noniid(train_dataset, num_users)

    return train_dataset, test_dataset, user_groups



def cifar_iid(dataset, num_users, num_data_each_client):
    np.random.seed(0)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_data_each_client,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def cifar_noniid(dataset, num_users,
                 num_shards=200, num_imgs=250, shards_per_user=2, num_data_each_client=1):
    np.random.seed(0)
    """
    num_shards: number of shards
    num_imgs: number of images per shard
    shards_per_user: number of shards per user
    """
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(
            np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        if len(dict_users[i]) > num_data_each_client:
            dict_users[i] = dict_users[i][0:num_data_each_client]
    return dict_users


def cifar_noniid_dir(dataset, num_users, alpha):
    np.random.seed(0)
    """
    dataset: training set of CIFAR
    """
    dict_users = {}
    num_classes = 10
    labels = np.array(dataset.targets)
    num_items = int(len(dataset)/num_users)
    
    base_prob = np.random.dirichlet(np.repeat(alpha, num_classes))

    idx = np.stack([np.roll(np.arange(num_classes), i) for i in range(num_users)])
    mat_prob = base_prob[idx] / num_users * num_classes   # the sum of each column equals 1, each row equals num_classes/num_users
    
    for u in range(num_users):
        dict_users[u] = []

    for cls_idx in range(num_classes):
        idx_by_cls = np.where(labels == cls_idx)[0]
        len_by_cls = len(idx_by_cls)
        
        np.random.shuffle(idx_by_cls)
        current_idx = 0
        for u in range(num_users):
            num_by_cls_by_user = int(mat_prob[u, cls_idx] * len_by_cls)
            end_idx = current_idx + num_by_cls_by_user
            dict_users[u] = dict_users[u] + idx_by_cls[current_idx:end_idx].tolist()
            current_idx = end_idx

    return dict_users


def get_dataset_cifar(data_dir, iid, num_users, num_data_each_client, alpha=0.6):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=transform_test)

    if iid:
        user_groups = cifar_iid(train_dataset, num_users, num_data_each_client)
    else:
        user_groups = cifar_noniid_dir(train_dataset, num_users, alpha)

    return train_dataset, test_dataset, user_groups




#定义ECG数据集的dataset和non-iid和iid的mitdb_iid()和mitdb_noniid_dir()
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np
import pandas as pd
import h5py
#import matplotlib.pyplot as plt




# def mitdb_iid(dataset, num_users, num_data_each_client):
#     np.random.seed(0)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_data_each_client, replace=False))  
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def mitdb_iid(dataset, num_users, num_data_each_client):
    np.random.seed(0)
    # 初始化返回的字典和所有可能的索引列表
    dict_users = {i: set() for i in range(num_users)}
    all_idxs = list(range(len(dataset)))
    # 随机打乱索引
    np.random.shuffle(all_idxs)
    # 分配数据给每个用户
    start_idx = 0
    for i in range(num_users):
        end_idx = start_idx + num_data_each_client
        dict_users[i] = set(all_idxs[start_idx:end_idx])
        start_idx = end_idx

    return dict_users  #dict_users 是一个字典，它表示每个用户分配到的数据索引集合



#使用Dirichlet分布来控制每个用户的数据分布，从而模拟现实世界中数据分布的不均匀性。 alpha:Dirichlet分布的参数，控制数据分布的集中程度。alpha 越小，数据分布越不均匀
def mitdb_noniid_dir(dataset, num_users, alpha):
    np.random.seed(0)
    """
    dataset: training set of CIFAR
    """
    dict_users = {}
    num_classes = 5

    ##加载训练数据集----------对应的模型EcgConv1d()
    load_data_path = "./results/CST/Data/mitdb/"
    with h5py.File(os.path.join(load_data_path, 'train_ecg.hdf5'), 'r') as hdf:
         dataset_label = hdf['y_train'][:]

    #dataset_image = np.array(dataset_image) #(96049, 1, 130)
    labels = np.array(dataset_label) #(13245, )


    #labels = np.array(dataset.targets)
    num_items = int(len(dataset)/num_users)
    
    base_prob = np.random.dirichlet(np.repeat(alpha, num_classes))

    idx = np.stack([np.roll(np.arange(num_classes), i) for i in range(num_users)])
    mat_prob = base_prob[idx] / num_users * num_classes   # the sum of each column equals 1, each row equals num_classes/num_users
    
    for u in range(num_users):
        dict_users[u] = []

    for cls_idx in range(num_classes):
        idx_by_cls = np.where(labels == cls_idx)[0]
        len_by_cls = len(idx_by_cls)
        
        np.random.shuffle(idx_by_cls)
        current_idx = 0
        for u in range(num_users):
            num_by_cls_by_user = int(mat_prob[u, cls_idx] * len_by_cls)
            end_idx = current_idx + num_by_cls_by_user
            dict_users[u] = dict_users[u] + idx_by_cls[current_idx:end_idx].tolist()
            current_idx = end_idx

    return dict_users


root_path = './results/CST/Data/'
data_dir = 'mitdb'

#NVLRA平衡数据集分类
# train_name = 'train_ecg.hdf5'
# val_name = 'test_ecg.hdf5'
# test_name = 'test_ecg.hdf5'
# all_name = "all_ecg.hdf5"

train_name = 'train.hdf5'
val_name = 'val.hdf5'
test_name = 'test.hdf5'
all_name = "all.hdf5"
#NSVF---aami不平衡数据集分类
# train_name = 'train_aami_alexnet.hdf5'
# val_name = 'val_aami_alexnet.hdf5'
# test_name = 'test_aami_alexnet.hdf5'
# all_name = "all_aami_alexnet.hdf5"

#按照与NVLRA相同预处理方式的NSVF---aami不平衡数据集分类
# train_name = 'train_aami.hdf5'
# val_name = 'val_aami.hdf5'
# test_name = 'test_aami.hdf5'
# all_name = "all_aami.hdf5"


class ECG(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            with h5py.File(os.path.join(root_path, data_dir, train_name), 'r') as hdf:
                self.x = hdf['x_train'][:]
                self.y = hdf['y_train'][:]
        elif mode == 'val':
            with h5py.File(os.path.join(root_path, data_dir, val_name), 'r') as hdf:
                self.x = hdf['x_val'][:]#hdf['x_val'][:]  #self.x = hdf['x_test'][:] 
                self.y = hdf['y_val'][:]#hdf['y_val'][:]  #self.y = hdf['y_test'][:]
        elif mode == 'test':
            with h5py.File(os.path.join(root_path, data_dir, test_name), 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
        elif mode == 'all':
            with h5py.File(os.path.join(root_path, data_dir, all_name), 'r') as hdf:
                self.x = hdf['x'][:]
                self.y = hdf['y'][:]
        else:
            raise ValueError('Argument of mode should be train, test, or all.')
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])
    

def get_dataset_mitdb(iid, num_users, alpha=0.6):
    

    train_dataset = ECG(mode='train')
    test_dataset = ECG(mode='test')
    num_data_each_client = len(train_dataset) // num_users
    print('the data number of each_client is {}'.format(num_data_each_client))

    
    if iid:
        user_groups = mitdb_iid(train_dataset, num_users, num_data_each_client)
    else:
        user_groups = mitdb_noniid_dir(train_dataset, num_users, alpha)

    return train_dataset, test_dataset, user_groups



def get_dataset_mitdb_backup(data_dir, iid, num_users, num_data_each_client, alpha=0.6):
    

    train_dataset = ECG(mode='train')
    test_dataset = ECG(mode='test')
    num_data_each_client = len(train_dataset) / num_users

    
    if iid:
        user_groups = mitdb_iid(train_dataset, num_users, num_data_each_client)
    else:
        user_groups = mitdb_noniid_dir(train_dataset, num_users, alpha)

    return train_dataset, test_dataset, user_groups


