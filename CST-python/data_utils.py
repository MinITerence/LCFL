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

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)

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
