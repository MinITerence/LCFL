import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from dataset_utils import check, separate_data, split_data, save_file
import h5py

load_data_path = "./workspace/FL2024/mitdb/"

# Allocate data to users
def generate_mitdb(dir_path, num_clients, num_classes, niid, balance, partition):
    
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    ##加载所有的数据----------对应的模型EcgConv1d()
    with h5py.File(os.path.join(load_data_path, 'all_ecg.hdf5'), 'r') as hdf:
         dataset_image = hdf['x'][:]
         dataset_label = hdf['y'][:]

    dataset_image = np.array(dataset_image) #(96049, 1, 130)
    dataset_label = np.array(dataset_label) #(96049, )
                
    #dataset_image.shape (70000, 1, 28, 28)  dataset_label (70000,) 
    #全部的minist数据和对应的label
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    
    random.seed(1)
    np.random.seed(1)
    num_clients = 10
    num_classes = 5
    dir_path = "./workspace/FL2024/mitdb/"

    niid = True
    balance = False
    partition = 'pat'
    generate_mitdb(dir_path, num_clients, num_classes, niid, balance, partition)