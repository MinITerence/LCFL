import json
import glob
import torch
import pickle
import random
import pdb, os
import torchaudio
import numpy as np
import os.path as osp
# import pickle5 as pickle

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset

def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

def collate_mm_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None: max_b_len = max(map(lambda x: x[1].shape[0], batch))

    # pad according to max_len
    x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        
        len_a.append(torch.tensor(batch[idx][2]))
        len_b.append(torch.tensor(batch[idx][3]))

        ys.append(batch[idx][-1])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, x_b, len_a, len_b, ys

def collate_unimodal_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    
    # pad according to max_len
    x_a, len_a, ys = list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        len_a.append(torch.tensor(batch[idx][1]))
        ys.append(batch[idx][-1])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    len_a = torch.stack(len_a, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, len_a, ys


class MMDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA, 
        modalityB, 
        default_feat_shape_a,
        default_feat_shape_b,
        data_len: int, 
        simulate_feat=None,
        dataset: str=''
    ):
        
        self.data_len = data_len
        
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.simulate_feat = simulate_feat
        
        self.default_feat_shape_a = default_feat_shape_a
        self.default_feat_shape_b = default_feat_shape_b
        self.dataset = dataset
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        
        # modality A, if missing replace with 0s, and mask
        if data_a is not None: 
            if len(data_a.shape) == 3: data_a = data_a[0]
            data_a = torch.tensor(data_a)
            len_a = len(data_a)
        else: 
            data_a = torch.tensor(np.zeros(self.default_feat_shape_a))
            len_a = 0

        # modality B, if missing replace with 0s
        if data_b is not None:
            if len(data_b.shape) == 3: data_b = data_b[0]
            data_b = torch.tensor(data_b)
            len_b = len(data_b)
        else: 
            data_b = torch.tensor(np.zeros(self.default_feat_shape_b))
            len_b = 0
        return data_a, data_b, len_a, len_b, label



class UniModalDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA,
        data_len: int, 
        simulate_feat=None,
        dataset: str=''
    ):
        self.dataset = dataset
        self.data_len = data_len
        self.modalityA = modalityA
        self.simulate_feat = simulate_feat
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        data_a = torch.tensor(data_a)
        len_a = len(data_a)
        return data_a, len_a, label




class DataloadManager():
    def __init__(
        self, 
        args: dict
    ):
        self.args = args
        self.label_dist_dict = dict()
        # Initialize video feature paths
        
        if self.args.dataset in ['ptb-xl']:     
            self.i_to_avf_path = Path('./results/CST/Results/ptb-xl/fedmultimodal-output').joinpath(
                'feature', 
                'I_to_AVF', 
                args.dataset
            )
            self.v1_to_v6_path = Path('./results/CST/Results/ptb-xl/fedmultimodal-output').joinpath(
                'feature', 
                'V1_to_V6', 
                args.dataset
            )
        if self.args.dataset in ['ptb-xl-cl']:     
            self.i_to_avf_path = Path('./results/CST/Results/ptb-xl/fedmultimodal-output').joinpath(
                'feature-cl', 
                'I_to_AVF', 
                args.dataset
            )
            self.v1_to_v6_path = Path('./results/CST/Results/ptb-xl/fedmultimodal-output').joinpath(
                'feature-cl', 
                'V1_to_V6', 
                args.dataset
            )
    
    
    def get_client_ids(
            self, 
            fold_idx: int=1
        ):
        """
        Load client ids.
        :param fold_idx: fold index
        :return: None
        """
        
        if self.args.dataset == "ptb-xl":
            data_path = self.v1_to_v6_path
        elif self.args.dataset == "ptb-xl-cl":
            data_path = self.v1_to_v6_path
        self.client_ids = [id.split('.pkl')[0] for id in os.listdir(str(data_path))]
        self.client_ids.sort()
        
    
    
    def load_ecg_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load ecg feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        i_to_avf_data_path = self.i_to_avf_path.joinpath(f'{client_id}.pkl')
        v1_to_v6_data_path = self.v1_to_v6_path.joinpath(f'{client_id}.pkl')
        with open(str(i_to_avf_data_path), "rb") as f:  i_to_avf_data_dict = pickle.load(f)
        with open(str(v1_to_v6_data_path), "rb") as f:  v1_to_v6_data_dict = pickle.load(f)
        return i_to_avf_data_dict, v1_to_v6_data_dict
    
    def load_cl_ecg_feat(
            self, 
            client_id: str
        ) -> dict:
        """
        Load ecg feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        i_to_avf_data_path = self.i_to_avf_path.joinpath(f'{client_id}.pkl')
        v1_to_v6_data_path = self.v1_to_v6_path.joinpath(f'{client_id}.pkl')
        with open(str(i_to_avf_data_path), "rb") as f:  i_to_avf_data_dict = pickle.load(f)
        with open(str(v1_to_v6_data_path), "rb") as f:  v1_to_v6_data_dict = pickle.load(f)
        return i_to_avf_data_dict, v1_to_v6_data_dict

    def get_client_sim_dict(
            self, 
            client_id
        ):
        """
        Set dataloader for training/dev/test.
        :param client_id: client_id
        :return: dataloader: torch dataloader
        """
        if self.sim_data:
            return self.sim_data[client_id]
        return None

    def get_label_dist(
        self, 
        data_dict,
        client_id: str
    ):
        """
        Set dataloader for training/dev/test.
        :param data_dict: data dictionary
        :return: data_dis_dict
        """
        label_list = list()
        for idx in range(len(data_dict)):
            label_list.append(data_dict[idx][-2])
        self.label_dist_dict[client_id] = Counter(label_list)
        
    def set_dataloader(
            self, 
            data_a: dict,
            data_b: dict,
            default_feat_shape_a: np.array=np.array([0, 0]),
            default_feat_shape_b: np.array=np.array([0, 0]),
            client_sim_dict: dict=None,
            shuffle: bool=False
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param data_b: modality B data
        :param default_feat_shape_a: default input shape for modality A, fill 0 in missing modality case
        :param default_feat_shape_b: default input shape for modality B, fill 0 in missing modality case
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :return: dataloader: torch dataloader
        """
        # modify data based on simulation
        labeled_data_idx, unlabeled_data_idx = list(), list()
        if client_sim_dict is not None:
            for idx in range(len(client_sim_dict)):
                # read simulate feature
                sim_data = client_sim_dict[idx][-1]
                # read modality A
                if sim_data[0] == 1: data_a[idx][-1] = None
                # read modality B
                if sim_data[1] == 1: data_b[idx][-1] = None
                # label noise
                data_a[idx][-2] = sim_data[2]
                # missing label
                if sim_data[-1] == 0: labeled_data_idx.append(idx)
                else: unlabeled_data_idx.append(idx)
            
            # return None when both modalities are missing
            if sim_data[0] == 1 and sim_data[1] == 1:
                return None
            
            labeled_data_a, unlabeled_data_a = list(), list()
            labeled_data_b, unlabeled_data_b = list(), list()
            if len(unlabeled_data_idx) > 0:
                for idx in labeled_data_idx:
                    labeled_data_a.append(data_a[idx])
                    labeled_data_b.append(data_b[idx])
                for idx in unlabeled_data_idx:
                    unlabeled_data_a.append(data_a[idx])
                    unlabeled_data_b.append(data_b[idx])
                data_a = labeled_data_a
                data_b = labeled_data_b

        if len(data_a) == 0: return None
        data_ab = MMDatasetGenerator(
            data_a, 
            data_b,
            default_feat_shape_a,
            default_feat_shape_b,
            len(data_a),
            self.args.dataset
        )
        if shuffle:
            # we use args input batch size for train, typically set as 16 in FL setup
            dataloader = DataLoader(
                data_ab, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        else:
            # we use a larger batch size for validation and testing
            dataloader = DataLoader(
                data_ab, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        return dataloader

    def set_unimodal_dataloader(
            self, 
            data_a: dict,
            client_sim_dict: dict=None,
            shuffle: bool=False
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :return: dataloader: torch dataloader
        """
        # modify data based on simulation
        if client_sim_dict is not None:
            for idx in range(len(client_sim_dict)):
                # read simulate feature
                sim_data = client_sim_dict[idx][-1]
                # pdb.set_trace()
                # read modality A
                if sim_data[0] == 1: data_a[idx][-1] = None
                # read modality B
                if sim_data[1] == 1: data_b[idx][-1] = None
                # label noise
                data_a[idx][-2] = sim_data[2]
            
            # return None when both modalities are missing
            if sim_data[0] == 1 and sim_data[1] == 1:
                return None
                
        data = UniModalDatasetGenerator(
            data_a, 
            len(data_a),
            self.args.dataset
        )
        if shuffle:
            # we use args input batch size for train, typically set as 16 in FL setup
            dataloader = DataLoader(
                data, 
                batch_size=int(self.args.batch_size), 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_unimodal_fn_padd
            )
        else:
            # we use a larger batch size for validation and testing
            dataloader = DataLoader(
                data, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_unimodal_fn_padd
            )
        return dataloader
    
    def load_sim_dict(
        self, 
        fold_idx: int=1,
        ext: str="json"
    ):
        """
        Load simulation dictionary.
        :param fold_idx: fold index
        :return: None
        """
        if self.setting_str == '': 
            self.sim_data = None
            return
        
        if self.args.dataset in ["ptb-xl"]:
            data_path = Path(self.args.data_dir).joinpath(
                'simulation_feature',
                self.args.dataset,
                f'{self.setting_str}.{ext}'
            )
        if ext == "pkl":
            with open(str(data_path), "rb") as f: 
                self.sim_data = pickle.load(f)
        else:
            with open(str(data_path), "r") as f: 
                self.sim_data = json.load(f)
    
    def get_simulation_setting(self, alpha=None):
        """
        Load get simulation setting string.
        :param alpha: alpha in manual split
        :return: None
        """
        self.setting_str = ''
        # 1. missing modality
        if self.args.missing_modality == True:
            self.setting_str += 'mm'+str(self.args.missing_modailty_rate).replace('.', '')
        # 2. label nosiy
        if self.args.label_nosiy == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ln'+str(self.args.label_nosiy_level).replace('.', '')
        # 3. missing labels
        if self.args.missing_label == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ml'+str(self.args.missing_label_rate).replace('.', '')
        # 4. alpha for manual split
        if len(self.setting_str) != 0:
            if alpha is not None:
                alpha_str = str(self.args.alpha).replace('.', '')
                self.setting_str += f'_alpha{alpha_str}'