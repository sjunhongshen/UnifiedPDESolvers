import numpy as np
import os
import pickle
import gzip
import requests
import glob
import h5py
import random
import math as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pde(root, batch_size, num_workers=4, subset=None, size="base"):
    
    train_data = MixedDataset(root,subset,if_test=False, ratio=0.5, size=size)
    if subset is None:
        subset = "Burgers"

    val_data = MixedDataset(root, subset, if_test=True, ratio=1, size=size)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=False)
    return train_loader, None, test_loader


class MixedDataset(Dataset):
    def __init__(self, saved_folder, subset=None,
                 if_test=False, ratio=1,
                 size="base"):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY

        """
        self.size = size
        self.if_test = if_test
        # Define path to files
        if self.if_test:
            self.root_path = os.path.abspath(saved_folder + "/mixed_data_test/"+ subset)
            p = self.root_path +"/shape.npy"
            s = np.load(p)[:2]
            s[1] -= 1
            if ratio < 1:
                s[0] = int(s[0] * ratio)
            total = s[0] * s[1]
            self.shape_list = s
            self.embeddings = torch.from_numpy(np.load(self.root_path +"/" + self.size + "_embeddings.npy"))
            self.grid = torch.from_numpy(np.load(self.root_path +"/grid.npy"))
            stats = torch.from_numpy(np.load(self.root_path +"/stats.npy"))
            self.mean, self.std = stats[0], stats[1]
            self.mask = torch.from_numpy(np.load(self.root_path +"/mask.npy"))

        else:
            self.root_path = os.path.abspath(saved_folder + "/mixed_data_train")
            train_names = ["SW","DS","Burgers","ADV","1DCFD","2DCFD", "NS"]#,"RD","RD2D"]
            if subset is not None:
                train_names = [subset]
            self.dataset_num = len(train_names)
            self.train_name_dict = {k: train_names[k] for k in range(len(train_names))}
            self.index_dict = {}
            self.shape_list = [[], []]
            total = 0
            for i, name in enumerate(train_names):
                p = self.root_path + "/" + name +"/shape.npy"
                s = np.load(p)[:2]
                s[1] -= 1
                if name == "2DCFD" or name == "NS":
                    r = 1
                else:
                    r = ratio
                if r < 1:
                    s[0] = int(s[0] * r)
                self.shape_list[0].append(s[0])
                self.shape_list[1].append(s[1])

                total += s[0] * s[1]

        self.total = total


    def __len__(self):
        return self.total

    def get_idx(self, idx):
        ridx = 0
        for k in range(self.dataset_num):
            s = self.shape_list[0][k] * self.shape_list[1][k]
            if idx < s:
                return ridx, idx
            idx -= s 
            ridx += 1
        return ridx

    
    def __getitem__(self, idx):
        if self.if_test:
            didx = idx // self.shape_list[1]
            tidx = idx % self.shape_list[1]
            x = torch.from_numpy(np.load(self.root_path + "/" + str(didx) +".npy")[tidx])
            y = torch.from_numpy(np.load(self.root_path + "/" + str(didx) +".npy")[tidx + 1])
            x = torch.cat([x, self.grid])
            return (x, self.embeddings), (y, self.mask)
        else:
            sidx, idx = self.get_idx(idx)
            
            didx = idx // self.shape_list[1][sidx]
            tidx = idx % self.shape_list[1][sidx]
            
            dsname = self.train_name_dict[sidx]
            p = self.root_path + "/" + dsname
            x = torch.from_numpy(np.load(p+ "/" + str(didx) +".npy")[tidx])
            y = torch.from_numpy(np.load(p+ "/" + str(didx) +".npy")[tidx+1])
            g = torch.from_numpy(np.load(p+ "/grid.npy"))
            e = torch.from_numpy(np.load(p+ "/" + self.size +"_embeddings.npy"))
            m = torch.from_numpy(np.load(p+ "/mask.npy"))
            x = torch.cat([x, g])
            return (x, e), (y, m)




if __name__ == '__main__':
    load_pde("datasets", 32, dataset='Burgers', reduced_resolution=1, prev_t=1, valid_split=-1, num_workers=0, subset=None, size="base")
