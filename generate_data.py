import numpy as np
import os
import h5py
import math as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data import Dataset


def simple_normalize(train_data=None, val_data=None, mean_x=None, std_x=None):
    if train_data is not None:
        res = 1
        mean_x = train_data.x.reshape(-1, 4,128*res,128*res).mean((0,2,3),keepdim=True)
        std_x = train_data.x.reshape(-1, 4,128*res,128*res).std((0,2,3),keepdim=True)
        std_x[std_x==0]=1

    mean_x = mean_x.unsqueeze(0)
    std_x = std_x.unsqueeze(0)

    if train_data is not None:
        train_data.x= (train_data.x-mean_x)/std_x
    if val_data is not None:
        val_data.x=(val_data.x-mean_x)/std_x

    return train_data, val_data, mean_x[0], std_x[0]


def load_pde(root, train_names, prev_t=1,  num_workers=4, size="base"):
    torch.cuda.empty_cache()
    
    for k, dataset in enumerate(train_names):

        print(k, dataset)

        if dataset == 'Burgers':

            filename = '1D_Burgers_Sols_Nu0.001.hdf5'
            reduced_resolution = 8
            reduced_resolution_t = 5 

        elif dataset == 'Burgers2':

            filename = '1D_Burgers_Sols_Nu1.0.hdf5'
            reduced_resolution = 8
            reduced_resolution_t = 5  

        elif dataset == '1DCFD':
            filename = '1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5'
            reduced_resolution = 8
            reduced_resolution_t = 5 

        elif dataset == 'ADV':
            filename = '1D_Advection_Sols_beta0.4.hdf5'
            reduced_resolution = 8
            reduced_resolution_t = 5 

        elif dataset == 'DS':
            filename = '1D_diff-sorp_NA_NA.h5'
            reduced_resolution = 8
            reduced_resolution_t = 5

        elif dataset == 'RD':
            filename = 'ReacDiff_Nu0.5_Rho1.0.hdf5'
            reduced_resolution = 8
            reduced_resolution_t = 5 

        elif dataset == 'SW':
            filename = '2D_rdb_NA_NA.h5'
            reduced_resolution = 1
            reduced_resolution_t = 1 

        elif dataset == 'Darcy':
            filename = '2D_DarcyFlow_beta0.1_Train.hdf5'
            reduced_resolution = 1
            reduced_resolution_t = 1

        elif dataset == 'Darcy2':
            filename = '2D_DarcyFlow_beta0.01_Train.hdf5'
            reduced_resolution = 1
            reduced_resolution_t = 1

        elif dataset == 'RD2D':
            filename = '2D_diff-react_NA_NA.h5'
            reduced_resolution = 1
            reduced_resolution_t = 1 

        elif dataset == '2DCFD':
            filename = '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
            reduced_resolution = 1
            reduced_resolution_t = 1 

        elif dataset == "NS":
            filename = 'ns_incom_inhom_2d_512-0.h5'
            reduced_resolution = 4
            reduced_resolution_t = 1

        print(filename)
        train_data = UNetDatasetSingle(filename,
                                            saved_folder=root,
                                            reduced_resolution=reduced_resolution,
                                            reduced_resolution_t=reduced_resolution_t,
                                            reduced_batch=1, prev_t=prev_t, size=size)
        
        val_data = None
        
        train_data, val_data, m_, s_ = simple_normalize(train_data, val_data)
        m = m_.cpu().numpy()
        s = s_.cpu().numpy()

        if not os.path.exists("datasets/mixed_data_test/" + dataset):    
            os.makedirs("datasets/mixed_data_test/" + dataset)
        np.save("datasets/mixed_data_test/" + dataset + "/stats.npy", np.array([m, s]))

        dirname = "datasets/mixed_data_train/"+ dataset
        if not os.path.exists(dirname):    
            os.makedirs(dirname)
        train_data.x = train_data.x.cpu().numpy()
        train_data.grid = train_data.grid.cpu().numpy()
        train_data.mask = train_data.mask.cpu().numpy()

        np.save(dirname + "/mask.npy",train_data.mask[0])
        np.save(dirname + "/grid.npy",train_data.grid[0])
        np.save(dirname + "/shape.npy",train_data.x.shape)

        for z in range(len(train_data.x)):
            np.save(dirname + '/' + str(z) + '.npy', train_data.x[z])

        del train_data
        
        val_data = UNetDatasetSingle(filename,
                                      saved_folder=root,
                                      reduced_resolution=reduced_resolution,
                                      reduced_resolution_t=reduced_resolution_t,
                                      reduced_batch=1, prev_t=prev_t,
                                      if_test=True, size=size)

        _, val_data, _, _ = simple_normalize(None, val_data, m_,s_)

        dirname = "datasets/mixed_data_test/" + dataset 
        val_data.x = val_data.x.cpu().numpy()
        val_data.grid = val_data.grid.cpu().numpy()
        val_data.mask = val_data.mask.cpu().numpy()

        np.save(dirname + "/grid.npy",val_data.grid[0])
        np.save(dirname + "/mask.npy",val_data.mask[0])
        np.save(dirname + "/shape.npy",val_data.x.shape)

        for z in range(len(val_data.x)):
            np.save(dirname + '/' + str(z) + '.npy', val_data.x[z])

        del val_data


MAXLEN = 28 
class UNetDatasetSingle(Dataset):
    def __init__(self, filename,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1, prev_t=1, size="base"):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY

        """
        self.prev_t = prev_t
        self.if_test = if_test
        # Define path to files
        root_path = os.path.abspath(saved_folder + "/" + filename)

        if filename[-2:] == "h5":
            flen = -3
            is_h5 = True
        else:
            flen = -5
            is_h5 = False
        self.flen = flen
        text_path = os.path.abspath(saved_folder + "/mixed_data_train/" + dataset + ("/" + size + "_embeddings.npy"))
        try:
            self.embeddings = torch.from_numpy(np.load(text_path))
        except:
            pass

        if is_h5:

            with h5py.File(root_path, 'r') as h5_file:
                data_list = sorted(h5_file.keys())
                test_idx = int(len(data_list) * (1-test_ratio))
                if if_test:
                    self.data_list = np.array(data_list[test_idx:])
                else:
                    self.data_list = np.array(data_list[:test_idx]) 
            
                # data dim = [t, x1, ..., xd, v]
                data = np.array([h5_file[self.data_list[k]]["data"] for k in range(len(self.data_list))], dtype='f')
                data = torch.tensor(data, dtype=torch.float)

                if len(data.shape) == 5:
                    data = data.permute(0,2,3,1,4) 
                
                    # convert to [x1, ..., xd, t, v]
                    self.data = data[:, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution_t, :]

                    x = np.array(h5_file[self.data_list[0]]["grid"]["x"], dtype='f')
                    y = np.array(h5_file[self.data_list[0]]["grid"]["y"], dtype='f')
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y, indexing='ij')
                    self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                else:
                    data = data.permute(0,2,1,3)
                    self.data = data[:,::reduced_resolution, ::reduced_resolution_t, :]
                    grid = np.array(h5_file[self.data_list[0]]["grid"]["x"], dtype='f')
                    self.grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)


        else:
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        self.data = torch.tensor(self.data)
                        
                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        
                    if len(idx_cfd)==4:  # 2D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)

                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,1] = _data   # batch, x, t, ch


                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y)
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                
                        
                    if len(idx_cfd)==5:  # 3D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z)
                        self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                                    ::reduced_resolution,\
                                                                    ::reduced_resolution]
                    

                else:  # scalar equations

                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...

                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data = _data[:, :, :, None]  # batch, x, t, ch
                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    

                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch
                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y)
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        test_idx = int(self.data.shape[0] * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:]     
        
        if not torch.is_tensor(self.data):
            self.data = torch.from_numpy(self.data)

        if len(self.data[..., 0,:].shape) == 3:
            self.data = self.data.permute(0,2,1,3)
            self.x = self.data
            self.grid = self.grid.transpose(-1,-2)

            ori_shape = self.x.shape
            self.x = self.x.permute(0,1,3,2)
            padzero = 0

            if self.x.shape[2] < 4:
                padzero = self.x.shape[2]
                self.x = torch.cat([self.x] + (4-self.x.shape[2]) * [torch.zeros_like(self.x)], 2)

            self.x = self.x.reshape(ori_shape[0],ori_shape[1],4,1,ori_shape[2])
            self.x = self.x.broadcast_to(ori_shape[0],ori_shape[1],4,ori_shape[2], ori_shape[2])
            self.mask = torch.zeros_like(self.x[:,0,...])

            if padzero:
                self.mask[:,:padzero,:padzero,:]=1
            else:
                self.mask[:,:,:1,:]=1
            self.grid = torch.concat([self.grid, torch.zeros_like(self.grid)],0)
            self.grid = self.grid.unsqueeze(-2).broadcast_to(self.grid.shape[0],self.grid.shape[1], self.grid.shape[1])
            print("1d->2d", self.grid.shape, self.x.shape, self.mask.shape)

        else:
            self.data = self.data.permute(0,3,1,2,4) #bs,t,x,y,ch
            self.x = self.data
            self.grid = self.grid.permute(2,0,1)
            
            padzero=0

            if self.x.shape[-1]<4:
                padzero=self.x.shape[-1]
                self.x=torch.cat([self.x] + (4-self.x.shape[-1]) * [torch.zeros_like(self.x[...,:1])], -1)

            self.x = self.x.permute(0,1,4,2,3).reshape(self.x.shape[0],self.x.shape[1],-1,self.x.shape[2],self.x.shape[3])
            self.mask = torch.ones_like(self.x[:,0,...])

            if padzero:
                self.mask[:,padzero:,...]=0

            print("2d->2d", self.x.shape, self.grid.shape, self.mask.shape)
        
        self.grid = torch.stack([self.grid] * self.x.shape[0])
        try:
            self.embeddings = torch.cat([self.embeddings] * len(self.x))
        except:
            pass
        self.mean, self.std = 0, 0

        print("Reduced res:", reduced_resolution, " Markov t:", self.prev_t, " Input shape:", self.x.shape, " grid shape:", self.grid.shape)

        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.embeddings[idx]), (self.y[idx], self.mask[idx])



if __name__ == '__main__':
    train_names = ["SW","DS","Burgers","ADV","1DCFD","2DCFD","RD","RD2D","NS"]
    load_pde("datasets", train_names, size="base")
