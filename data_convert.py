import os
import torch
import h5py
from torch.utils.data import Dataset as PTDataset
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
class HDF5Dataset(PTDataset):
    def __init__(self, path, num_samples, npoin, x_left, x_right, idx_obs,batch_size):
        """ Reads data from HDF5 file as two lists of numpy arrays.
        """
        newdata = np.load(path)  # 10种4个热源1000个功率4096点的热场
        self.data_NT, self.data_NF = newdata['T'], newdata['F']
        #print("self.data_NT",self.data_NT.shape)
        self.NN_data = len(self.data_NT[0, :])  
        self.nsamples = num_samples
        self.npoin = npoin
        self.x_left = x_left
        self.x_right = x_right
        self.idx_obs = idx_obs

        self.ns = idx_obs.shape[0]

        self.data_NTS = self.data_NT.reshape(self.data_NT.shape[0]*self.data_NT.shape[1], self.data_NT.shape[2])
        self.data_NFS = self.data_NF.reshape(self.data_NF.shape[0]*self.data_NF.shape[1], self.data_NF.shape[2])


        self.datas = self.data_NTS[:self.nsamples,:]
        self.F = self.data_NFS[:self.nsamples,:]

       
        self.batch_size = batch_size

class HDF5Dataset1d(PTDataset):
    def __init__(self, pathT, path, num_samples, npoin,x_left, x_right,batch_size):
        """ 1D Reads data from HDF5 file as two lists of numpy arrays.
        """
        newdataT = np.load(pathT)
        self.data_NT = newdataT['T']
        newdata = np.load(path)
        self.data_NF = newdata['F']

        self.NN_data = len(self.data_NT[0, :])  
        self.nsamples = num_samples
        self.npoin = npoin
        self.x_left = x_left
        self.x_right = x_right
        self.batch_size = batch_size

        

        self.data_NTS = self.data_NT.reshape(self.data_NT.shape[0]*self.data_NT.shape[1], self.data_NT.shape[2])
        self.data_NFS = self.data_NF.reshape(self.data_NF.shape[0]*self.data_NF.shape[1], self.data_NF.shape[2])

        self.datas = self.data_NTS[:self.nsamples,:]
        self.F = self.data_NFS[:self.nsamples,:]

       
class HDF5Dataset2dvae(PTDataset):
    def __init__(self, pathT, path, patha,num_samples, npoin,x_left, x_right,batch_size):
        """ VAE Reads data from HDF5 file as two lists of numpy arrays.
        """
        self.data_NT = torch.load(pathT).type(torch.FloatTensor)
        self.data_NT = self.data_NT.numpy()
        self.data_NF = torch.load(path).type(torch.FloatTensor)
        self.data_NF = self.data_NF.numpy()
        self.data_NA = torch.load(patha).type(torch.FloatTensor)
        self.data_NA = self.data_NA.numpy()

        self.nsamples = num_samples
        self.datas = self.data_NT[:self.nsamples,:,:]
        self.F = self.data_NF[:self.nsamples,0,:,:]
        self.A = self.data_NA[:self.nsamples,:]
        self.NN_data = len(self.data_NT[0, :])  
        
        self.npoin = npoin
        self.x_left = x_left
        self.x_right = x_right
        self.batch_size = batch_size


class DataLoader2dvae:
    def __init__(self, dataset, num_samples, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.nsamples = num_samples
        self.num_batches = self.nsamples // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        perm = np.arange(self.nsamples)

        for i in range(self.num_batches):
            batch_indices = perm[i*self.batch_size:i*self.batch_size + self.batch_size]
            batch_F = np.array([self.dataset.F[idx,:]  for idx in batch_indices]).reshape(self.batch_size,1,64,64)
            batch_A = np.array([self.dataset.A[idx,:]  for idx in batch_indices]).reshape(self.batch_size,9)
            batch_datas = np.array([self.dataset.datas[idx,:]  for idx in batch_indices]).reshape(self.batch_size,1,64,64)
            batch_u = torch.Tensor(batch_datas)
            batch_f = torch.Tensor(batch_F)
            batch_a = torch.Tensor(batch_A)
            yield batch_a, batch_u, batch_f

class DataLoader:
    def __init__(self, dataset, num_samples, batch_size):
        """ 2D
                """
        self.dataset = dataset
        self.batch_size = batch_size
        self.nsamples = num_samples
        self.num_batches = self.nsamples // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        perm = np.arange(self.nsamples)
        for i in range(self.num_batches):
            batch_indices = perm[i*self.batch_size:i*self.batch_size + self.batch_size]
            batch_F = np.array([self.dataset.F[idx,:]  for idx in batch_indices]).reshape(self.batch_size,1,64,64)
            batch_datas = np.array([self.dataset.datas[idx,:]  for idx in batch_indices]).reshape(self.batch_size,1,64,64)
            batch_u = torch.Tensor(batch_datas)
            batch_f = torch.Tensor(batch_F)

            yield batch_f, batch_f

class DataLoader1d:
    def __init__(self, dataset, num_samples, batch_size):
        """ 1D
            """
        self.dataset = dataset
        self.batch_size = batch_size
        self.nsamples = num_samples
        self.num_batches = self.nsamples // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        perm = np.arange(self.nsamples)
        for i in range(self.num_batches):
            batch_indices = perm[i*self.batch_size:i*self.batch_size + self.batch_size]
            batch_F = np.array([self.dataset.F[idx,:]  for idx in batch_indices]).reshape(self.batch_size,101)
            batch_datas = np.array([self.dataset.datas[idx,:]  for idx in batch_indices]).reshape(self.batch_size,101)
            batch_u = torch.Tensor(batch_datas)
            batch_f = torch.Tensor(batch_F)

            yield batch_f, batch_f
class Dataset:
    def __init__(self, F, datas):
        self.F = F
        self.datas = datas

    def __len__(self):
        return self.datas.shape[1]

    def __getitem__(self, index):

        return self.F[index], self.datas[index]

class Datasetvae:
    def __init__(self, A, datas,F):
        self.A = A
        self.datas = datas
        self.F = F


    def __len__(self):
        return self.datas.shape[1]

    def __getitem__(self, index):
        return self.A[index], self.datas[index], self.F[index]