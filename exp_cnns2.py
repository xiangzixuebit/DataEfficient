#!/usr/bin/env python3

import os
import argparse
import csv
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from model import UNetV2,UNetV2_Encoder,UNetV2_Decoder,MLPModel
from data_convert import HDF5Dataset
from trainings2 import train_and_test,validate
import numpy as np

class Experiment:

    def __init__(self, rparams, data_path, npoin, x_left, x_right, idx_obs,batch_size,load_path,load_pathDs2, load_pathEs2,load_pathF1,load_pathD1,save_lossfile):

        self.device = "cpu"
        if torch.cuda.is_available():
            print('We have a GPU!')
            self.device = "cuda"
        else:
            print('CPU only.')
        self.params = rparams
        self.data_path = data_path

        self.npoin = npoin
        self.x_left = x_left
        self.x_right = x_right
        self.idx_obs = idx_obs
        self.batch_size = batch_size
        self.load_path = load_path
        self.load_pathDs2 = load_pathDs2
        self.load_pathEs2 = load_pathEs2
        self.load_pathF1 = load_pathF1
        self.load_pathD1 = load_pathD1
        self.save_lossfile = save_lossfile
        

    def train_test(self, out_path):
        Encoder = UNetV2_Encoder(in_channels=1, num_classes=1).to(self.device)
        Encoder.load_state_dict(torch.load(self.load_pathF1))
        # Encoder.load_state_dict(torch.load(self.load_pathEs2))
        model = MLPModel().to(self.device)
        # model.load_state_dict(torch.load(self.load_path))
        Decoder = UNetV2_Decoder(in_channels=1, num_classes=1).to(self.device)
        # Decoder.load_state_dict(torch.load(self.load_pathD1))
        # Decoder.load_state_dict(torch.load(self.load_pathDs2))
        summary_writer = SummaryWriter(out_path)

        train_size = 1000
        test_size = 100
        train_dataset = HDF5Dataset(self.data_path, train_size, self.npoin, self.x_left, self.x_right, self.idx_obs,self.batch_size)
        test_dataset = HDF5Dataset(self.data_path, test_size, self.npoin, self.x_left, self.x_right,
                                          self.idx_obs, self.batch_size)
        print(f"No. train samples = {train_size}, test samples = {test_size}.")
        self.params["num_train_samples"] = train_size

        train_and_test(Encoder,model,Decoder,train_dataset, test_dataset,train_size,test_size,
                            self.params["training"], summary_writer, self.device, self.batch_size,self.load_path,self.load_pathDs2,self.load_pathEs2,self.save_lossfile)
    
    def validate(self):
        print("\nValidating.")
        scores = []
        Encoder = UNetV2_Encoder(in_channels=1, num_classes=1).to(self.device)
        Encoder.load_state_dict(torch.load(self.load_pathF1))
        model = MLPModel().to(self.device)
        model.load_state_dict(torch.load(self.load_path))
        Decoder = UNetV2_Decoder(in_channels=1, num_classes=1).to(self.device)
        Decoder.load_state_dict(torch.load(self.load_pathD1))
        validate_set = HDF5Dataset(self.data_path, 100, self.npoin, self.x_left, self.x_right,self.idx_obs, self.batch_size)
        # print(f"Loaded dataset with total {len(validate_set)} samples for validation.")
        score = validate(Encoder,model,Decoder,validate_set,100,self.batch_size, self.device)
        #print(f"Validation R2score = {score}")
        scores.append(score)
        return scores


if __name__ == "__main__":
    ix = []
    for i in range(0, 4):
        #print(i)
        for j in range(0, 64, 16):
            #print(j)
            ix.append(16 * i + 64 * j + 8 + 8 * 64)
    parser = argparse.ArgumentParser(
        description="Trains and tests a 2D problem")
    
    parser.add_argument('--inpath', default='./Heat_Types10_source4_number1000_Normalizedn64.npz', help="Path to input training data")
    parser.add_argument('--npoin', default=64, type=int)
    parser.add_argument('--x_left', default=0, type=float)
    parser.add_argument('--x_right', default=2.0, type=float)
    parser.add_argument('--idx_obs', default=np.array(ix))
    parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--seq_net', default=[2, 50, 50, 50, 25, 1])
    # parser.add_argument('--activation', default=torch.tanh)
    parser.add_argument('--load_path', default='./MLPu1000_onlyf.pth')
    parser.add_argument('--load_pathDs2', default='./modelDs21000_onlyf.pth')
    parser.add_argument('--load_pathEs2', default='./modelEs21000_onlyu.pth')
    parser.add_argument('--load_pathF1', default='./modelF1.pth')
    parser.add_argument('--load_pathD1', default='./modelD1.pth')
    parser.add_argument('--save_lossfile', default='./lossFD1uf1000.csv')
    parser.add_argument('--outpath', default=".", help="Directory for output files")
    parser.add_argument("--report_file", help="File to append validation scores to")
    parser.add_argument('--validation_set_path',
                        help="Path to validation data; see also the JSON input")
    parser.add_argument('--save_model', action="store_true",
                        help="Pass this flag to request writing the trained model out to a file.")
    args = parser.parse_args(args=[])



    # sane defaults
    params = {
        "training": {
            "batch_size": 32, "learning_rate": 1e-3, "num_epochs": 10, "shuffle": True
        }
    }

  

    exp = Experiment(params, args.inpath, args.npoin, args.x_left, args.x_right, args.idx_obs, args.batch_size, args.load_path, args.load_pathDs2, args.load_pathEs2,args.load_pathF1,args.load_pathD1,args.save_lossfile)
    exp.train_test(out_path=args.outpath)

    #scores = exp.validate()
    
  
