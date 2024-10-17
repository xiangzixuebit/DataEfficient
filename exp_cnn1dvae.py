#!/usr/bin/env python3

import os
import argparse
import csv
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from model import Encoder,Decoder
from data_convert import HDF5Dataset1d
from training1dvae import train_and_test,validate
import numpy as np

class Experiment:

    def __init__(self, rparams, data_pathT,data_path, npoin, x_left, x_right, batch_size,load_pathencoder,load_pathdecoder,save_lossfile):

        self.device = "cpu"
        if torch.cuda.is_available():
            print('We have a GPU!')
            self.device = "cuda"
        else:
            print('CPU only.')
        self.params = rparams
        self.data_pathT = data_pathT
        self.data_path = data_path

        self.npoin = npoin
        self.x_left = x_left
        self.x_right = x_right
        self.batch_size = batch_size
        self.load_pathencoder = load_pathencoder
        self.load_pathdecoder = load_pathdecoder
        self.save_lossfile = save_lossfile
        

    def train_test(self, out_path):
        encoder =  Encoder(101, 64, 16, 0.99).to(self.device)
        decoder = Decoder(16, 64, 101, 0.99).to(self.device)
        # model.load_state_dict(torch.load('./modelF1.pth'))
        # summary_writer = SummaryWriter(out_path)
        train_size = 500
        test_size = 500
        train_dataset = HDF5Dataset1d(self.data_pathT,self.data_path, train_size, self.npoin, self.x_left, self.x_right, self.batch_size)
        test_dataset = HDF5Dataset1d(self.data_pathT,self.data_path, test_size, self.npoin, self.x_left, self.x_right,self.batch_size)
        print(f"No. train samples = {train_size}, test samples = {test_size}.")
        self.params["num_train_samples"] = train_size

        train_and_test(encoder, decoder,train_dataset, test_dataset,train_size,test_size,
                            self.params["training"], self.device, self.batch_size,self.load_pathencoder,self.load_pathdecoder,self.save_lossfile)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and tests a 1D problem")
    parser.add_argument('--inpathT', default='./Heat1d_Types10_source4_number100.npz', help="Path to input training data")
    parser.add_argument('--inpath', default='./Heat1d_Types10_source4_number100_Normalized.npz', help="Path to input training data")
    parser.add_argument('--intermidiate_dim', default=32, type=int)
    parser.add_argument('--latent_dim', default=8, type=int)
    parser.add_argument('--npoin', default=100, type=int)
    parser.add_argument('--x_left', default=0, type=float)
    parser.add_argument('--x_right', default=1.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--load_pathencoder', default='./encode900u.pth')
    parser.add_argument('--load_pathdecoder', default='./decode900u.pth')
    parser.add_argument('--save_lossfile', default='./vae900u.csv')
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
            "batch_size": 16, "learning_rate": 1e-3, "num_epochs": 1000, "shuffle": True
        }
    }

  

    exp = Experiment(params, args.inpathT, args.inpath, args.npoin, args.x_left, args.x_right, args.batch_size, args.load_pathencoder,args.load_pathdecoder,args.save_lossfile)
    exp.train_test(out_path=args.outpath)

  
  
