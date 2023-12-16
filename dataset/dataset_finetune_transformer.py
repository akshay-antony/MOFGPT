from __future__ import print_function, division

import csv
import functools
import  json
#import  you
import  random
import warnings
import math
import  numpy  as  np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm



class CORE_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1, which_label = 'void_fraction'):
            label_dict = {
                'void_fraction':2,
                'pld':3,
                'lcd':4
            }
            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, label_dict[which_label]].astype(float)
            # self.label = self.label/np.max(self.label)
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return X, y.float()

class MOF_ID_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, 
                 data, 
                 tokenizer,
                 ignore_index):
            self.data = data
        #     self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 0].astype(str)
        #     self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.tokens = []
            loop = tqdm(range(len(self.mofid)), 
                        desc='Tokenizing', 
                        colour='green')
            for i in loop:
                curr_token = tokenizer.encode(self.mofid[i])
                self.tokens.append(curr_token)
        #     self.tokens = np.array(self.tokens)
            print("Tokenizing finished")
            print(f"Number of mofs: {len(self.tokens)}")
            self.label = self.data[:, 1].astype(float)
            self.tokenizer = tokenizer
            self.ignore_index = ignore_index

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            token_ids = torch.from_numpy(np.asarray(self.tokens[index]))
            target_token_ids = token_ids.clone()[1:]
            mask_ids = torch.ones_like(token_ids)
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return {'token_ids':token_ids, 
                    'mask_ids':mask_ids, 
                    'target_token_ids':target_token_ids,
                    'label':y.float()}
    
    def collate_fn(self, data):
        """
        add padding to the batch of data
        """
        padded_tokens, \
        padded_masks, \
        target_tokens = self.tokenizer.pad_batched_tokens([i['token_ids'] for i in data],
                                                          [i['mask_ids'] for i in data],
                                                          [i['target_token_ids'] for i in data])
        labels = torch.stack([i['label'] for i in data])
        return {"token_ids":padded_tokens,
                "mask_ids":padded_masks,
                "target_token_ids":target_tokens,
                "label":labels}


class MOF_pretrain_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data.astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.mofid)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))

            return X.type(torch.LongTensor)


class MOF_tsne_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer):
            self.data = data
            self.mofid = self.data[:, 0].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 1].astype(float)

            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = self.label[index]
            topo = self.mofid[index].split('&&')[-1].split('.')[0]
            return X, y, topo

