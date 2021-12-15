import torch
from torch_geometric.data import InMemoryDataset, download_url,Data
from itertools import chain
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np

class FeatureDataset(InMemoryDataset):
    def __init__(self,root,city,semantic_type,lochis_type='cmeans',n_cluster=30,n_tpc=100, transform=None, pre_transform=None):
        self.city = city
        self.semantic_type = semantic_type
        self.lochis_type = lochis_type
        self.n_tpc = n_tpc
        self.n_cluster=n_cluster
        super(FeatureDataset, self).__init__(root,transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        '''
        We only upload Region1 dataset, which is produced by Multi-Modal Encoder:
        - Location encoder: fuzzy c-means with n_c = 65
        - Semantic service encoder: Biterm Topic Model(BTM) with n_s = 70
        We concatenate the three modals as input.
        '''
        return ['{}_{}{}_{}{}_data.pt'.format(self.city,self.semantic_type,self.n_tpc,self.lochis_type,self.n_cluster)]

    def download(self):
        pass

    def process(self):
        print("Start to process FeatureDataset")
        pass

