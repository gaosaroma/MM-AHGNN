import torch
from sklearn.model_selection import train_test_split
from itertools import chain
import os.path as osp
import numpy as np

class Edges():
    def __init__(self,root,city):
        self.root = root
        self.city = city

    @property
    def processed_file_names(self):
        dic = {
            'edge':'{}_edge.pt'.format(self.city),
            'dict':'{}_dict.pt'.format(self.city),
            'mask':'{}_mask.pt'.format(self.city),
            'train_edge':'{}_train_edge.pt'.format(self.city),
            'train_edge_attr':'{}_train_edge_attr.pt'.format(self.city),
            'edge_attr':'{}_edge_attr.pt'.format(self.city),
        }
        return dic

    def process_data(self,num_nodes,num_negs):
        print("Process Edge Done!")
        group_edge = self.get_group_edge()
        group_dict =self.get_group_dict()
        group_mask =self.get_group_mask()
        
        print("Process Edge Done!")
        return group_edge,group_dict,group_mask

    def get_group_edge(self):
        edge_path = osp.join(self.root,self.processed_file_names['edge'])    
        edge = torch.load(edge_path)
        return edge

    def get_group_dict(self):
        dict_path = osp.join(self.root,self.processed_file_names['dict'])
        edge = torch.load(dict_path)
        return edge
    
    def get_group_mask(self):
        mask_path = osp.join(self.root,self.processed_file_names['mask'])
        edge = torch.load(mask_path)
        return edge

    def get_edge_attr(self):
        edge_attr_path= osp.join(self.root,self.processed_file_names['edge_attr'])
        edge_attr = torch.load(edge_attr_path)
        return edge_attr

    def train_test_split_edge(self,edges,test_size,random_state,edge_attr):
        
        train_edge_path = osp.join(self.root,"{}_rs{}_train_edge.pt".format(self.city,random_state))
        test_dict_path=osp.join(self.root,"{}_rs{}_test_dict.pt".format(self.city,random_state))
        train_edge_attr_path = osp.join(self.root,"{}_rs{}_train_edge_attr.pt".format(self.city,random_state))
        if osp.exists(train_edge_attr_path) and osp.exists(train_edge_path) and osp.exists(test_dict_path):
            train_edge = torch.load(train_edge_path)
            test_dict = torch.load(test_dict_path)
            train_edge_attr = torch.load(train_edge_attr_path)
            return train_edge, test_dict, train_edge_attr

        print("Start train_test_split_edge")
        train_index, test_index = train_test_split([[i]for i in range(edges.shape[1])], test_size=test_size, random_state=random_state)


        # process test edges
        test_index = list(chain(*test_index))
        test_source_nodes = [int(edges[0][i]) for i in test_index]
        test_target_nodes = [int(edges[1][i]) for i in test_index]

        test_dict = dict()
        for i in range(len(test_source_nodes)):
            source = test_source_nodes[i]
            target = test_target_nodes[i]

            if test_dict.get(target) == None:
                test_dict[target]=[source]
            else:
                test_dict[target].append(source)

        torch.save(test_dict, test_dict_path)

        # process train edges
        train_index = list(chain(*train_index))
        train_source_nodes = []
        train_target_nodes = []
        train_edge_attrs = []
        for i in train_index:
            _s = int(edges[0][i])
            _t = int(edges[1][i])
            if test_dict.get(_s)!=None and _t in test_dict[_s]:
                continue

            train_source_nodes.append(_s)
            train_target_nodes.append(_t)
            
            train_edge_attrs.append(edge_attr[i])

        train_edge = torch.tensor([train_source_nodes, train_target_nodes], dtype=torch.long)
        torch.save(train_edge, train_edge_path)

        # process edge_attrs
        train_edge_attr = torch.tensor(train_edge_attrs,dtype=torch.float32)
        torch.save(train_edge_attr,train_edge_attr_path)
        
                  
        print('edges.shape[1] {}'.format(edges.shape[1]))
        print('train_index {}'.format(len(train_index)))
        print('train_edge[0] {}'.format(len(train_edge[0])))
        print('test_index {}'.format(len(test_index)))
        print('avg edge for test {}'.format(len(test_index)/len(test_dict)))
        print("Split Edge Done!")
        
        return train_edge,test_dict,train_edge_attr
    
            

    def get_neg_dict(self,pos_dict,test_dict,max_size,n):
        
        dict_path = osp.join(self.root,"{}_{}_test_negs.pt".format(self.city,n))
        if osp.exists(dict_path):
            res = torch.load(dict_path)
            return res

        print("Start Process Neg")
        res ={}
        
        for node,activate in test_dict.items():
            
            neg_n = n*len(activate) # sample n negatives for one positive
            all_act = pos_dict[node]
            neg = self.get_n_notin_list(l=all_act,max_size=max_size,n=neg_n)
            
            if neg_n!=len(neg):
                print("Error, there is some Neg not Equal!")
            
            res[node] = neg


        torch.save(res,dict_path)
        
        print("Process {} Neg Done!")
        return res
        

    def get_n_notin_list(self,l,max_size,n):
        
        set_l = set(l) 
        res = set(np.random.choice(max_size, size=n, replace=False, p=None).tolist())
        for i in range(n):
            res = res - set_l
            if len(res) == n:
                break
            res = res | set(np.random.choice(max_size, size=n-len(res), replace=False, p=None).tolist())
        return list(res)
        