import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from datasets.FeatureDataset import FeatureDataset
from datasets.Edges import Edges
from utils.sampler import NeighborSampler
from utils.metrics import top_k_eval
from models.MMAHGNN import MMAHGNN
import numpy as np
import argparse
import time

localtime = time.localtime(time.time())
start_time = time.time()
launchTimestamp = "{}{}-{}:{}".format(localtime.tm_mon,localtime.tm_mday,localtime.tm_hour,localtime.tm_min)
print("launchTimestamp {}".format(launchTimestamp))
parser = argparse.ArgumentParser(description='Setting')

parser.add_argument('-s', dest='semantic_type', action='store',choices={'btm_o2m'}, default='btm_o2m',
                    help='semantic_type')
parser.add_argument('-l', dest='lochis_type', action='store',choices={'cmeans'}, default='cmeans',
                    help='lochis_type')
parser.add_argument('-n_s', dest='n_tpc', action='store',choices={'30','40','50','60','70','80','90','100'}, default='0',
                    help='n_tpc')
parser.add_argument('-nhead', dest='nhead',type=int, action='store', default=1,
                    help='nhead')
parser.add_argument('-nlayer', dest='nlayer',type=int, action='store', default=1,
                    help='nlayer')                                
parser.add_argument('-n_c', dest='n_cluster', action='store',choices={'30','35','40','45','50','55','60','65','70'}, default='30',
                    help='n_cluster')
args = parser.parse_args()

# Output the collected arguments
print("semantic_type {}".format(args.semantic_type))
print("lochis_type {}".format(args.lochis_type))
print("n_tpc {}".format(args.n_tpc))
print("nhead {}".format(args.nhead))
print("nlayer {}".format(args.nlayer))
print('n_cluster {}'.format(args.n_cluster))

city = "Region1"
semantic_type=args.semantic_type
lochis_type=args.lochis_type
n_tpc = int(args.n_tpc)
nhead = args.nhead
nlayer = args.nlayer
n_cluster=int(args.n_cluster)

feature_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Feature')
edge_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Edges')
checkpoint_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoint')


dataset = FeatureDataset(root = feature_path,city=city,semantic_type=semantic_type,lochis_type=lochis_type,n_tpc=n_tpc,n_cluster=n_cluster) 
edges = Edges(root = edge_path, city=city)
data = dataset.data

_edge,group_dict,group_mask = edges.process_data(num_nodes=data.num_nodes,num_negs=5)
edge_attr = edges.get_edge_attr()
train_edge,test_dict,train_edge_attr = edges.train_test_split_edge(_edge,test_size=0.15,random_state=38,edge_attr=edge_attr)
neg_dict=edges.get_neg_dict(pos_dict=group_dict,test_dict=test_dict,max_size=data.num_nodes,n=5)

data.edge_attr = torch.unsqueeze(train_edge_attr,-1)
data.edge_index=train_edge

node_idx = torch.tensor(group_mask)
input_dim_list=[n_cluster, n_tpc,n_tpc]
    
print("input_dim_list {}".format(input_dim_list))
print('train_edge {}'.format(train_edge.shape[1]))
print('test_dict {}'.format(len(test_dict)))
print('data.num_nodes {}'.format(data.num_nodes))
print('data.num_node_features {}'.format(data.num_node_features))
print('train_edge_attr {}'.format(data.edge_attr.shape))


n_edge_channels = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = MMAHGNN(data.num_node_features, hidden_channels=64, edge_channels=n_edge_channels,
                input_dim_list=input_dim_list,nhead = nhead,nlayer=nlayer, num_layers=2)  # input dim list assumes that the node features are first

train_loader = NeighborSampler(data.edge_index, sizes=[15,10], batch_size=256,shuffle=True,node_idx= node_idx)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = data.x.to(device)


def train(loader):
    model.train()

    total_loss = 0

    for batch_size, n_id, adjs in loader:

        edge_attrs = [data.edge_attr[e_id] for (edge_index, e_id, size) in adjs]
        adjs = [adj.to(device) for adj in adjs]
        edge_attrs = [edge_attr.to(device) for edge_attr in edge_attrs]
        

        optimizer.zero_grad()
        
        out= model(x[n_id], adjs, edge_attrs)
        
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        

        # binary skipgram loss can be replaced with margin-based pairwise ranking loss.
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        
        loss = -pos_loss - neg_loss
        
        
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    
    x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
    
    model.eval()
    out = model.full_forward(x, edge_index, edge_attr).cpu()
    return out   

def get_rank(out,pos_dict,neg_dict):
    rank_dict={}
    for node,pos in pos_dict.items():
        neg = neg_dict[node]
        test_node = [*pos,*neg]
        grade = [float(F.logsigmoid((out[node]*out[i]).sum(-1))) for i in test_node]

        rank_ =sort_grade(grade,node=test_node)
        rank_dict[node]=rank_
       
    return rank_dict

def sort_grade(grade,node):
    sorted_ = sorted(zip(grade,node))
    _, rank_ = zip(*sorted_)
    rank_=list(rank_)
    rank_.reverse()
    return rank_

def get_metrics(out,test_dict,neg_dict,k):
    rank_dict = get_rank(out,pos_dict=test_dict,neg_dict=neg_dict)
    hit_=[]
    precision_=[]
    recall_=[]
    map_metrics_=[]
    ndcg_=[]
    mrr_=[]

    for i in k:
        hit, precision, recall, map_metrics, ndcg, mrr = top_k_eval(rank_dict, gt_dict=test_dict, k=i)
        hit_.append(hit)
        precision_.append(precision)
        recall_.append(recall)
        map_metrics_.append(map_metrics)
        ndcg_.append(ndcg)
        mrr_.append(mrr)
    return hit_,precision_,recall_,map_metrics_,ndcg_,mrr_



    
best_loss = 1000
record_metrics =[]
for epoch in range(1, 50):
    loss = train(train_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch<47:
        continue
    st_time=time.time()
    out = test()

    k= [3,5,10,15]
    hit, precision, recall, map_metrics, ndcg, mrr = get_metrics(out,test_dict,neg_dict,k)
    
    for i in range(len(k)):
        k_=k[i]
        print('hit_rate@{}:{} precision@{}:{} recall@{}:{} map@{}:{} ndcg@{}:{} mrr@{}:{}'.format(
            k_,hit[i],
            k_,precision[i],
            k_,recall[i],
            k_,map_metrics[i],
            k_,ndcg[i],
            k_,mrr[i]))
    i=0
    metric_tmp = 'hit_rate@{}:{} precision@{}:{} recall@{}:{} map@{}:{} ndcg@{}:{} mrr@{}:{}'.format(
            k[i],hit[i],
            k[i],precision[i],
            k[i],recall[i],
            k[i],map_metrics[i],
            k[i],ndcg[i],
            k[i],mrr[i])

    record_metrics.append(metric_tmp)
    en_time=time.time()
    print("test model in {} mins".format((en_time-st_time)/60))
    
    # torch.save({'epoch': epoch + 1,
    # 'state_dict': model.state_dict(),
    # 'optimizer': optimizer.state_dict()},
    # osp.join(checkpoint_path ,'m-{}-{}-epoch{}-hr3@{}.pth.tar'.format(city,model_type,epoch+1,hit[0])))
        
        
print("===================================")
for i in record_metrics:
    print(i)
print("===================================")
end_time = time.time()
print("Finished in {} mins".format((end_time-start_time)/60))