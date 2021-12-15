import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import InterModalFusion,IntraModalGraphConv
from torch_geometric.nn.conv import GCNConv
from torch.nn import Parameter

class MMAHGNN(nn.Module):
    """
    Multi-Modal Attention-based Hierarchical Graph Neural Network (MM-AHGNN) for object interaction recommendation
    with activity location features, semantic service features and pairwise interaction features:
    (a) Multi-modal encoder: has been processed in dataset.
    (b) Intra-modal graph convolution based on hybrid-attention: IntraModalGraphConv
    (c) Inter-modal fusion based on transformer encoder: InterModalFusion
    (d) Multi-modal multi-scale encoder: we use GCN as summary convolution
    """

    def __init__(self, in_channels, hidden_channels, edge_channels, input_dim_list,nhead,nlayer,num_layers=2):
        """
        :param in_channels: total cardinality of node features.
        :param hidden_channels: latent embedding dimensionality.
        :param edge_channels: number of interaction features.
        :param input_dim_list: list containing the cardinality of node features per modality.
        :param nhead: number of head in transformer encoder.
        :param nlayer: number of sub-encoder-layers in transformer encoder.
        :param num_layers: number of message passing layers. we assume 2 layers intra-modal fusion.
        """
        super(MMAHGNN, self).__init__()
        self.num_layers = num_layers
        self.intra_modal_conv2 = nn.ModuleList()
        self.intra_modal_conv1 = nn.ModuleList()
        self.edge_channels = edge_channels
        
        self.input_dim_list = input_dim_list
        self.summary_convs = nn.ModuleList()
        self.multi_linear = nn.Linear(hidden_channels*len(input_dim_list), hidden_channels, bias=True)
        self.multi_attn = nn.Sequential(self.multi_linear, nn.Tanh(), nn.Linear(hidden_channels, 1, bias=True))
        
        # init multi-level fusion Summary Convolution: GCNConv
        for inp_dim in self.input_dim_list:
            summary_conv_list = nn.ModuleList()
            for i in range(num_layers):
                in_channels = in_channels if i == 0 else hidden_channels
                summary_conv_list.append(GCNConv(hidden_channels, hidden_channels,aggr='mean'))
            self.summary_convs.append(summary_conv_list)

        # init 2 layers IntraModalGraphConv
        for inp_dim in self.input_dim_list:
            module_list = nn.ModuleList()
            for i in range(num_layers):
                in_channels = inp_dim if i == 0 else hidden_channels
                module_list.append(IntraModalGraphConv((hidden_channels + edge_channels, in_channels), hidden_channels,normalize=True))  
            self.intra_modal_conv2.append(module_list) # recursive with two layer of intra-modal graph conv
            # one layer of intra-modal graph conv
            self.intra_modal_conv1.append(IntraModalGraphConv((hidden_channels + edge_channels, inp_dim), hidden_channels,normalize=True)) 

        # init InterModalFusion
        self.inter_modal_fusion1 = InterModalFusion(ninp=hidden_channels, nhead=nhead, nhid=hidden_channels*2, nlayers=nlayer, dropout=0.1)
        self.inter_modal_fusion2 = InterModalFusion(ninp=hidden_channels, nhead=nhead, nhid=hidden_channels*2, nlayers=nlayer, dropout=0.1)
        self.summary_inter_modal_fusion1 = InterModalFusion(ninp=hidden_channels, nhead=nhead, nhid=hidden_channels*2, nlayers=nlayer, dropout=0.1)


    def forward(self, x, adjs, edge_attrs):
        """ Compute node embeddings by recursive intra-modality graph conv, followed by inter-modal fusion and multi-modal multi-scale encoder.
        :param x: node features [B, in_channels] where B is the number of nodes and neighbors for each batch.
        :param adjs: list of sampled edge index per layer in PyTorch Geometric for each batch.
        :param edge_attrs: [E, edge_channels] where E is the number of sampled edge indices per layer in the mini-batch.
        :return: node embeddings. [B', hidden_channels] where B' is the number of target nodes for each batch.
        """
        result1 = []
        result2 = []
        
        summary1 = []
        
        
        for k, convs_k in enumerate(self.intra_modal_conv2):
            
            emb_k2=None # 2-layer embedding for k-th modality
            emb_k1=None # 1-layer embedding for k-th modality
            summary_k1 = None # local context embedding for k-th modality
            
            target_len=0
            for i, ((edge_index, _, size), edge_attr) in enumerate(zip(adjs, edge_attrs)):
                target_len = size[1]
                
                # context embedding of 2-layer embedding
                if i==1:
                    summary_k1 = self.summary_convs[k][0](emb_k2,edge_index)

                # recursive convolution with 2-layer
                if i ==0:
                    x_target = x[:size[1]]  # Target nodes are always placed first.
                    x_list = torch.split(x, split_size_or_sections=self.input_dim_list, dim=-1)  # modality partition
                    x_target_list = torch.split(x_target, split_size_or_sections=self.input_dim_list, dim=-1)
                    x_k, x_target_k = x_list[k], x_target_list[k]
                else:
                    x_k = emb_k2
                    x_target_k=x_k[:size[1]]
                    
                emb_k2 = convs_k[i]((x_k, x_target_k), edge_index, edge_attr=edge_attr)
                    

                if i != self.num_layers - 1:
                    emb_k2 = emb_k2.relu()
                    emb_k2 = F.dropout(emb_k2, p=0.5, training=self.training)

                # convolution with 1-layer
                if i==1:
                    x_target = x[:size[1]]  # Target nodes are always placed first.
                    x_list = torch.split(x, split_size_or_sections=self.input_dim_list, dim=-1)  # modality partition
                    x_target_list = torch.split(x_target, split_size_or_sections=self.input_dim_list, dim=-1)
                    x_k, x_target_k = x_list[k], x_target_list[k]
                    
                    emb_k1=self.intra_modal_conv1[k]((x_k, x_target_k), edge_index,edge_attr=edge_attr)

                

            
            emb_k1 = emb_k1[:target_len]
            emb_k2 = emb_k2[:target_len]
            summary_k1 =summary_k1[:target_len]
            
            result1.append(emb_k1)
            result2.append(emb_k2)
            
            summary1.append(summary_k1)
            
        # inter-modal fusion
        inter_res1 = self.inter_modal_fusion1(result1)
        inter_res2 = self.inter_modal_fusion2(result2)
        inter_sum1 = self.summary_inter_modal_fusion1(summary1)
        
        # fuse multi-scale embedding
        result = torch.cat([inter_res1.unsqueeze(-2),inter_res2.unsqueeze(-2),inter_sum1.unsqueeze(-2)], -2)  # [..., 3, hidden_channels*K]
        wts = torch.softmax(self.multi_attn(result).squeeze(-1), dim=-1)
        res = torch.sum(wts.unsqueeze(-1) * result, dim=-2)
        
        return res


    def full_forward(self, x, edge_index, edge_attr):
        """ Auxiliary function to compute node embeddings for all nodes at once for small graphs.
        :param x: node features [N, in_channels] where N is the total number of nodes in the graph.
        :param edge_index: edge indices [2, E] where E is the total number of edges in the graph.
        :param edge_attr: interaction features [E, edge_channels] across all edges in the graph.
        :return: node embeddings. [N, hidden_channels] for all nodes in the graph.
        """
        x_list = torch.split(x, split_size_or_sections=self.input_dim_list, dim=-1)  # modality partition
        result1 = []
        result2 = []
        
        summary1=[]
        
        for k, convs_k in enumerate(self.intra_modal_conv2):
            x_k = x_list[k]
            emb_k1 = None
            emb_k2 = None
            
            for i, conv in enumerate(convs_k):
                
                if i==0:
                    emb_k2 = conv(x_k, edge_index, edge_attr=edge_attr)
                else:
                    emb_k2=conv(emb_k2, edge_index, edge_attr=edge_attr)

                if i != self.num_layers - 1:
                    emb_k2 = emb_k2.relu()
                    emb_k2 = F.dropout(emb_k2, p=0.5, training=self.training)
            
            
            emb_k1=self.intra_modal_conv1[k](x_k, edge_index, edge_attr=edge_attr)
            summary_k1 = self.summary_convs[k][0](emb_k1,edge_index)
            result1.append(emb_k1)
            result2.append(emb_k2)
            summary1.append(summary_k1)
            
            
        
        inter_res1 = self.inter_modal_fusion1(result1)
        inter_res2 = self.inter_modal_fusion2(result2)
        inter_sum1 = self.summary_inter_modal_fusion1(summary1)
        
        
        result = torch.cat([inter_res1.unsqueeze(-2),inter_res2.unsqueeze(-2),inter_sum1.unsqueeze(-2)], -2)  # [...., 2, hidden_channels*K]
        
        wts = torch.softmax(self.multi_attn(result).squeeze(-1), dim=-1)
        print("wts {}".format(wts[0])) # [3*Batch_size, 3]
        res = torch.sum(wts.unsqueeze(-1) * result, dim=-2)
        
        
        return res
        

    