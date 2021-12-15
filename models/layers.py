from typing import Optional
from typing import Union, Tuple
import torch
import copy

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_geometric.utils import softmax
from torch.nn import TransformerEncoder, TransformerEncoderLayer,LayerNorm,Dropout
import torch.nn.functional as F
from torch_sparse import SparseTensor

class IntraModalGraphConv(MessagePassing):
    """
    Intra-modal Graph Conv of MM-AHGNN is based on Hybrid Attention to investigate
    the three modality representations independently for each object.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(IntraModalGraphConv, self).__init__(** kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.negative_slope = 0.2
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.self_linear = nn.Linear(in_channels[1], out_channels, bias=bias)
        self.message_linear = nn.Linear(in_channels[1], out_channels, bias=bias)
        

        self.attn_sim = nn.Linear(out_channels, 1, bias=bias)
        self.attn_channel = nn.Linear(out_channels, out_channels, bias=bias)
        self.attn_inf = nn.Linear(out_channels, 1, bias=bias)
        self.inter_attn = nn.Linear(3, 1, bias=False)
        
        self.lin_l = nn.Linear(out_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(out_channels, out_channels, bias=False)

        self.reset_parameters()
        self.dropout = 0

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x_src, x_trg = x[0], x[1]
        
        trg_emb = self.self_linear(x_trg)
        trg_emb = F.elu(trg_emb, self.negative_slope)
        
        out = self.propagate(edge_index, x=(x_src, x_trg), size=size,selfemb=trg_emb, edge_attr=edge_attr)
        
        # channel attention
        alpha = self.attn_channel(out)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha,dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = out*alpha

        # residual-add operation
        out = out+trg_emb
       
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, selfemb_i: Tensor,edge_attr: Tensor,index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """
        :param x_j: the features of all x, size([num_x,dim_features])
        :param edge_attr: has been processed as `alpha_int` in dataset
        """
    
        out = self.message_linear(x_j)
        out = F.elu(out, self.negative_slope)
        
        alpha_sim = self.attn_sim(out*selfemb_i)
        alpha_sim = F.leaky_relu(alpha_sim, self.negative_slope)
        alpha_sim = softmax(alpha_sim, index, ptr, size_i)
        

        alpha_inf = self.attn_inf(out)
        alpha_inf = F.leaky_relu(alpha_inf, self.negative_slope)
        alpha_inf = softmax(alpha_inf, index, ptr, size_i)

        
        # alpha: multi-attentional coefficient
        alpha = self.inter_attn(torch.cat([alpha_inf,edge_attr,alpha_sim],dim=-1))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return out*alpha
        

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class InterModalFusion(nn.Module):
    """
    Inter-Modal Fusion of MM-AHGNN is implemented by Transformer Encoder across the three modalities.
    """

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(InterModalFusion, self).__init__()
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.nhid = nhid
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nhid) 
        self.init_weights()
     

    def _generate_zero_mask(self, sz):
        mask = torch.ones(sz, sz)
        mask = mask.float().masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, modality_list):
        """
        :param modality_list: list of intra-modal embeddings.
        :return final object embedding after inter-modal fusion.
        """

        src = torch.cat([x.relu().unsqueeze(0) for x in modality_list], 0)  
        self.src_mask = self._generate_zero_mask(len(modality_list)).to(src.device)

        res = self.transformer_encoder(src, self.src_mask)
        res = res.permute(1,0,2) 
        res = res.reshape(res.shape[0],res.shape[1]*res.shape[2])
        return res

        

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.nhid,
                                   self.nhid)
