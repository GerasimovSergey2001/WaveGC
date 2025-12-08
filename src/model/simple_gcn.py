import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import MessagePassing

class SimpleGCN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, hidden_num=3, **kwargs):
        super().__init__()
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.hidden_dim = hidden_dim
        act = nn.GELU
        self.conv = nn.ModuleList([pyg_nn.GCN(
                                    in_channels=inp_dim,
                                    hidden_channels=hidden_dim,
                                    num_layers=hidden_num,
                                    act = act(),
                                )])
        for i in range(hidden_num):
            self.conv.append( pyg_nn.GCN(
                    in_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    num_layers=3,
                    act = act())
            )
            if i!= hidden_num-1:
                self.conv.append(nn.BatchNorm1d(hidden_dim))
                self.conv.append(act())
        self.out = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, edge_index, **kwargs):
        for module in self.conv: 
            if isinstance(module, (pyg_nn.GCN, MessagePassing)): 
                x = module(x, edge_index)
            else:
                x = module(x) 
                
        return self.out(x)
