import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from src.layer.utils import FFN

class ChebNet(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, K, num_layers, dropout=0.0, res_connection=True):
        super().__init__()
        self.out_dim = out_dim 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.res_connection = res_connection
        
        self.K = K 

        self.norm = nn.BatchNorm1d
        self.act = nn.GELU
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        for i in range(num_layers):
            inp = inp_dim if i==0 else hidden_dim
            
            self.convs.append(ChebConv(inp, hidden_dim, K=self.K)) 
            self.norms.append(self.norm(hidden_dim))
            self.acts.append(self.act())
            self.dropouts.append(nn.Dropout(self.dropout_rate))
            
            if self.res_connection and inp != hidden_dim:
                 self.skips.append(nn.Linear(inp, hidden_dim, bias=False))
            else:
                 self.skips.append(nn.Identity())
        
        self.out = FFN(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        
        for i in range(self.num_layers):
            
            skip_input = x
            
            x = self.convs[i](x, edge_index)
            
            x = self.norms[i](x) 
            
            if self.res_connection:
                skip_val = self.skips[i](skip_input)
                x = x + skip_val
            
            x = self.acts[i](x)
            
            x = self.dropouts[i](x)
                
        return self.out(x)