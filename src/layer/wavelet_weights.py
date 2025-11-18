import torch
import torch.nn as nn

from ..layer.encoders import TrigonometricEncoder


class FFN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, hidden_num, act=nn.SiLU):
        super().__init__()
        self.ffn = [nn.Linear(inp_dim, hidden_dim), act()]
        
        for _ in range(hidden_num):
            self.ffn.append(nn.Linear(hidden_dim, hidden_dim))
            self.ffn.append(act())
        
        self.ffn.append(nn.Linear(hidden_dim, out_dim))
        self.ffn = nn.Sequential(*self.ffn)

    def forward(self, x):
        return self.ffn(x)
    
class WaveletCoefs(nn.Module):
    
    def __init__(self, hidden_dim, dropout, heads_num, eps=100, hidden_num=0, act=nn.SiLU):
        super().__init__()
        self.encoder = TrigonometricEncoder(hidden_dim, eps, project=True)
        self.layer_norm_transformer = nn.LayerNorm(hidden_dim)
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
        self.transformer = nn.MultiheadAttention(hidden_dim, heads_num, dropout, batch_first=True)
        self.ffn = FFN(hidden_dim, hidden_dim, hidden_dim, hidden_num, act)
        self.transformer_dropout = nn.Dropout(dropout)
        self.ffn_droput = nn.Dropout(dropout)

    def forward(self, eigvs, eigvs_mask=None):
        x = self.encoder(eigvs)
        x = self.layer_norm_transformer(x)
        x, _ = self.transformer(x, x, x, key_padding_mask=eigvs_mask)

        x = x + self.transformer_dropout(x)
        
        x = x + self.ffn_droput(self.ffn(self.layer_norm_ffn(x)))
        return x 


        