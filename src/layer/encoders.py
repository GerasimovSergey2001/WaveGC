import torch
import torch.nn as nn

from src.layer.utils import FFN


class TrigonometricEncoder(nn.Module):

    def __init__(self, hidden_dim, eps, base=10_000, project=True):
        super().__init__()
        self.d = hidden_dim
        self.eps = eps
        powers = 2 / self.d * torch.arange(0, self.d, 2, dtype=torch.float32)
        base = torch.tensor([base], dtype=torch.float32)
        self.den = torch.exp(-torch.log(base) * powers)
        self.proj = nn.Linear(self.d + 1, self.d)
        self.project = project
        self.register_buffer("denominator", self.den)

    def forward(self, eigvs: torch.tensor):
        """
        eigvs: [B, N]
        """
        x = self.eps * eigvs[:, :, None] * self.den

        if self.project:
            x = torch.cat([eigvs[:, :, None], torch.sin(x), torch.cos(x)], axis=2)
            return self.proj(x)
        else:
            return eigvs[:, :, None] + torch.cat([torch.sin(x), torch.cos(x)], axis=2)
        
class WaveletCoefs(nn.Module):

    def __init__(self, hidden_dim, heads_num, dropout, eps=100, hidden_num=0):
        super().__init__()
        self.encoder = TrigonometricEncoder(hidden_dim, eps, project=True)
        self.layer_norm_transformer = nn.LayerNorm(hidden_dim)
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
        self.transformer = nn.MultiheadAttention(
            hidden_dim, heads_num, dropout, batch_first=True
        )
        self.ffn = FFN(hidden_dim, hidden_dim, hidden_dim, hidden_num)
        self.transformer_dropout = nn.Dropout(dropout)
        self.ffn_droput = nn.Dropout(dropout)

    def forward(self, eigvs, eigvs_mask=None):
        z = self.encoder(eigvs)
        z = self.layer_norm_transformer(z)
        z, _ = self.transformer(z, z, z, key_padding_mask=eigvs_mask)
        z = z + self.transformer_dropout(z)
        z = z + self.ffn_droput(self.ffn(self.layer_norm_ffn(z)))
        return z
        

class DataProcessing(nn.Module):
      
      def __init__(self, inp_dim, hidden_dim, heads_num, scale, K=6, J=5, dropout=0.1, eps=100, hidden_num=0):
        super().__init__()

        self.rho = K//2
        self.J = J

        self.proj_features = FFN(inp_dim=inp_dim, out_dim=hidden_dim, hidden_num=hidden_num)
        self.proj_a = nn.Linear(hidden_dim, self.rho)
        self.proj_b = nn.Linear(hidden_dim, self.rho)
        self.proj_s = nn.Linear(hidden_dim, self.J)

        self.sigmoid = nn.Sigmoid()

        self.weight_function = WaveletCoefs(hidden_dim, heads_num, dropout, eps, hidden_num)
       
        self.register_buffer('scale', scale.view(1, self.J))

      def forward(self, features, eigvs, eigvs_mask=None):

        feature_encoding = self.proj_features(features)
        pos_encoding = self.weight_function(eigvs, eigvs_mask)
        a_tilde = self.proj_a(pos_encoding).mean(dim=-2) 
        b_tilde = self.proj_b(pos_encoding).mean(dim=-2)
        scale_tilde = self.sigmoid( self.proj_s(pos_encoding).mean(dim=-2) ) * self.scale 
        return torch.cat([feature_encoding, pos_encoding], dim=-1), a_tilde, b_tilde, scale_tilde
