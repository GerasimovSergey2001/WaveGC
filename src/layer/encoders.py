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
        eigvs: [1, N]
        """
        x = self.eps * eigvs[:,:, None] * self.den
        pos_enc = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        if self.project:
            x = torch.cat([eigvs[:,:, None], pos_enc], axis=-1).squeeze(0)
            return self.proj(x)
        else:
            return (eigvs[:, :, None] + pos_enc).squeeze(0)
        
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
    
class LapPE(nn.Module):
    
    def __init__(self, inp_dim, emb_dim, pe_dim, eigvs_dim, hidden_num):
        super().__init__()
        self.dim_emb = emb_dim
        self.norm = nn.BatchNorm1d(eigvs_dim)
        self.pe_encoder = FFN(inp_dim=2, out_dim=pe_dim, hidden_dim=2*pe_dim, hidden_num=hidden_num)
        if emb_dim-pe_dim < 0:
            raise ValueError(f"Positional Encoding embeddings are of size {pe_dim} which is greater than feature embeddings of size {emb_dim}")

        self.proj_x = nn.Linear(inp_dim, emb_dim-pe_dim)


    def forward(self, x, eigvs, U):

        sign = torch.randn((1, U.shape[1]), device=eigvs.device)
        
        if self.training:
            sign[sign>0] = 1
            sign[sign<0] = -1
        
        U = U * sign 

        pe = torch.cat(
            [
            U.unsqueeze(-1), 
            eigvs.expand(U.shape[0],-1).unsqueeze(-1)
            ], 
            dim=-1)
        pe = self.norm(pe)
        pe = self.pe_encoder(pe)
        pe = torch.sum(pe, dim=-2)
        embs = self.proj_x(x)
        return torch.cat([embs, pe], dim=-1)
        

        
class DataProcessing(nn.Module):
      
      def __init__(self, 
                   inp_dim, 
                   emb_dim, 
                   pe_dim, 
                   eigvs_dim,
                   lape_hidden_num,
                   hidden_dim, 
                   heads_num, 
                   scale, 
                   K, 
                   J, 
                   dropout=0.01, 
                   eps=100, 
                   hidden_num=0):
        
        super().__init__()

        self.rho = K//2 if K%2==0 else (K+1)//2
        self.J = J
        self.lape = LapPE(
            inp_dim=inp_dim, 
            emb_dim=emb_dim, 
            pe_dim=pe_dim, 
            eigvs_dim=eigvs_dim, 
            hidden_num=lape_hidden_num)

        self.proj_features = FFN(inp_dim=inp_dim, out_dim=hidden_dim, hidden_num=hidden_num)
        self.proj_a = nn.Linear(hidden_dim, self.rho)
        self.proj_b = nn.Linear(hidden_dim, self.rho)
        self.proj_s = nn.Linear(hidden_dim, self.J)

        self.sigmoid = nn.Sigmoid()

        self.weight_function = WaveletCoefs(hidden_dim, heads_num, dropout, eps, hidden_num)
       
        self.register_buffer('scale', scale.view(1, self.J))

      def forward(self, x, eigvs, U, eigvs_mask=None):

        feature_encoding = self.lape(x, eigvs, U)
        pos_encoding = self.weight_function(eigvs, eigvs_mask)
        a_tilde = self.proj_a(pos_encoding).mean(dim=-2).unsqueeze(0)
        b_tilde = self.proj_b(pos_encoding).mean(dim=-2).unsqueeze(0)
        scale_tilde = self.sigmoid( self.proj_s(pos_encoding).mean(dim=-2) ) * self.scale 
        return feature_encoding, a_tilde, b_tilde, scale_tilde

