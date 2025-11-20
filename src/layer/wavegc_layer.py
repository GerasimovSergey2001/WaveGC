import torch 
import torch.nn as nn

from src.layer.wavelet_weights import WaveletCoefs

def generate_g(a, scaled_eigvs, rho):
    T_odd= torch.ones_like(scaled_eigvs).to(scaled_eigvs.device)
    T_even = scaled_eigvs
    g = a[:, 0]*T_even
    for i in range(1, rho):
        T_odd = 2*scaled_eigvs*T_even - T_odd
        T_even = 2*scaled_eigvs*T_odd - T_even
        g += a[:, i] * T_even
    return g

def generate_h(b, eigvs, rho):
    T_odd= torch.ones_like(eigvs).to(eigvs.device)
    T_even = eigvs
    h = b[:, 0]*T_odd
    for i in range(1, rho):
        T_odd = 2*eigvs*T_even - T_odd
        T_even = 2*eigvs*T_odd - T_even
        h += b[:, i] * T_odd
    return h

        

class WaveConv(nn.Module):

    def __init__(self, hidden_dim, dropout, heads_num, scale, K, J=5):
        super().__init__()

        self.J = J
        self.rho = K//2
        self.weight_function = WaveletCoefs(hidden_dim, dropout, heads_num)
        self.proj_a = nn.Linear(hidden_dim, self.rho)
        self.proj_b = nn.Linear(hidden_dim, self.rho)
        self.proj_s = nn.Linear(hidden_dim, self.J)
        self.sigmoid = nn.Sigmoid()
        self.scale = scale # register
        self.S_kernel = nn.Linear(hidden_dim, hidden_dim)
        self.M_kernels = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for j in range(self.J)])
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, eigvs, x, U):
        B, N, _ = eigvs.shape

        coefs = self.weight_function(eigvs) # add masking ? 

        a_tilde = self.proj_a(eigvs).mean(dim=-2) 
        b_tilde = self.proj_b(eigvs).mean(dim=-2)
        scale_tilde = self.sigmoid( self.proj_s(eigvs).mean(dim=-2) ) * self.scale 

        res = [generate_h(b_tilde, eigvs, self.rho)]

        
        for j in range(self.J):
            g = generate_g(a_tilde, scale_tilde[j]*eigvs, self.rho)
            res.append(g)

        # add tight-frames !!!

        




    