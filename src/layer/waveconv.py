import torch 
import torch.nn as nn

from src.layer.wavelet_weights import WaveletCoefs, FFN

class WaveConv(nn.Module):

    def __init__(self, hidden_dim, dropout, heads_num, scale, K=6, J=5):
        super().__init__()

        self.J = J
        self.K = K
        self.rho = K//2
        self.weight_function = WaveletCoefs(hidden_dim, dropout, heads_num)
        self.proj_a = nn.Linear(hidden_dim, self.rho)
        self.proj_b = nn.Linear(hidden_dim, self.rho)
        self.proj_s = nn.Linear(hidden_dim, self.J)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('scale', scale.view(1, self.J))

        self.SM_kernels = nn.ModuleList(
            [ FFN(hidden_dim, hidden_dim) for j in range(self.J+1)]
            )
        self.proj_final = nn.Linear(hidden_dim*(self.J+1), hidden_dim)
        self.act = nn.GELU()

    def get_transformed_chebyshev(self, eigvs):
        x = eigvs - 1
        # T_0(x) = 1, T_1(x) = x
        T_list = [torch.ones_like(x), x]
        
        for k in range(2,self.K + 1):
            # T_k = 2 * x * T_{k-1} - T_{k-2}
            T_next = 2 * x * T_list[-1] - T_list[-2]
            T_list.append(T_next)
            
        # T_new = 0.5 * (-T_old + 1)
        T_transformed = [0.5 * (-t + 1) for t in T_list]
        
        return T_transformed
    
    def generate_g(self, a, scaled_eigvs):
        mask = (scaled_eigvs <= 2.0).float()
        safe_eigvs = torch.clamp(scaled_eigvs, max=2.0)

        T_even = self.get_transformed_chebyshev(safe_eigvs)[::2][1:] # T0 is always 0
        return torch.einsum('bi, bik-> bk', a, torch.stack(T_even, dim=1)) * mask

    def generate_h(self, b, eigvs):
        T_odd = self.get_transformed_chebyshev(eigvs)[1::2]
        return torch.einsum('bi, bik-> bk', b, torch.stack(T_odd, dim=1))
    

    def forward(self, x, Us, eigvs, eigvs_mask=None, tight_frames=True):

        coefs = self.weight_function(eigvs, eigvs_mask)

        a_tilde = self.proj_a(coefs).mean(dim=-2) 
        b_tilde = self.proj_b(coefs).mean(dim=-2)
        scale_tilde = self.sigmoid( self.proj_s(coefs).mean(dim=-2) ) * self.scale 

        h_g = []
        v_sq = 0

        for j in range(self.J+1):
            if j == 0: 
                h_g.append( self.generate_h(b_tilde, eigvs) )
            else: 
                h_g.append( self.generate_g(a_tilde, scale_tilde[:,j-1].view(-1,1)*eigvs) )

            v_sq += h_g[j]**2

        h_g = torch.stack(h_g) # [1+J, B, N, N]
        
        if tight_frames:
            h_g /= (torch.sqrt(v_sq) + 1e-6)
        
        U_lambda = torch.einsum('bik, jbk -> jbik', Us, h_g) 
        T = torch.einsum('jbil, blk -> jbik', U_lambda, Us.transpose(1,2))
        
        H = []
        for j in range(self.J+1):
            WSH = self.SM_kernels[j]( 
                torch.einsum('bik, bkd -> bid',T[j], x)
                )
            H.append(
                torch.einsum('bik, bkj -> bij', T[j].transpose(-1,-2), WSH)
            )

        H = torch.cat(H, dim=-1)

        return self.act(self.proj_final(H))

        




    