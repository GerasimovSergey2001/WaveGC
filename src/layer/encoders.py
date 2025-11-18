import torch
import torch.nn as nn

class TrigonometricEncoder(nn.Module):

    def __init__(self, hidden_dim, eps, base=10_000, project=True):
        super().__init__()
        self.d = hidden_dim
        self.eps = eps
        powers = 2/self.d*torch.arange(0, self.d , 2, dtype=torch.float32)
        base = torch.tensor([base], dtype=torch.float32)
        self.den = torch.exp(-torch.log(base)*powers)
        self.proj =  nn.Linear(self.d + 1, self.d)
        self.project = project
        self.register_buffer('denominator', self.den)

    def forward(self, eigvs: torch.tensor):
        """
        eigvs: [B, N]
        """
        x = self.eps*eigvs[:,:,None]*self.den

        if self.project:
            x = torch.cat([eigvs[:,:,None], torch.sin(x), torch.cos(x)], axis=2)
            return self.proj(x) 
        else:
            return eigvs[:,:,None] + torch.cat([torch.sin(x), torch.cos(x)], axis=2)
    
        