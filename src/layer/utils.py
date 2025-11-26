import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim=None, hidden_num=0):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = out_dim

        self.ffn = [nn.Linear(inp_dim, hidden_dim), nn.GELU()]

        for _ in range(hidden_num):
            self.ffn.append(nn.Linear(hidden_dim, hidden_dim))
            self.ffn.append(nn.GELU())

        self.ffn.append(nn.Linear(hidden_dim, out_dim))
        self.ffn = nn.Sequential(*self.ffn)

    def forward(self, x):
        return self.ffn(x)



