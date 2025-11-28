import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor

from src.layer.waveconv import WaveGC
from src.layer.utils import FFN
from src.layer.encoders import DataProcessing
from typing import *


class WaveGCNet(nn.Module):
    def __init__(
        self,
        inp_dim, 
        out_dim,
        emb_dim, 
        pe_dim, 
        eigvs_dim,
        lape_hidden_num,
        hidden_dim, 
        heads_num, 
        scale, 
        num_layers,
        mpnn="gcn",
        K=6,
        J=5,
        tight_frames=True,
        dropout=0,
        ffn_hidden_num=2,
        ffn_hidden_dim=None,
        mpnn_hidden_num=1,
        eps=100,
        aggr="max",
    ):
        super().__init__()

        if ffn_hidden_dim is None: 
            ffn_hidden_dim = emb_dim

        self.processing = DataProcessing(
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
                dropout=dropout, 
                eps=eps, 
                hidden_num=0
        )

        self.conv_layer = nn.ModuleList(
            [
                WaveGC(
                    emb_dim,
                    emb_dim,
                    mpnn,
                    K,
                    J,
                    tight_frames,
                    dropout,
                    ffn_hidden_num,
                    mpnn_hidden_num,
                    aggr,
                )
            ]
        )

        for _ in range(num_layers):
            self.conv_layer.append(
                WaveGC(
                    emb_dim,
                    emb_dim,
                    mpnn,
                    K,
                    J,
                    tight_frames,
                    dropout,
                    ffn_hidden_num,
                    mpnn_hidden_num,
                    aggr,
                )
            )

        self.proj_out = FFN(emb_dim, out_dim, hidden_dim, ffn_hidden_num)

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        eigvs: Tensor,
        U: Tensor,
        eigvs_mask: Optional[Tensor] = None,
        **kwargs,
    ):

        x, a_tilde, b_tilde, scale_tilde = self.processing(x, eigvs, U, eigvs_mask)
        for i in range(len(self.conv_layer)):
            x = self.conv_layer[i](x, edge_index, eigvs, U, a_tilde, b_tilde, scale_tilde, **kwargs)
        return self.proj_out(x)