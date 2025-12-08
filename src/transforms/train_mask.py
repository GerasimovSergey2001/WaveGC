import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional

class TrainTestSplit:

    def __init__(self, train_mask:Optional[Tensor] = None, test_size:float = 0.2, seed:int = 42):
        self.train_mask = train_mask
        self.seed = seed
        self.test_size = test_size
    
    def __call__(self, data: Data) -> Data:
        data.generate_ids()
        if self.train_mask is None:
            data.train_mask = torch.rand(len(data.n_id))<(1-self.test_size)
        else: data.train_mask = self.train_mask
        data.test_mask = ~data.train_mask 
        return data