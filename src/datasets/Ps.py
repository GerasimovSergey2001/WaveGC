import torch
from torch_geometric.datasets import LRGBDataset
import torch_geometric.transforms as T
from ..transforms.spectral import WaveGCSpectralTransform

class PeptidesStructDataset(LRGBDataset):
    def __init__(self, root):
        # 1. Transform: Full Spectrum, Variable Sizes (No padding needed here)
        pre_transform = T.Compose([
            WaveGCSpectralTransform(mode='long', top_k_pct=1.0) 
        ])
        
        # 2. Initialize
        super().__init__(
            root=root, 
            name='Peptides-struct', 
            pre_transform=pre_transform
        )

    # --- OVERRIDES TO ENABLE LIST STORAGE (No Concatenation) ---

    def process(self):
        # This overrides the default LRGB process which forces concatenation
        import os.path as osp
        from torch_geometric.data import download_url, extract_zip
        import pickle

        # Ensure raw data exists (handled by download(), but good to check)
        
        # Load raw data
        with open(self.raw_paths[0], 'rb') as f:
            data_list = pickle.load(f)

        if self.pre_transform is not None:
            print("Transforming data... (This may take a while)")
            data_list = [self.pre_transform(data) for data in data_list]

        # SAVE AS A LIST (Avoids 'Sizes of tensors must match' error)
        print("Saving processed data as a list...")
        torch.save(data_list, self.processed_paths[0])

    def load(self, path):
        # Load the list directly into memory
        self._data_list = torch.load(path)
    
    def len(self):
        return len(self._data_list)
    
    def get(self, idx):
        return self._data_list[idx]