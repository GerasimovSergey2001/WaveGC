from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
from ..transforms.spectral import WaveGCSpectralTransform


class AmazonComputerDataset(Amazon):
    def __init__(self, root):
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            # Short-range settings
            WaveGCSpectralTransform(mode='short', top_k_pct=0.30, threshold=0.1)
        ])
        super().__init__(root=root, name='Computers', transform=transform)

    # Compatibility with your repo's BaseDataset style
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data  # The collate_fn will handle the unpacking
