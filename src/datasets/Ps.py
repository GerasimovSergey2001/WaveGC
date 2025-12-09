import torch
import os
import os.path as osp
from torch_geometric.datasets import LRGBDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from ..transforms.spectral import WaveGCSpectralTransform

class PeptidesStructDataset(LRGBDataset):
    def __init__(self, root):
        # Define Transform
        pre_transform = T.Compose([
            WaveGCSpectralTransform(mode='long', top_k_pct=1.0)
        ])
        
        super().__init__(
            root=root, 
            name='Peptides-struct', 
            pre_transform=pre_transform
        )

    @property
    def raw_file_names(self):
        # We now look specifically for the file you downloaded
        return ['geometric_data_processed.pt']

    def process(self):
        print(f"Processing {self.name} from {self.raw_file_names[0]}...")
        raw_path = self.raw_paths[0]
        
        # 1. Load the file
        raw_content = torch.load(raw_path)
        
        # 2. Unpack Data
        # Case A: It's already a list of Data objects
        if isinstance(raw_content, list):
            data_list = raw_content
            print("Loaded raw data as a list.")
            
        # Case B: It's a tuple (data, slices) - Standard PyG processed format
        elif isinstance(raw_content, tuple) and len(raw_content) >= 2:
            print("Detected compressed (data, slices) format. Unpacking...")
            data, slices = raw_content
            data_list = []
            
            # Reconstruct individual graphs using slices
            # We iterate through the number of graphs (slices['x'] has N+1 entries)
            num_graphs = len(slices['x']) - 1
            for i in range(num_graphs):
                item = Data()
                # For each attribute (x, edge_index, y, etc.), slice it
                for key in data.keys():
                    if key in slices:
                        start, end = slices[key][i], slices[key][i+1]
                        # Determine dimension to slice
                        # edge_index is usually sliced on dim 1, others on dim 0
                        if key == 'edge_index':
                            item[key] = data[key][:, start:end]
                        else:
                            item[key] = data[key][start:end]
                    else:
                        # Global attributes (same for all)
                        item[key] = data[key]
                data_list.append(item)
            print(f"Successfully unpacked {len(data_list)} graphs.")

        else:
            raise RuntimeError(f"Unknown file format: {type(raw_content)}. Expected List or (Data, Slices) Tuple.")

        # 2.5 Clean up edge attributes and ensure proper format
        for data in data_list:
            # Remove edge_attr completely to avoid dimension mismatches
            if hasattr(data, 'edge_attr'):
                delattr(data, 'edge_attr')
            # Ensure edge_index is contiguous and long type
            if hasattr(data, 'edge_index'):
                data.edge_index = data.edge_index.long().contiguous()

        # 3. Apply WaveGC Transform (Compute Eigenvectors)
        if self.pre_transform is not None:
            # Check if first graph already has 'eigvs' to avoid re-computing if not needed
            if hasattr(data_list[0], 'eigvs') and data_list[0].eigvs is not None:
                print("Data already contains spectral features. Skipping transform.")
            else:
                print("Applying Spectral Transform (Calculating Eigenvectors)...")
                # This will take time (~5-10 mins)
                data_list = [self.pre_transform(data) for data in data_list]

        # 4. Save as List (Safe Format)
        print("Saving processed data to disk...")
        torch.save(data_list, self.processed_paths[0])

    def load(self, path):
        self._data_list = torch.load(path)
    
    def len(self):
        return len(self._data_list)
    
    def get(self, idx):
        return self._data_list[idx]