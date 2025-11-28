import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch


def collate_fn(dataset_items: list):
    """
    Pads spatial dimensions (N) to N_max.
    Stacks spectral dimensions (k) based on the computed pre-transform.
    """
    batch_size = len(dataset_items)

    # 1. Determine Dimensions
    # Spatial: Max nodes in this batch
    max_nodes = max([item.num_nodes for item in dataset_items])

    # Spectral: Max eigenvalues available in this batch
    # (If using fixed top_k in transform, this is constant. If pct, it varies)
    max_k = max([item.eigvs.shape[0] for item in dataset_items])

    feat_dim = dataset_items[0].x.shape[1]

    # 2. Allocate Dense Tensors
    x_padded = torch.zeros((batch_size, max_nodes, feat_dim))
    eigvs_padded = torch.zeros((batch_size, max_k))
    U_padded = torch.zeros((batch_size, max_nodes, max_k))

    # Mask: True = Padding/Ignored
    eigvs_mask = torch.ones((batch_size, max_k), dtype=torch.bool)

    labels = []

    for i, data in enumerate(dataset_items):
        num_n = data.num_nodes
        num_k = data.eigvs.shape[0]

        # Spatial Features
        x_padded[i, :num_n, :] = data.x

        # Spectral Features
        eigvs_padded[i, :num_k] = data.eigvs

        # Eigenvectors (Map Spatial N to Spectral K)
        U_padded[i, :num_n, :num_k] = data.U

        # Valid Mask (False = Real Data)
        eigvs_mask[i, :num_k] = False

        labels.append(data.y)
    
    pyg_batch = Batch.from_data_list(dataset_items)
    sparse_edge_index = pyg_batch.edge_index  # type: ignore[attr-defined]

    return {
        "x": x_padded,
        "eigvs": eigvs_padded,
        "U": U_padded,
        "eigvs_mask": eigvs_mask,
        "labels": torch.stack(labels),
        "edge_index": sparse_edge_index,
        "batch_idx": pyg_batch.batch  # type: ignore[attr-defined]

    }
