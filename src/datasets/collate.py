import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list):
    """
    Custom collate for WaveGC.
    Pads graphs to the maximum size in the batch for Dense training.
    """

    # 1. Determine Max Nodes in this batch
    # dataset_items are typically PyG Data objects or dicts depending on your dataset class
    # Assuming they are PyG Data objects for this logic:
    max_nodes = max([item.num_nodes for item in dataset_items])
    batch_size = len(dataset_items)

    # Get feature dimension (D) and Eigenvector dimension (k)
    # Note: For short-range, k is smaller than N. For long-range, k == N.
    feat_dim = dataset_items[0].x.shape[1]

    # 2. Initialize Dense Tensors
    # x_padded: [B, N_max, D]
    x_padded = torch.zeros((batch_size, max_nodes, feat_dim))

    # eigvs_padded: [B, N_max]
    eigvs_padded = torch.zeros((batch_size, max_nodes))

    # U_padded: [B, N_max, N_max] (or [B, N_max, k] for truncated)
    # We determine the 3rd dimension based on the max 'k' in the batch
    max_k = max([item.U.shape[1] for item in dataset_items])
    U_padded = torch.zeros((batch_size, max_nodes, max_k))

    # Labels can be stacked directly
    labels_list = []

    # 3. Mask for the Transformer (True = Padding / Ignored)
    # Shape: [B, N_max]
    eigvs_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool)

    for i, data in enumerate(dataset_items):
        num_n = data.num_nodes
        k_dim = data.U.shape[1]

        # Fill Features
        x_padded[i, :num_n, :] = data.x

        # Fill Spectral Info
        eigvs_padded[i, :k_dim] = data.eigvs
        U_padded[i, :num_n, :k_dim] = data.U

        # Unmask valid nodes (False = Keep)
        # Note: In WaveletCoefs transformer, key_padding_mask expects True for ignored positions
        eigvs_mask[i, :k_dim] = False

        labels_list.append(data.y)

    result_batch = {
        "x": x_padded,
        "eigvs": eigvs_padded,
        "U": U_padded,
        "eigvs_mask": eigvs_mask,
        "labels": torch.stack(labels_list) if len(labels_list) > 0 else None
    }

    return result_batch
