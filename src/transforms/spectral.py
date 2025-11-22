import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj


class WaveGCSpectralTransform:
    def __init__(self, mode='long', top_k_pct=1.0, threshold=0.0):
        """
        Computes Eigendecomposition for WaveGC.

        Args:
            mode (str): 'short' or 'long'.
            top_k_pct (float): Percentage of eigenvalues to keep.
                               Paper uses 0.30 for short-range.
            threshold (float): Threshold for sparsifying U.
                               Paper uses 0.1 for short-range.
        """
        self.mode = mode
        self.top_k_pct = top_k_pct
        self.threshold = threshold

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes

        # 1. Compute Normalized Laplacian: L = I - D^-0.5 A D^-0.5
        assert data.edge_index is not None, "Data object must have edge_index"
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            edge_weight=data.edge_attr,
            normalization='sym',
            num_nodes=num_nodes
        )

        # Convert to dense for eigendecomposition (Complexity O(N^3))
        # For large graphs like ogbn-arxiv, you would need scipy.sparse.linalg here.
        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze(0)

        # 2. Eigendecomposition
        # eigh returns eigenvalues in ascending order
        eigvs, U = torch.linalg.eigh(L)
        assert U is not None, "Eigendecomposition failed"

        # Clamp to theoretical bounds [0, 2] to fix numerical precision issues
        eigvs = torch.clamp(eigvs, 0.0, 2.0)

        # --- NUANCE: Spectral Truncation (Short-Range) ---
        if self.mode == 'short':
            # Keep only the lowest 'k' frequencies (smooth signals)
            assert isinstance(num_nodes, int), "num_nodes must be an integer"
            k = int(num_nodes * self.top_k_pct)
            if k < 2:
                k = 2  # Safety floor

            eigvs = eigvs[:k]
            U = U[:, :k]

        # --- NUANCE: Hard Thresholding (Short-Range) ---
        if self.threshold > 0:
            mask = torch.abs(U) >= self.threshold
            U = U * mask

        # 3. Store in Data object
        data.eigvs = eigvs
        data.U = U

        return data
