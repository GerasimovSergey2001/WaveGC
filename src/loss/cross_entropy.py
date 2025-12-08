import torch
from torch import nn
from typing import Optional

class MaskedCrossEntropyLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None, **batch):
        """
        Loss function calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
            mask (Tensor|None): label mask.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(logits[mask], labels[mask]) if mask is not None else self.loss(logits, labels)}
