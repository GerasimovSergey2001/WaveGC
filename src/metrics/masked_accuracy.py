import torch
import numpy as np
from typing import Optional
from src.metrics.base_metric import BaseMetric


class MaskedAccuracy(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
            mask (Tensor|None): loss mask.
        Returns:
            metric (float): calculated metric.
        """
        classes = logits.argmax(dim=-1).to(torch.long)
        
        if mask is not None:
            pred = classes[mask]
            true = labels[mask]
        else:
            pred = classes
            true = labels
            
        device = self.metric.device
        
        pred = pred.to(device)
        true = true.to(device)
        
        acc = self.metric(pred, true)
        return acc
