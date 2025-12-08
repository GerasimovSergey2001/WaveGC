from itertools import repeat
import torch
from hydra.utils import instantiate
import importlib
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms_cfg = config.transforms.get('batch_transforms')
    if batch_transforms_cfg is not None:
        batch_transforms = instantiate(batch_transforms_cfg)
        move_batch_transforms_to_device(batch_transforms, device)
    else:
        batch_transforms = None
    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside
    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition][0]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )
        partition_dataloader = instantiate(config.dataloader)(
            dataset=dataset,
            collate_fn=collate_fn,
            #drop_last=(dataset_partition == "train"),
            #shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader
    dataloaders['test'] = dataloaders['train']
    return dataloaders, batch_transforms

def allowlist_pyg_for_torch_load():
    """Allowlist common PyG classes used in pickled datasets (PyTorch 2.6+)."""
    candidate_paths = [
        "torch_geometric.data.data.Data",
        "torch_geometric.data.data.DataTensorAttr",
        "torch_geometric.data.data.DataEdgeAttr",
        "torch_geometric.data.storage.BaseStorage",
        "torch_geometric.data.storage.NodeStorage",
        "torch_geometric.data.storage.EdgeStorage",
        "torch_geometric.data.storage.GlobalStorage",
    ]
    allowed = []
    for dotted in candidate_paths:
        try:
            module_path, name = dotted.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            allowed.append(getattr(mod, name))
        except Exception:
            pass
    if allowed:
        torch.serialization.add_safe_globals(allowed)
