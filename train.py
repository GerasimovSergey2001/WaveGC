import warnings
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

# CHANGE 1: Point config_path to the root 'configs' folder
@hydra.main(version_base=None, config_path="src/configs", config_name="train_config")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    # Ensure src.utils.init_utils exists, otherwise use a simple logger
    try:
        logger = setup_saving_and_logging(config)
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
    
    # Instantiate Writer (Tensorboard/WandB)
    writer = instantiate(config.writer)( logger=logger, project_config=project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # 1. Setup DataLoaders (Uses your dense collate_fn automatically)
    dataloaders, batch_transforms = get_dataloaders(config, device)
    inp_dim = dataloaders['train'].dataset.x.shape[1]
    out_dim = dataloaders['train'].dataset.y.unique().shape[0]
    eigvs_dim = dataloaders['train'].dataset.eigvs.shape[1]
    
    # 2. Build Model (WaveGCNet)
    try:
        model = instantiate(config.model)(inp_dim=inp_dim, out_dim=out_dim, eigvs_dim=eigvs_dim).to(device)
    except:
        model = instantiate(config.model)(inp_dim=inp_dim, out_dim=out_dim).to(device)

    logger.info(f"Model: {config.model._target_}")

    # 3. Setup Loss and Metrics
    loss_function = instantiate(config.loss_function).to(device)

    # CHANGE 2: Manually instantiate metrics to ensure correct dict structure
    # This prevents errors if Hydra doesn't automatically parse the list structure
    metrics = {
        "train": [instantiate(m) for m in config.metrics.train],
        "inference": [instantiate(m) for m in config.metrics.inference]
    }

    # 4. Build Optimizer & Scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer)(params=trainable_params)
    
    lr_scheduler = None
    if "lr_scheduler" in config and config.lr_scheduler is not None:
        lr_scheduler = instantiate(config.lr_scheduler)(optimizer=optimizer)

    # 5. Initialize Trainer
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    # 6. Run Training
    trainer.train()

if __name__ == "__main__":
    main()