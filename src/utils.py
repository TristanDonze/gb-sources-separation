import os
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "rng"):
        worker_info.dataset.rng = np.random.default_rng(worker_seed)


def get_device():
    free_memory = []

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_memory.append(free)

    best_gpu = free_memory.index(max(free_memory))

    device = torch.device(f"cuda:{best_gpu}")
    return device

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    train_losses,
    train_source_losses,
    train_residual_losses,
    val_losses,
    val_source_losses,
    val_residual_losses,
    val_nmse_sources,
    val_nmse_source_dbs,
    val_metrics_by_k,
    best_val_loss,
    best_val_loss_epoch,
    epoch,
    path,
    min_lr_reached_epoch=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "train_source_losses": train_source_losses,
        "train_residual_losses": train_residual_losses,
        "val_losses": val_losses,
        "val_source_losses": val_source_losses,
        "val_residual_losses": val_residual_losses,
        "val_nmse_sources": val_nmse_sources,
        "val_nmse_source_dbs": val_nmse_source_dbs,
        "val_metrics_by_k": val_metrics_by_k,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "min_lr_reached_epoch": min_lr_reached_epoch,
        "epoch": epoch
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    train_losses = checkpoint["train_losses"]
    train_source_losses = checkpoint["train_source_losses"]
    train_residual_losses = checkpoint["train_residual_losses"]
    val_losses = checkpoint["val_losses"]
    val_source_losses = checkpoint["val_source_losses"]
    val_residual_losses = checkpoint["val_residual_losses"]
    val_nmse_sources = checkpoint.get("val_nmse_sources", [])
    val_nmse_source_dbs = checkpoint.get("val_nmse_source_dbs", [])
    val_metrics_by_k = checkpoint.get("val_metrics_by_k", [])
    best_val_loss = checkpoint["best_val_loss"]
    best_val_loss_epoch = checkpoint["best_val_loss_epoch"]
    epoch = checkpoint["epoch"]
    min_lr_reached_epoch = checkpoint.get("min_lr_reached_epoch")
    return (
        train_losses,
        train_source_losses,
        train_residual_losses,
        val_losses,
        val_source_losses,
        val_residual_losses,
        val_nmse_sources,
        val_nmse_source_dbs,
        val_metrics_by_k,
        best_val_loss,
        best_val_loss_epoch,
        epoch,
        min_lr_reached_epoch,
    )
