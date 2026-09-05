import os
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def load_warm_start_checkpoint(model, path):
    """Load model weights while expanding the categorical K embedding.

    All parameters except ``k_emb.weight`` must have exactly the same shape.
    Existing K rows are copied verbatim and new rows start from the last
    learned K embedding. Optimizer, scheduler, histories and epoch counters
    are deliberately not restored.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    embedding_key = "k_emb.weight"

    if embedding_key not in source_state or embedding_key not in target_state:
        raise KeyError(f"Missing {embedding_key!r} in warm-start checkpoint or model")

    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise ValueError(
            "Warm-start architecture mismatch: "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )

    for name, source_value in source_state.items():
        if name == embedding_key:
            continue
        if source_value.shape != target_state[name].shape:
            raise ValueError(
                f"Warm-start shape mismatch for {name}: "
                f"checkpoint={tuple(source_value.shape)}, "
                f"model={tuple(target_state[name].shape)}"
            )

    source_embedding = source_state[embedding_key]
    target_embedding = target_state[embedding_key].clone()
    if (
        source_embedding.ndim != 2
        or source_embedding.shape[1] != target_embedding.shape[1]
    ):
        raise ValueError(
            f"Incompatible {embedding_key} shapes: "
            f"checkpoint={tuple(source_embedding.shape)}, "
            f"model={tuple(target_embedding.shape)}"
        )
    if source_embedding.shape[0] > target_embedding.shape[0]:
        raise ValueError(
            "Warm-start cannot shrink k_emb: "
            f"checkpoint rows={source_embedding.shape[0]}, "
            f"model rows={target_embedding.shape[0]}"
        )

    learned_rows = source_embedding.shape[0]
    target_embedding[:learned_rows] = source_embedding
    if learned_rows < target_embedding.shape[0]:
        target_embedding[learned_rows:] = source_embedding[-1]

    adapted_state = dict(source_state)
    adapted_state[embedding_key] = target_embedding
    model.load_state_dict(adapted_state, strict=True)

    checkpoint_epoch = checkpoint.get("epoch")
    return {
        "checkpoint_epoch": None if checkpoint_epoch is None else checkpoint_epoch + 1,
        "source_max_k": learned_rows - 1,
        "target_max_k": target_embedding.shape[0] - 1,
        "new_k_initialized_from": learned_rows - 1,
    }


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
