import os
import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    train_mses,
    val_mses,
    val_nmses,
    val_mixture_consistency,
    best_val_nmse,
    best_val_nmse_epoch,
    epoch,
    path,
    min_lr_reached_epoch=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_mses": train_mses,
        "val_mses": val_mses,
        "val_nmses": val_nmses,
        "val_mixture_consistency": val_mixture_consistency,
        "best_val_nmse": best_val_nmse,
        "best_val_nmse_epoch": best_val_nmse_epoch,
        "min_lr_reached_epoch": min_lr_reached_epoch,
        "epoch": epoch
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    train_mses = checkpoint["train_mses"]
    val_mses = checkpoint["val_mses"]
    val_nmses = checkpoint["val_nmses"]
    val_mixture_consistency = checkpoint["val_mixture_consistency"]
    best_val_nmse = checkpoint["best_val_nmse"]
    best_val_nmse_epoch = checkpoint["best_val_nmse_epoch"]
    epoch = checkpoint["epoch"]
    min_lr_reached_epoch = checkpoint.get("min_lr_reached_epoch")
    return (
        train_mses,
        val_mses,
        val_nmses,
        val_mixture_consistency,
        best_val_nmse,
        best_val_nmse_epoch,
        epoch,
        min_lr_reached_epoch,
    )