import os
import torch

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
    val_losses,
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
        "val_losses": val_losses,
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
    val_losses = checkpoint["val_losses"]
    best_val_loss = checkpoint["best_val_loss"]
    best_val_loss_epoch = checkpoint["best_val_loss_epoch"]
    epoch = checkpoint["epoch"]
    min_lr_reached_epoch = checkpoint.get("min_lr_reached_epoch")
    return (
        train_losses,
        val_losses,
        best_val_loss,
        best_val_loss_epoch,
        epoch,
        min_lr_reached_epoch,
    )