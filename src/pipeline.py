import logging
import torch
from torch.utils.data import DataLoader

from src.model import FlowSeparator
from src.dataset import create_train_val_datasets
from src.losses import (
    FlowMatchingPET_DBNormalizedLoss,
    ReconstructionPIT_MSELoss,
    ReconstructionPIT_NMSELoss
)
from src.training import train_one_epoch
from src.validation import evaluate
from src.utils import (
    get_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    seed_worker,
)
from src.aim_instance import aim_run, track_metric


from config import (
    dataset_path,

    TRAIN_SIZE,
    MAX_SAMPLES_TRAIN,
    MAX_SAMPLES_VAL,
    SPLIT_STRATEGY,

    FIX_ALL_SEEDS,
    SPLIT_SEED,
    SEED,
    SEED_TRAIN,
    SEED_VAL,

    MAX_K,
    CONSTANT_K,
    BATCH_SIZE,
    WEIGHT_DECAY,
    NB_EPOCHS,
    ENABLE_TIMING,
    LR,
    LR_MIN,
    FACTOR,
    PATIENCE,
)

logger = logging.getLogger(__name__)


def train(checkpoint_dir, load_checkpoint_path=None):
    if FIX_ALL_SEEDS:
        seed_everything(SEED)
        logger.info(f"All random number generators seeded with: {SEED}")
    
    device = get_device()
    logger.info(f"Using device: {device}")

    train_criterion = FlowMatchingPET_DBNormalizedLoss(residual_weight=1.0, eps=1e-8)
    val_criteria = (ReconstructionPIT_MSELoss(), ReconstructionPIT_NMSELoss())
    logger.info(f"Training criterion: {train_criterion.__class__.__name__}")
    logger.info("Source permutation: PIT at t=0, fixed along each training path")


    model = FlowSeparator(max_k=MAX_K).to(device)
    logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("Model architecture:")
    for name, module in model.named_modules():
        logger.info(f"  {name}: {module}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=FACTOR,
        patience=PATIENCE,
        min_lr=LR_MIN,
    )

    train_dataset, val_dataset = create_train_val_datasets(
        dataset_path,
        train_size=TRAIN_SIZE,
        max_K=MAX_K,
        constant_K=CONSTANT_K,
        max_samples_train=MAX_SAMPLES_TRAIN,
        max_samples_val=MAX_SAMPLES_VAL,
        noise_train=True,
        noise_val=True,
        deterministic_train=False,
        deterministic_val=True,
        split_seed=SPLIT_SEED,
        split_strategy=SPLIT_STRATEGY,
        seed_train=SEED_TRAIN,
        seed_val=SEED_VAL,
    )

    train_generator = (
        torch.Generator().manual_seed(SEED) if FIX_ALL_SEEDS else None
    )
    val_generator = (
        torch.Generator().manual_seed(SEED + 1) if FIX_ALL_SEEDS else None
    )
    worker_init_fn = seed_worker if FIX_ALL_SEEDS else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        generator=val_generator,
        worker_init_fn=worker_init_fn,
    )

    aim_run["hparams"] = {
        "fix_all_seeds": FIX_ALL_SEEDS,
        "seed": SEED if FIX_ALL_SEEDS else None,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "learning_rate_min": LR_MIN,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": FACTOR,
        "scheduler_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NB_EPOCHS,
        "train_loss": train_criterion.__class__.__name__,
        "source_permutation": "PIT at t=0",
        "max_k": MAX_K,
        "constant_k": CONSTANT_K,
    }
    aim_run["dataset"] = {
        "dataset_path": str(dataset_path),
        "train_size": TRAIN_SIZE,
        "split_seed": SPLIT_SEED,
        "train_waveforms": train_dataset.total_waveforms,
        "val_waveforms": val_dataset.total_waveforms,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    train_losses = []
    train_source_losses = []
    train_residual_losses = []
    val_losses = []
    val_source_losses = []
    val_residual_losses = []
    val_nmse_sources = []
    val_nmse_source_dbs = []
    val_metrics_by_k = []
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    min_lr_reached_epoch = None

    if load_checkpoint_path is not None:
        (
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
            last_completed_epoch,
            min_lr_reached_epoch,
        ) = load_checkpoint(model, optimizer, scheduler, load_checkpoint_path)
        start_epoch = last_completed_epoch + 1
        logger.info(f"Loaded checkpoint from {load_checkpoint_path}, starting from epoch {start_epoch+1}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, NB_EPOCHS):
        train_loss, train_source_loss, train_residual_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_criterion,
            device,
            timing=ENABLE_TIMING,
        )
        (
            val_loss,
            val_source_loss,
            val_residual_loss,
            val_nmse_source,
            val_nmse_source_db,
            epoch_metrics_by_k,
        ) = evaluate(
            model,
            val_loader,
            val_criteria,
            device,
            timing=ENABLE_TIMING,
        )


        train_losses.append(train_loss)
        train_source_losses.append(train_source_loss)
        train_residual_losses.append(train_residual_loss)
        
        val_losses.append(val_loss)
        val_source_losses.append(val_source_loss)
        val_residual_losses.append(val_residual_loss)
        val_nmse_sources.append(val_nmse_source)
        val_nmse_source_dbs.append(val_nmse_source_db)
        val_metrics_by_k.append(epoch_metrics_by_k)

        scheduler.step(val_loss)

        aim_epoch = epoch + 1
        lr_at_min = all(
            param_group["lr"] <= LR_MIN + 1e-12
            for param_group in optimizer.param_groups
        )
        if lr_at_min and min_lr_reached_epoch is None:
            min_lr_reached_epoch = aim_epoch
            logger.info(f"Minimum LR {LR_MIN:.2e} reached at epoch {aim_epoch}.")

        track_metric(
            "Training dB-Normalized Flow Matching Loss",
            train_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "Training Source Velocity MSE",
            train_source_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "Training Residual Velocity MSE",
            train_residual_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "Validation Reconstruction Loss",
            val_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "Validation Reconstruction Source Loss",
            val_source_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "Validation Reconstruction Residual Loss",
            val_residual_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "Validation Reconstruction Source NMSE",
            val_nmse_source,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "Validation Reconstruction Source NMSE (dB)",
            val_nmse_source_db,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        for k, metrics in sorted(epoch_metrics_by_k.items()):
            for metric_name, aim_name in (
                ("loss", "Validation Reconstruction Loss"),
                ("source_mse", "Validation Reconstruction Source Loss"),
                ("residual_mse", "Validation Reconstruction Residual Loss"),
                ("source_nmse", "Validation Reconstruction Source NMSE"),
                ("source_nmse_db", "Validation Reconstruction Source NMSE (dB)"),
            ):
                track_metric(
                    aim_name,
                    metrics[metric_name],
                    step=aim_epoch,
                    epoch=aim_epoch,
                    split="val",
                    granularity="epoch",
                    source_count=k,
                )
        track_metric(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            step=aim_epoch,
            epoch=aim_epoch,
            granularity="epoch",
        )

        logger.info(
            f"Epoch {epoch+1}/{NB_EPOCHS} :\n"
            f" - Train dB-Normalized Loss: {train_loss:.4f}\n"
            f" - Train Velocity MSE: Source: {train_source_loss:.4f}, Residual: {train_residual_loss:.4f}\n"
            f" - Validation Loss: {val_loss:.4f} (Source: {val_source_loss:.4f}, Residual: {val_residual_loss:.4f})\n"
            f" - Validation Source NMSE: {val_nmse_source:.4f} (dB: {val_nmse_source_db:.4f})\n"
            + "".join(
                f" - K={k}: MSE={metrics['loss']:.4f}, "
                f"NMSE={metrics['source_nmse']:.4f} "
                f"({metrics['source_nmse_db']:.4f} dB)\n"
                for k, metrics in sorted(epoch_metrics_by_k.items())
            )
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch + 1
            track_metric(
                "Best Validation Reconstruction Loss",
                best_val_loss,
                step=aim_epoch,
                epoch=aim_epoch,
                split="val",
                granularity="epoch",
            )
            save_checkpoint(
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
                f"{checkpoint_dir}/best_checkpoint.pth",
            )
        else:
            save_checkpoint(
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
                f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth",
                min_lr_reached_epoch=min_lr_reached_epoch,
            )

    logger.info(
        f"Training completed. Best Val Loss: {best_val_loss:.4f} "
        f"at epoch {best_val_loss_epoch}"
    )
    aim_run.close()
