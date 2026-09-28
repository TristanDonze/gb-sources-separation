import logging

import torch
from config import NB_OF_STEPS

from src.masking import mask_slots
from src.timing import StageTimer
from src.training import build_initial_state

logger = logging.getLogger(__name__)


def evaluate(model, dataloader, criteria, device, timing=True):
    model.eval()

    total_loss = 0.0
    total_source_loss = 0.0
    total_residual_loss = 0.0
    total_samples = 0

    total_nmse_source = 0.0
    total_nmse_source_db = 0.0
    total_sources = 0
    metrics_by_k = {}

    time_grid = 0.5 * (
        1 - torch.cos(torch.linspace(0.0, torch.pi, NB_OF_STEPS + 1))
    ) # scheduler's euler solver
    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    timer = StageTimer(device, logger, prefix="Validation", enabled=timing)

    with torch.no_grad():
        for mixture, X_1, K, slot_mask in timer.iter_batches(dataloader):
            with timer.measure("to_device"):
                mixture = mixture.to(device)
                X_1 = X_1.to(device)
                K = K.to(device)
                slot_mask = slot_mask.to(device)

            B = X_1.shape[0]

            with timer.measure("initial_state"):
                X_t = build_initial_state(
                    mixture, X_1, K, slot_mask, generator=generator
                )

            for step in range(NB_OF_STEPS):
                t_val = time_grid[step].item()
                dt = (time_grid[step + 1] - time_grid[step]).item()
                t = torch.full(
                    (B,),
                    t_val,
                    device=device,
                    dtype=mixture.dtype,
                )

                with timer.measure("model_forward"):
                    v = model(X_t, t, mixture, K, slot_mask)

                X_t = mask_slots(X_t + dt * v, slot_mask)

            X_hat = X_t

            for k_tensor in torch.unique(K):
                k = int(k_tensor.item())
                batch_mask = K == k
                count = int(batch_mask.sum().item())
                subset_slot_mask = slot_mask[batch_mask]

                with timer.measure("criterion_1"):
                    loss, source_loss, residual_loss = criteria[0](
                        X_hat[batch_mask], X_1[batch_mask], subset_slot_mask
                    )
                with timer.measure("criterion_2"):
                    nmse_source, nmse_source_db = criteria[1](
                        X_hat[batch_mask, :-1],
                        X_1[batch_mask, :-1],
                        subset_slot_mask[:, :-1],
                    )

                entry = metrics_by_k.setdefault(
                    k,
                    {"count": 0, "loss": 0.0, "source_mse": 0.0,
                     "residual_mse": 0.0, "source_nmse": 0.0,
                     "source_nmse_db": 0.0},
                )
                entry["count"] += count
                entry["loss"] += loss.item() * count
                entry["source_mse"] += source_loss.item() * count
                entry["residual_mse"] += residual_loss.item() * count
                entry["source_nmse"] += nmse_source.item() * count
                entry["source_nmse_db"] += nmse_source_db.item() * count

                source_count = count * k
                total_source_loss += source_loss.item() * source_count
                total_residual_loss += residual_loss.item() * count
                total_nmse_source += nmse_source.item() * source_count
                total_nmse_source_db += nmse_source_db.item() * source_count
                total_sources += source_count
            total_samples += B

    average_source_loss = total_source_loss / total_sources
    average_residual_loss = total_residual_loss / total_samples
    average_loss = average_source_loss + criteria[0].residual_weight * average_residual_loss
    average_nmse_source = total_nmse_source / total_sources
    average_nmse_source_db = total_nmse_source_db / total_sources
    for values in metrics_by_k.values():
        count = values["count"]
        for name in ("loss", "source_mse", "residual_mse", "source_nmse", "source_nmse_db"):
            values[name] /= count
    timer.log(
        renamed_averages={"model_forward": "model_forward_per_call"},
        extra_averages={
            "model_forward_per_batch": ("model_forward", len(dataloader)),
        },
    )

    return (
        average_loss,
        average_source_loss,
        average_residual_loss,
        average_nmse_source,
        average_nmse_source_db,
        metrics_by_k,
    )
