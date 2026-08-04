import logging

import torch
from config import NB_OF_STEPS

from src.timing import StageTimer

logger = logging.getLogger(__name__)


def build_initial_state(mixture, X_1, generator=None):
    B, L, _, _ = X_1.shape

    S_bar = mixture[:, None, :, :] / L
    S_bar = S_bar.repeat(1, L, 1, 1)

    Z = torch.randn(
        X_1.shape,
        device=X_1.device,
        dtype=X_1.dtype,
        generator=generator,
    )
    Z = Z - Z.mean(dim=1, keepdim=True)

    return S_bar + Z


def evaluate(model, dataloader, criteria, device, timing=True):
    model.eval()

    total_loss = 0.0
    total_source_loss = 0.0
    total_residual_loss = 0.0
    total_samples = 0

    total_nmse_source = 0.0
    total_nmse_source_db = 0.0

    dt = 1.0 / NB_OF_STEPS
    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    timer = StageTimer(device, logger, prefix="Validation", enabled=timing)

    with torch.no_grad():
        for mixture, X_1, K in timer.iter_batches(dataloader):
            with timer.measure("to_device"):
                mixture = mixture.to(device)
                X_1 = X_1.to(device)
                K = K.to(device)

            B = X_1.shape[0]

            with timer.measure("initial_state"):
                X_t = build_initial_state(mixture, X_1, generator=generator)

            for step in range(NB_OF_STEPS):
                t_val = step * dt
                t = torch.full(
                    (B,),
                    t_val,
                    device=device,
                    dtype=mixture.dtype,
                )

                with timer.measure("model_forward"):
                    v = model(X_t, t, mixture, K)

                X_t = X_t + dt * v

            X_hat = X_t

            # Classic Reconstrion PIT MSE Loss
            with timer.measure("criterion_1"):
                loss, source_loss, residual_loss = criteria[0](X_hat, X_1)

            total_loss += loss.item() * B
            total_source_loss += source_loss.item() * B
            total_residual_loss += residual_loss.item() * B

            # NMSE Loss

            with timer.measure("criterion_2"):
                pred_sources = X_hat[:, :-1]
                true_sources = X_1[:, :-1]
                nmse_source, nmse_source_db = criteria[1](pred_sources, true_sources)

            total_nmse_source += nmse_source.item() * B
            total_nmse_source_db += nmse_source_db.item() * B
            total_samples += B

    average_loss = total_loss / total_samples
    average_source_loss = total_source_loss / total_samples
    average_residual_loss = total_residual_loss / total_samples
    average_nmse_source = total_nmse_source / total_samples
    average_nmse_source_db = total_nmse_source_db / total_samples
    timer.log(
        renamed_averages={"model_forward": "model_forward_per_call"},
        extra_averages={
            "model_forward_per_batch": ("model_forward", len(dataloader)),
        },
    )

    return average_loss, average_source_loss, average_residual_loss, average_nmse_source, average_nmse_source_db
