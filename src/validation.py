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


def evaluate(model, dataloader, criterions, device, timing=True):
    model.eval()

    totals = {
        name: {
            "source_numerator": 0.0,
            "source_denominator": 0.0,
            "residual_loss": 0.0,
        }
        for name in criterions
    }
    total_samples = 0

    dt = 1.0 / NB_OF_STEPS
    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    timer = StageTimer(device, logger, prefix="Validation", enabled=timing)

    with torch.no_grad():
        for mixture, X_1, K, snrs in timer.iter_batches(dataloader):
            with timer.measure("to_device"):
                mixture = mixture.to(device)
                X_1 = X_1.to(device)
                K = K.to(device)
                snrs = snrs.to(device)

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

            with timer.measure("criterion"):
                reconstruction_diff = (
                    X_hat[:, :-1, None] - X_1[:, None, :-1]
                )
                assignment_cost = reconstruction_diff.pow(2).mean(dim=(-1, -2))

                source_count = B * snrs.shape[1]
                for name, criterion in criterions.items():
                    _, source_loss, residual_loss = criterion(
                        X_hat,
                        X_1,
                        snrs,
                        assignment_cost=assignment_cost,
                    )

                    if hasattr(criterion, "alpha"):
                        weight_mean = (
                            snrs.clamp_min(1e-8)
                            .pow(-criterion.alpha)
                            .mean()
                            .item()
                        )
                    else:
                        weight_mean = 1.0

                    totals[name]["source_numerator"] += (
                        source_loss.item() * weight_mean * source_count
                    )
                    totals[name]["source_denominator"] += (
                        weight_mean * source_count
                    )
                    totals[name]["residual_loss"] += residual_loss.item() * B

            total_samples += B

    metrics = {}
    for name, criterion in criterions.items():
        average_source_loss = (
            totals[name]["source_numerator"]
            / totals[name]["source_denominator"]
        )
        average_residual_loss = totals[name]["residual_loss"] / total_samples
        average_loss = (
            average_source_loss
            + criterion.residual_weight * average_residual_loss
        )
        metrics[name] = {
            "loss": average_loss,
            "source_loss": average_source_loss,
            "residual_loss": average_residual_loss,
        }

    timer.log(
        renamed_averages={"model_forward": "model_forward_per_call"},
        extra_averages={
            "model_forward_per_batch": ("model_forward", len(dataloader)),
        },
    )

    return metrics
