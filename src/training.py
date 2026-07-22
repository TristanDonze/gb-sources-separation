import logging

import torch

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

def train_one_epoch(model, dataloader, optimizer, criterion, device, timing=True):
    model.train()
    total_loss = 0.0
    total_source_loss = 0.0
    total_residual_loss = 0.0
    timer = StageTimer(device, logger, prefix="Training", enabled=timing)

    for mixture, X_1, K, snrs in timer.iter_batches(dataloader):
        B, L, _, _ = X_1.shape

        with timer.measure("to_device"):
            mixture = mixture.to(device)
            X_1 = X_1.to(device)
            K = K.to(device)
            snrs = snrs.to(device)

        with timer.measure("prepare"):
            X_0 = build_initial_state(mixture, X_1)

            t = torch.rand(B, device=mixture.device)
            t_view = t[:, None, None, None]

            X_t = (1 - t_view) * X_0 + t_view * X_1

        with timer.measure("model_forward"):
            v = model(X_t, t, mixture, K)

        with timer.measure("loss_backward_step"):
            loss, source_loss, residual_loss = criterion(v, X_0, X_1, snrs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        total_source_loss += source_loss.item()
        total_residual_loss += residual_loss.item()

    timer.log()

    avg_loss = total_loss / len(dataloader)
    avg_source_loss = total_source_loss / len(dataloader)
    avg_residual_loss = total_residual_loss / len(dataloader)
    
    return avg_loss, avg_source_loss, avg_residual_loss


if __name__ == "__main__":
    from config import small_dataset_path
    from src.model import FlowSeparator
    from src.dataset import create_train_val_datasets
    from torch.utils.data import DataLoader
    from src.losses import FlowMatchingPIT_MSELoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device : {device}")

    train_dataset, val_dataset = create_train_val_datasets(
        dataset_path=small_dataset_path,
        train_size=0.8,
        max_K=10,
        constant_K=2,
        max_samples_train=None,
        max_samples_val=10_000,
        noise_train=True,
        noise_val=True,
        # random_global_scale_train=True,
        deterministic_train=False,
        deterministic_val=True,
        # target_energy_val=22000.0,
        return_params=False,
        returns_snr=True,
        split_seed=42,
        split_strategy="snr",
        snr_bin_width=1.0,
        seed_train=42,
        seed_val=0 
    )

    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    criterion = FlowMatchingPIT_MSELoss()
    model = FlowSeparator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_one_epoch(model, dataloader, optimizer, criterion, device)
