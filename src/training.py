import logging

import torch

from src.masking import mask_slots
from src.timing import StageTimer

logger = logging.getLogger(__name__)


def build_initial_state(mixture, X_1, K, slot_mask, generator=None):
    B, L, _, _ = X_1.shape

    active_count = (K + 1).to(mixture.dtype)
    S_bar = mixture[:, None, :, :] / active_count[:, None, None, None]
    S_bar = S_bar.repeat(1, L, 1, 1)
    S_bar = mask_slots(S_bar, slot_mask)

    Z = torch.randn(
        X_1.shape,
        device=X_1.device,
        dtype=X_1.dtype,
        generator=generator,
    )
    Z = mask_slots(Z, slot_mask)
    Z_mean = Z.sum(dim=1, keepdim=True) / active_count[:, None, None, None]
    Z = mask_slots(Z - Z_mean, slot_mask)

    return S_bar + Z

def train_one_epoch(model, dataloader, optimizer, criterion, device, timing=True):
    model.train()
    total_loss = 0.0
    total_source_mse = 0.0
    total_residual_mse = 0.0
    timer = StageTimer(device, logger, prefix="Training", enabled=timing)

    total_samples = 0
    total_sources = 0

    for mixture, X_1, K, slot_mask in timer.iter_batches(dataloader):
        B, L, _, _ = X_1.shape

        with timer.measure("to_device"):
            mixture = mixture.to(device)
            X_1 = X_1.to(device)
            K = K.to(device)
            slot_mask = slot_mask.to(device)

        with timer.measure("prepare"):
            X_0 = build_initial_state(mixture, X_1, K, slot_mask)

        with timer.measure("pet_assignment"):
            model.eval()
            with torch.no_grad():
                t_0 = torch.zeros(B, device=mixture.device, dtype=mixture.dtype)
                v_0 = model(X_0, t_0, mixture, K, slot_mask)
                assignment = criterion.find_assignment(v_0, X_0, X_1, slot_mask)
                aligned_X_1 = criterion.align_targets(X_1, assignment, slot_mask)
            model.train()

        with timer.measure("prepare"):
            t = torch.rand(B, device=mixture.device)
            t_view = t[:, None, None, None]

            X_t = (1 - t_view) * X_0 + t_view * aligned_X_1

        with timer.measure("model_forward"):
            v = model(X_t, t, mixture, K, slot_mask)

        with timer.measure("loss_backward_step"):
            loss, source_mse, residual_mse = criterion(
                v,
                X_0,
                X_1,
                assignment,
                slot_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        source_count = int(K.sum().item())
        total_loss += loss_value * B
        total_source_mse += source_mse.item() * source_count
        total_residual_mse += residual_mse.item() * B
        total_samples += B
        total_sources += source_count

    timer.log()

    avg_loss = total_loss / total_samples
    avg_source_mse = total_source_mse / total_sources
    avg_residual_mse = total_residual_mse / total_samples
    
    return avg_loss, avg_source_mse, avg_residual_mse


if __name__ == "__main__":
    from config import small_dataset_path
    from src.model import FlowSeparator
    from src.dataset import create_train_val_datasets
    from torch.utils.data import DataLoader
    from src.losses import FlowMatchingPET_DBNormalizedLoss

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
        split_seed=42,
        split_strategy="snr",
        snr_bin_width=1.0,
        seed_train=42,
        seed_val=0 
    )

    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    criterion = FlowMatchingPET_DBNormalizedLoss()
    model = FlowSeparator(max_k=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_one_epoch(model, dataloader, optimizer, criterion, device)
