import torch
from config import NB_OF_STEPS


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


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    dt = 1.0 / NB_OF_STEPS
    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    with torch.no_grad():
        for batch_idx, (mixture, X_1, K) in enumerate(dataloader):
            mixture = mixture.to(device)
            X_1 = X_1.to(device)
            K = K.to(device)

            B = X_1.shape[0]
            X_t = build_initial_state(mixture, X_1, generator=generator)

            for step in range(NB_OF_STEPS):
                t_val = step * dt
                t = torch.full(
                    (B,),
                    t_val,
                    device=device,
                    dtype=mixture.dtype,
                )

                v = model(X_t, t, mixture, K)
                X_t = X_t + dt * v

            X_hat = X_t
            loss = criterion(X_hat, X_1)

            total_loss += loss.item() * B
            total_samples += B

    average_loss = total_loss / total_samples
    return average_loss
