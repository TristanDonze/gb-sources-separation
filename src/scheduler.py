import torch


class PolynomialDecayLR(torch.optim.lr_scheduler.LRScheduler):
    """Décroissance polynomiale du LR, puis maintien à eta_min."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        eta_min: float = 0.0,
        power: float = 3.0,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if power <= 0:
            raise ValueError("power must be positive")

        self.total_steps = total_steps
        self.eta_min = eta_min
        self.power = power

        super().__init__(optimizer, last_epoch)

    def _compute_lr(self, base_lr: float) -> float:
        progress = min(max(self.last_epoch / self.total_steps, 0.0), 1.0)
        decay = (1.0 - progress) ** self.power
        return self.eta_min + (base_lr - self.eta_min) * decay

    def get_lr(self) -> list[float]:
        return [self._compute_lr(base_lr) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self) -> list[float]:
        return [self._compute_lr(base_lr) for base_lr in self.base_lrs]