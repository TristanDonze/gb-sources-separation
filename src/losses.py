import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

def _reduce(loss, reduction="mean"):
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")


def _masked_source_reduce(values, source_mask, reduction="mean"):
    masked = values * source_mask.to(values.dtype)
    if reduction == "mean":
        return masked.sum() / source_mask.sum().clamp_min(1)
    if reduction == "sum":
        return masked.sum()
    if reduction == "none":
        return masked.sum(dim=1) / source_mask.sum(dim=1).clamp_min(1)
    raise ValueError(f"Unknown reduction: {reduction}")


def hungarian_assignment(cost, K=None):
    """Return source assignments, using only each sample's active K x K cost."""
    B, num_pred, num_true = cost.shape
    if num_pred != num_true:
        raise ValueError("Hungarian source cost must be square")

    if K is None:
        K = torch.full((B,), num_pred, dtype=torch.long, device=cost.device)
    if K.shape != (B,):
        raise ValueError("K must have shape (B,)")

    cost_cpu = cost.detach().cpu().float().numpy()
    assignments = torch.full(
        (B, num_pred), -1, dtype=torch.long, device=cost.device
    )
    for b, k_value in enumerate(K.detach().cpu().tolist()):
        k = int(k_value)
        row_ind, col_ind = linear_sum_assignment(cost_cpu[b, :k, :k])
        assignments[b, torch.as_tensor(row_ind, device=cost.device)] = torch.as_tensor(
            col_ind, device=cost.device
        )
    return assignments


def _selected_source_cost(cost, assignment, source_mask):
    selected = cost.gather(dim=2, index=assignment.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return selected * source_mask.to(selected.dtype)


class FlowMatchingPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def pairwise_cost(self, diff):
        raise NotImplementedError

    def scalar_loss(self, diff):
        raise NotImplementedError

    def forward(self, v_pred, X_0, X_1, slot_mask, reduction="mean"):
        assert v_pred.shape == X_0.shape == X_1.shape
        K = slot_mask[:, :-1].sum(dim=1)
        source_mask = slot_mask[:, :-1]

        target_velocity = X_1[:, None, :-1] - X_0[:, :-1, None]
        diff = v_pred[:, :-1, None] - target_velocity
        cost = self.pairwise_cost(diff)
        assignment = hungarian_assignment(cost, K)
        selected_cost = _selected_source_cost(cost, assignment, source_mask)

        residual_diff = v_pred[:, -1] - (X_1[:, -1] - X_0[:, -1])
        residual_loss = self.scalar_loss(residual_diff)
        source_loss = _masked_source_reduce(selected_cost, source_mask, reduction)
        reduced_residual = _reduce(residual_loss, reduction)
        return (
            source_loss + self.residual_weight * reduced_residual,
            source_loss,
            reduced_residual,
        )


class ReconstructionPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def pairwise_cost(self, diff):
        raise NotImplementedError

    def scalar_loss(self, diff):
        raise NotImplementedError

    def forward(self, predictions, targets, slot_mask, reduction="mean"):
        assert predictions.shape == targets.shape
        K = slot_mask[:, :-1].sum(dim=1)
        source_mask = slot_mask[:, :-1]

        diff = predictions[:, :-1, None] - targets[:, None, :-1]
        cost = self.pairwise_cost(diff)
        assignment = hungarian_assignment(cost, K)
        selected_cost = _selected_source_cost(cost, assignment, source_mask)

        residual_loss = self.scalar_loss(predictions[:, -1] - targets[:, -1])
        source_loss = _masked_source_reduce(selected_cost, source_mask, reduction)
        reduced_residual = _reduce(residual_loss, reduction)
        return (
            source_loss + self.residual_weight * reduced_residual,
            source_loss,
            reduced_residual,
        )


class FlowMatchingPET_DBNormalizedLoss(nn.Module):
    def __init__(self, residual_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.residual_weight = residual_weight
        self.eps = eps

    def find_assignment(self, v_pred, X_0, X_1, slot_mask):
        assert v_pred.shape == X_0.shape == X_1.shape
        K = slot_mask[:, :-1].sum(dim=1)

        candidate_velocity = X_1[:, None, :-1] - X_0[:, :-1, None]
        candidate_diff = v_pred[:, :-1, None] - candidate_velocity
        assignment_cost = candidate_diff.pow(2).mean(dim=(-1, -2))
        return hungarian_assignment(assignment_cost, K)

    def align_targets(self, X_1, assignment, slot_mask):
        B, L, C, F = X_1.shape
        if assignment.shape != (B, L - 1):
            raise ValueError("assignment has an invalid shape")

        gather_index = assignment.clamp_min(0)[:, :, None, None].expand(B, L - 1, C, F)
        aligned_sources = X_1[:, :-1].gather(dim=1, index=gather_index)
        aligned_sources = aligned_sources * slot_mask[:, :-1, None, None]
        return torch.cat([aligned_sources, X_1[:, -1:]], dim=1)

    def forward(self, v_pred, X_0, X_1, assignment, slot_mask, reduction="mean"):
        assert v_pred.shape == X_0.shape == X_1.shape
        K = slot_mask[:, :-1].sum(dim=1)

        aligned_X1 = self.align_targets(X_1, assignment, slot_mask)
        target_velocity = aligned_X1 - X_0
        diff = v_pred - target_velocity
        source_mask = slot_mask[:, :-1]

        source_error_by_slot = diff[:, :-1].pow(2).sum(dim=(-1, -2))
        source_target_by_slot = target_velocity[:, :-1].pow(2).sum(dim=(-1, -2))
        source_error = (source_error_by_slot * source_mask).sum(dim=1)
        source_target = (source_target_by_slot * source_mask).sum(dim=1)
        residual_error = diff[:, -1].pow(2).sum(dim=(-1, -2))
        residual_target = target_velocity[:, -1].pow(2).sum(dim=(-1, -2))

        loss = 10 * torch.log10(
            (source_error + self.residual_weight * residual_error + self.eps)
            / (source_target + self.residual_weight * residual_target + self.eps)
        )
        source_mse_by_slot = diff[:, :-1].pow(2).mean(dim=(-1, -2))
        source_mse = _masked_source_reduce(source_mse_by_slot, source_mask, reduction)
        residual_mse = _reduce(diff[:, -1].pow(2).mean(dim=(-1, -2)), reduction)
        return _reduce(loss, reduction), source_mse, residual_mse


class ReconstructionPIT_NMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions, targets, source_mask, reduction="mean"):
        assert predictions.shape == targets.shape
        if source_mask.shape != predictions.shape[:2]:
            raise ValueError("source_mask must match prediction source dimensions")
        K = source_mask.sum(dim=1)

        cost = (predictions[:, :, None] - targets[:, None]).pow(2).mean(dim=(-1, -2))
        assignment = hungarian_assignment(cost, K)
        B, S, C, F = targets.shape
        gather_index = assignment.clamp_min(0)[:, :, None, None].expand(B, S, C, F)
        aligned = targets.gather(dim=1, index=gather_index)

        error_energy = (predictions - aligned).pow(2).sum(dim=(-1, -2))
        source_energy = aligned.pow(2).sum(dim=(-1, -2))
        nmse = error_energy / (source_energy + self.eps)
        nmse_db = 10 * torch.log10(nmse + self.eps)
        if reduction == "none":
            mask = source_mask.to(nmse.dtype)
            return nmse * mask, nmse_db * mask
        return (
            _masked_source_reduce(nmse, source_mask, reduction),
            _masked_source_reduce(nmse_db, source_mask, reduction),
        )


class FlowMatchingPIT_MSELoss(FlowMatchingPIT_Loss):
    def pairwise_cost(self, diff):
        return diff.pow(2).mean(dim=(-1, -2))

    def scalar_loss(self, diff):
        return diff.pow(2).mean(dim=(-1, -2))


class FlowMatchingPIT_RMSELoss(FlowMatchingPIT_Loss):
    def __init__(self, residual_weight: float = 1.0, eps: float = 1e-8):
        super().__init__(residual_weight=residual_weight)
        self.eps = eps

    def pairwise_cost(self, diff):
        return torch.sqrt(diff.pow(2).mean(dim=(-1, -2)) + self.eps)

    def scalar_loss(self, diff):
        return torch.sqrt(diff.pow(2).mean(dim=(-1, -2)) + self.eps)


class ReconstructionPIT_MSELoss(ReconstructionPIT_Loss):
    def pairwise_cost(self, diff):
        return diff.pow(2).mean(dim=(-1, -2))

    def scalar_loss(self, diff):
        return diff.pow(2).mean(dim=(-1, -2))


class ReconstructionPIT_RMSELoss(ReconstructionPIT_Loss):
    def __init__(self, residual_weight: float = 1.0, eps: float = 1e-8):
        super().__init__(residual_weight=residual_weight)
        self.eps = eps

    def pairwise_cost(self, diff):
        return torch.sqrt(diff.pow(2).mean(dim=(-1, -2)) + self.eps)

    def scalar_loss(self, diff):
        return torch.sqrt(diff.pow(2).mean(dim=(-1, -2)) + self.eps)
