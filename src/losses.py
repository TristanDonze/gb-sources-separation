import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

def _reduce(loss, reduction="mean"):
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def hungarian_assignment(cost):
    B = cost.shape[0]
    cost_cpu = cost.detach().cpu().float().numpy()

    assignments = []

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost_cpu[b])

        row_ind = torch.as_tensor(row_ind, device=cost.device)
        col_ind = torch.as_tensor(col_ind, device=cost.device)

        assignment = torch.empty(cost.shape[1], dtype=torch.long, device=cost.device)
        assignment[row_ind] = col_ind
        assignments.append(assignment)

    return torch.stack(assignments)  # (B, K_pred)


def hungarian_loss(cost):
    assignment = hungarian_assignment(cost)
    selected_cost = cost.gather(dim=2, index=assignment.unsqueeze(-1)).squeeze(-1)

    return selected_cost.mean(dim=1)  # (B,)


class FlowMatchingPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def pairwise_cost(self, diff):
        raise NotImplementedError

    def scalar_loss(self, diff):
        raise NotImplementedError

    def forward(self, v_pred, X_0, X_1, reduction="mean"):
        # v_pred: (B, K+1, C, F)
        # X_0:    (B, K+1, C, F)
        # X_1:    (B, K+1, C, F)

        assert v_pred.shape == X_0.shape == X_1.shape

        pred_sources = v_pred[:, :-1]
        pred_residual = v_pred[:, -1]

        X0_sources = X_0[:, :-1]
        X1_sources = X_1[:, :-1]

        X0_residual = X_0[:, -1]
        X1_residual = X_1[:, -1]

        # target_velocity[b, i, j] = X1_source[j] - X0_slot[i]
        target_velocity = X1_sources[:, None] - X0_sources[:, :, None]
        # (B, K_pred, K_true, C, F)

        diff = pred_sources[:, :, None] - target_velocity
        # (B, K_pred, K_true, C, F)

        cost = self.pairwise_cost(diff)
        # (B, K, K)

        source_loss = hungarian_loss(cost) # (B,)

        residual_target_velocity = X1_residual - X0_residual
        residual_diff = pred_residual - residual_target_velocity
        residual_loss = self.scalar_loss(residual_diff) # (B, )

        loss = source_loss + self.residual_weight * residual_loss

        return _reduce(loss, reduction), _reduce(source_loss, reduction), _reduce(residual_loss, reduction)


class ReconstructionPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def pairwise_cost(self, diff):
        raise NotImplementedError

    def scalar_loss(self, diff):
        raise NotImplementedError

    def forward(self, predictions, targets, reduction="mean"):
        # predictions: (B, K+1, C, F)
        # targets:     (B, K+1, C, F)

        assert predictions.shape == targets.shape

        pred_sources = predictions[:, :-1]
        true_sources = targets[:, :-1]

        pred_residual = predictions[:, -1]
        true_residual = targets[:, -1]

        diff = pred_sources[:, :, None] - true_sources[:, None]
        # (B, K_pred, K_true, C, F)

        cost = self.pairwise_cost(diff)
        # (B, K, K)

        source_loss = hungarian_loss(cost)  # (B,)

        residual_diff = pred_residual - true_residual
        residual_loss = self.scalar_loss(residual_diff) # (B, )

        loss = source_loss + self.residual_weight * residual_loss

        return _reduce(loss, reduction), _reduce(source_loss, reduction), _reduce(residual_loss, reduction)


class FlowMatchingPET_DBNormalizedLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def find_assignment(self, v_pred, X_0, X_1):
        assert v_pred.shape == X_0.shape == X_1.shape

        pred_sources = v_pred[:, :-1]
        X0_sources = X_0[:, :-1]
        X1_sources = X_1[:, :-1]

        candidate_target_velocity = X1_sources[:, None] - X0_sources[:, :, None]
        candidate_diff = pred_sources[:, :, None] - candidate_target_velocity
        assignment_cost = candidate_diff.pow(2).mean(dim=(-1, -2))
        return hungarian_assignment(assignment_cost)

    def align_targets(self, X_1, assignment):
        X1_sources = X_1[:, :-1]

        B, K, C, F = X1_sources.shape
        assert assignment.shape == (B, K)

        gather_index = assignment[:, :, None, None].expand(B, K, C, F)
        aligned_X1_sources = X1_sources.gather(dim=1, index=gather_index)

        return torch.cat(
            [aligned_X1_sources, X_1[:, -1:]],
            dim=1,
        )

    def forward(self, v_pred, X_0, X_1, assignment, reduction="mean"):
        assert v_pred.shape == X_0.shape == X_1.shape

        aligned_X1 = self.align_targets(X_1, assignment)
        target_velocity = aligned_X1 - X_0

        diff = v_pred - target_velocity # represents the difference between predicted and target velocities
        error_energy = diff.pow(2).sum(dim=(1, 2, 3)) # shape (B,)
        target_energy = target_velocity.pow(2).sum(dim=(1, 2, 3))
        loss = 10 * torch.log10(
            (error_energy + self.eps) / (target_energy + self.eps)
        )

        source_mse = diff[:, :-1].pow(2).mean(dim=(1, 2, 3))
        residual_mse = diff[:, -1].pow(2).mean(dim=(1, 2))

        return (
            _reduce(loss, reduction),
            _reduce(source_mse, reduction),
            _reduce(residual_mse, reduction),
        )


class ReconstructionPIT_NMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions, targets, reduction="mean"):
        assert predictions.shape == targets.shape

        cost = (
            predictions[:, :, None] - targets[:, None]
        ).pow(2).mean(dim=(-1, -2))

        assignment = hungarian_assignment(cost)

        B, K, C, F = targets.shape
        gather_index = assignment[:, :, None, None].expand(B, K, C, F)
        aligned_slots = targets.gather(dim=1, index=gather_index)

        error_energy = (
            predictions - aligned_slots
        ).pow(2).sum(dim=(-1, -2))

        source_energy = aligned_slots.pow(2).sum(dim=(-1, -2))

        nmse_source = error_energy / (source_energy + self.eps)
        nmse_source_db = 10 * torch.log10(nmse_source + self.eps)

        return (
            _reduce(nmse_source, reduction),
            _reduce(nmse_source_db, reduction),
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
