import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def hungarian_mean_loss(cost):
    B = cost.shape[0]
    cost_cpu = cost.detach().cpu().float().numpy()

    losses = []

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost_cpu[b])

        row_ind = torch.as_tensor(row_ind, device=cost.device)
        col_ind = torch.as_tensor(col_ind, device=cost.device)

        losses.append(cost[b, row_ind, col_ind].mean())

    return torch.stack(losses).mean()


class FlowMatchingPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def pairwise_cost(self, diff):
        raise NotImplementedError

    def scalar_loss(self, diff):
        raise NotImplementedError

    def forward(self, v_pred, X_0, X_1):
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

        source_loss = hungarian_mean_loss(cost)

        residual_target_velocity = X1_residual - X0_residual
        residual_diff = pred_residual - residual_target_velocity
        residual_loss = self.scalar_loss(residual_diff)

        return source_loss + self.residual_weight * residual_loss


class ReconstructionPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def pairwise_cost(self, diff):
        raise NotImplementedError

    def scalar_loss(self, diff):
        raise NotImplementedError

    def forward(self, predictions, targets):
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

        source_loss = hungarian_mean_loss(cost)

        residual_diff = pred_residual - true_residual
        residual_loss = self.scalar_loss(residual_diff)

        return source_loss + self.residual_weight * residual_loss


class FlowMatchingPIT_MSELoss(FlowMatchingPIT_Loss):
    def pairwise_cost(self, diff):
        return diff.pow(2).mean(dim=(-1, -2))

    def scalar_loss(self, diff):
        return diff.pow(2).mean()


class FlowMatchingPIT_RMSELoss(FlowMatchingPIT_Loss):
    def __init__(self, residual_weight: float = 1.0, eps: float = 1e-8):
        super().__init__(residual_weight=residual_weight)
        self.eps = eps

    def pairwise_cost(self, diff):
        return torch.sqrt(diff.pow(2).mean(dim=(-1, -2)) + self.eps)

    def scalar_loss(self, diff):
        return torch.sqrt(diff.pow(2).mean() + self.eps)


class ReconstructionPIT_MSELoss(ReconstructionPIT_Loss):
    def pairwise_cost(self, diff):
        return diff.pow(2).mean(dim=(-1, -2))

    def scalar_loss(self, diff):
        return diff.pow(2).mean()


class ReconstructionPIT_RMSELoss(ReconstructionPIT_Loss):
    def __init__(self, residual_weight: float = 1.0, eps: float = 1e-8):
        super().__init__(residual_weight=residual_weight)
        self.eps = eps

    def pairwise_cost(self, diff):
        return torch.sqrt(diff.pow(2).mean(dim=(-1, -2)) + self.eps)

    def scalar_loss(self, diff):
        return torch.sqrt(diff.pow(2).mean() + self.eps)