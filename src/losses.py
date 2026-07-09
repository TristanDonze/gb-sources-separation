import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from scipy.optimize import linear_sum_assignment


class FlowMatchingPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def compute_loss(self, predictions, targets):
        raise NotImplementedError

    def forward(self, v_pred, X_0, X_1):
        assert v_pred.shape == X_0.shape == X_1.shape

        pred_sources = v_pred[:, :-1]
        pred_residual = v_pred[:, -1]

        X0_sources = X_0[:, :-1]
        X1_sources = X_1[:, :-1]

        X0_residual = X_0[:, -1]
        X1_residual = X_1[:, -1]

        B, K, C, F = pred_sources.shape

        source_losses = []

        for b in range(B):
            cost = torch.zeros(
                (K, K),
                device=v_pred.device,
                dtype=v_pred.dtype,
            )

            for i in range(K):
                for j in range(K):
                    target_velocity = X1_sources[b, j] - X0_sources[b, i]
                    cost[i, j] = self.compute_loss(
                        pred_sources[b, i],
                        target_velocity,
                    )

            row_ind, col_ind = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )

            row_ind = torch.as_tensor(row_ind, device=v_pred.device)
            col_ind = torch.as_tensor(col_ind, device=v_pred.device)

            source_losses.append(cost[row_ind, col_ind].mean())

        source_loss = torch.stack(source_losses).mean()

        residual_target_velocity = X1_residual - X0_residual
        residual_loss = self.compute_loss(
            pred_residual,
            residual_target_velocity,
        )

        return source_loss + self.residual_weight * residual_loss


class ReconstructionPIT_Loss(nn.Module):
    def __init__(self, residual_weight: float = 1.0):
        super().__init__()
        self.residual_weight = residual_weight

    def compute_loss(self, predictions, targets):
        raise NotImplementedError

    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape

        pred_sources = predictions[:, :-1]
        true_sources = targets[:, :-1]

        pred_residual = predictions[:, -1]
        true_residual = targets[:, -1]

        B, K, C, F = pred_sources.shape

        source_losses = []

        for b in range(B):
            cost = torch.zeros(
                (K, K),
                device=predictions.device,
                dtype=predictions.dtype,
            )

            for i in range(K):
                for j in range(K):
                    cost[i, j] = self.compute_loss(
                        pred_sources[b, i],
                        true_sources[b, j],
                    )

            row_ind, col_ind = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )

            row_ind = torch.as_tensor(row_ind, device=predictions.device)
            col_ind = torch.as_tensor(col_ind, device=predictions.device)

            source_losses.append(cost[row_ind, col_ind].mean())

        source_loss = torch.stack(source_losses).mean()
        residual_loss = self.compute_loss(pred_residual, true_residual)

        return source_loss + self.residual_weight * residual_loss 

class FlowMatchingPIT_MSELoss(FlowMatchingPIT_Loss):
    def compute_loss(self, predictions, targets):
        return mse_loss(predictions, targets, reduction="mean")


class FlowMatchingPIT_RMSELoss(FlowMatchingPIT_Loss):
    def __init__(self, residual_weight: float = 1.0, eps: float = 1e-8):
        super().__init__(residual_weight=residual_weight)
        self.eps = eps

    def compute_loss(self, predictions, targets):
        return torch.sqrt(
            mse_loss(predictions, targets, reduction="mean") + self.eps
        )
    
class ReconstructionPIT_MSELoss(ReconstructionPIT_Loss):
    def compute_loss(self, predictions, targets):
        return mse_loss(predictions, targets, reduction="mean")


class ReconstructionPIT_RMSELoss(ReconstructionPIT_Loss):
    def __init__(self, residual_weight: float = 1.0, eps: float = 1e-8):
        super().__init__(residual_weight=residual_weight)
        self.eps = eps

    def compute_loss(self, predictions, targets):
        return torch.sqrt(
            mse_loss(predictions, targets, reduction="mean") + self.eps
        )