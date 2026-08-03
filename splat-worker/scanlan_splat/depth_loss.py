from __future__ import annotations

def masked_robust_depth_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 0.05,
) -> torch.Tensor:
    import torch

    valid = mask & torch.isfinite(predicted) & torch.isfinite(target) & (target > 0)
    if not torch.any(valid):
        return predicted.sum() * 0.0
    residual = torch.abs(predicted[valid] - target[valid])
    delta_tensor = torch.as_tensor(delta, dtype=residual.dtype, device=residual.device)
    huber = torch.where(
        residual <= delta_tensor,
        0.5 * residual.square() / delta_tensor,
        residual - 0.5 * delta_tensor,
    )
    return huber.mean()


def depth_weight(step: int, total_steps: int, start: float = 0.2, end: float = 0.02) -> float:
    stable_fraction = min(max(step / max(total_steps * 0.65, 1.0), 0.0), 1.0)
    return start * (1.0 - stable_fraction) + end * stable_fraction
