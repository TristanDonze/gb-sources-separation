import torch

def mask_slots(x: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != slot_mask.shape[0] or x.shape[1] != slot_mask.shape[1]:
        raise ValueError("x and slot_mask batch/slot dimensions must match")
    view_shape = slot_mask.shape + (1,) * (x.ndim - 2)
    return x * slot_mask.view(view_shape).to(dtype=x.dtype)
