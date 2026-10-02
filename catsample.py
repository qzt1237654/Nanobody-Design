import torch
import torch.nn.functional as F


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    # A categorical sampler must never silently accept invalid Euler weights.
    # Permit only rounding-scale negative residuals (e.g. 1-sum(p) at p≈1).
    if categorical_probs.dtype != torch.float64:
        categorical_probs = categorical_probs.float()
    if not torch.isfinite(categorical_probs).all():
        raise ValueError("Categorical weights must be finite")
    tolerance = (
        8 * torch.finfo(categorical_probs.dtype).eps
        * categorical_probs.abs().sum(-1, keepdim=True)
    )
    if (categorical_probs < -tolerance).any():
        raise ValueError("Negative categorical weights: invalid reverse transition")
    categorical_probs = categorical_probs.clamp_min(0)
    if (categorical_probs.sum(-1) <= 0).any():
        raise ValueError("Categorical rows must have positive mass")
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
