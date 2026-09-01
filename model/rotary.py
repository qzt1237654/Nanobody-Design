import torch
from torch import nn


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        
        # Check if cache needs refresh (length, device, or first time)
        need_refresh = (
            self.cos_cached is None
            or seq_len != self.seq_len_cached
            or self.cos_cached.device != x.device
        )
        
        if need_refresh:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        # Ensure returned tensors match input device/dtype
        cos = self.cos_cached.to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached.to(device=x.device, dtype=x.dtype)
        
        return cos, sin


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    """
    Apply rotary positional embeddings to QKV tensor using pure PyTorch operations.
    
    Args:
        qkv: [B, L, 3, H, D] tensor
        cos: [1, L, 3, 1, D] tensor
        sin: [1, L, 3, 1, D] tensor
    
    Returns:
        [B, L, 3, H, D] tensor with rotary embeddings applied
    """
    return (qkv * cos) + (rotate_half(qkv) * sin)
