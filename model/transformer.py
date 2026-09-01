import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c, seqlens=None, attention_mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        # Apply rotary positional embeddings
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        
        # Split Q, K, V and rearrange for PyTorch SDPA
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each: [B, L, H, D]
        
        # Rearrange to [B, H, L, D] for scaled_dot_product_attention
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)  # [B, H, L, D]
        v = v.transpose(1, 2)  # [B, H, L, D]
        
        # Prepare attention mask for SDPA
        # PyTorch SDPA: True = attend, False = mask out
        # Our attention_mask: 1 = valid, 0 = padding
        attn_mask = None
        if attention_mask is not None:
            # Create 4D mask: [B, 1, 1, L] -> broadcasts to [B, H, L, L]
            # This masks out padded keys
            attn_mask = attention_mask.bool()[:, None, None, :]  # [B, 1, 1, L]
        
        # Apply PyTorch native scaled dot-product attention
        # Force MATH backend to avoid any FlashAttention dependency
        from torch.nn.attention import SDPBackend, sdpa_kernel
        
        with sdpa_kernel(SDPBackend.MATH):
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        
        # Rearrange back to [B, L, H, D] and flatten to [B, L, H*D]
        x = x_attn.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Zero out padded positions to prevent any drift
        if attention_mask is not None:
            x = x * attention_mask[..., None].to(x.dtype)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        
        # Zero out padded positions again after MLP
        if attention_mask is not None:
            x = x * attention_mask[..., None].to(x.dtype)
        
        return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SEDD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        # Global [MASK]-absorbing SEDD adds one extra vocabulary token.
        # Germline-absorbing diffusion does NOT add a token because the absorbing
        # state at each position is the germline amino-acid token itself.
        self.absorb = config.graph.type == "absorb"
        self.germline_conditioning = config.graph.type == "germline_absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)

        # For MochiDiff-style germline diffusion the score network must receive
        # the germline sequence as an additional input, not only through the
        # diffusion graph.  Re-use the amino-acid embedding table and learn a
        # projection that injects the germline embedding into the current x_t
        # representation position-by-position.
        #
        # Zero initialization preserves the behavior of an existing pretrained
        # SEDD/ESM-initialized model at step 0; training then learns how strongly
        # and in which directions to use germline information.
        if self.germline_conditioning:
            self.germline_proj = nn.Linear(
                config.model.hidden_size,
                config.model.hidden_size,
                bias=False,
            )
            nn.init.zeros_(self.germline_proj.weight)
        else:
            self.germline_proj = None

        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, indices, sigma, germline=None, attention_mask=None):
        """
        Args:
            indices: Current diffusion state x_t, shape [B, L].
            sigma: Noise level, shape [B] (or compatible with TimestepEmbedder).
            germline: Germline conditioning sequence, shape [B, L]. Required
                when config.graph.type == "germline_absorb".
            attention_mask: Attention mask, shape [B, L]. 1 for valid tokens,
                0 for PAD tokens. Optional.
        """
        x = self.vocab_embed(indices)

        if self.germline_conditioning:
            if germline is None:
                raise ValueError(
                    "SEDD with graph.type='germline_absorb' requires a germline "
                    "sequence in model.forward(..., germline=germline)."
                )
            if germline.shape != indices.shape:
                raise ValueError(
                    f"germline shape {tuple(germline.shape)} must match "
                    f"indices shape {tuple(indices.shape)}"
                )

            # Keep token ids on the same device/dtype as the current sequence.
            germline = germline.to(device=indices.device, dtype=indices.dtype)
            germline_emb = self.vocab_embed(germline)
            x = x + self.germline_proj(germline_emb)
        
        # Validate and convert attention_mask
        if attention_mask is not None:
            if attention_mask.shape != indices.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} must match "
                    f"indices shape {tuple(indices.shape)}"
                )
            attention_mask = attention_mask.to(device=indices.device, dtype=torch.bool)

        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None, attention_mask=attention_mask)

            x = self.output_layer(x, c)


        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
            
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x
