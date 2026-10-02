"""Sample-specific germline-absorbing graph for VHH discrete diffusion."""

import torch
import torch.nn.functional as F

from graph_lib import Graph, unsqueeze_as


class GermlineAbsorbing(Graph):
    """
    Forward marginal at each residue position i:

        p(x_t^i | x_0^i, g^i)
          = exp(-sigma) * delta(x_t^i = x_0^i)
          + (1-exp(-sigma)) * delta(x_t^i = g^i)

    The vocabulary contains only the 20 canonical amino-acid states.  There is
    no global MASK token: the absorbing state is the sample-specific germline
    residue g^i at each position.
    """

    def __init__(self, dim):
        self._dim = int(dim)

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return True

    @staticmethod
    def _require_germline(i, germline):
        if germline is None:
            raise ValueError("germline is required for GermlineAbsorbing")
        if germline.shape != i.shape:
            raise ValueError(
                f"germline shape {tuple(germline.shape)} must match "
                f"state shape {tuple(i.shape)}"
            )

    def rate(self, i, germline=None):
        """Column i of Q: non-germline states jump to their germline state."""
        self._require_germline(i, germline)
        return (
            F.one_hot(germline, num_classes=self.dim).float()
            - F.one_hot(i, num_classes=self.dim).float()
        )

    def transp_rate(self, i, germline=None):
        """
        Row i of Q, used by the reverse rate.

        If the current state is the germline residue, every non-germline source
        has forward rate 1 into that germline state.  Therefore the transpose
        row contains 1 for every non-germline token and 0 at germline.
        """
        self._require_germline(i, germline)

        edge = -F.one_hot(i, num_classes=self.dim).float()
        at_germline = (i == germline)
        edge = edge + at_germline[..., None].to(edge.dtype)
        return edge

    def transition(self, i, sigma, germline=None):
        """Column i of exp(sigma Q), i.e. the forward transition marginal."""
        self._require_germline(i, germline)
        sigma = unsqueeze_as(sigma, i[..., None])

        stay = torch.exp(-sigma)
        move = 1.0 - stay

        edge = stay * F.one_hot(i, num_classes=self.dim).to(sigma.dtype)
        edge = edge.scatter_add(
            -1,
            germline[..., None],
            move.expand(*i.shape, 1),
        )
        return edge

    def transp_transition(self, i, sigma, germline=None):
        """Row i of exp(sigma Q), used by Tweedie/denoising updates."""
        self._require_germline(i, germline)
        sigma = unsqueeze_as(sigma, i[..., None])

        stay = torch.exp(-sigma)
        move = 1.0 - stay

        edge = stay * F.one_hot(i, num_classes=self.dim).to(sigma.dtype)

        # For i == germline, every non-germline source reaches germline with
        # probability 1-exp(-sigma); the germline source remains with prob 1.
        edge = edge + (i == germline)[..., None].to(edge.dtype) * move
        return edge

    def sample_transition(self, i, sigma, germline=None):
        """Sample forward corruption x0 -> x_t."""
        self._require_germline(i, germline)
        move_chance = 1.0 - torch.exp(-sigma)
        move_indices = torch.rand(i.shape, device=i.device) < move_chance
        return torch.where(move_indices, germline, i)

    def staggered_score(self, score, dsigma, germline=None):
        """SEDD staggered-score update adapted to a per-position germline state."""
        if germline is None:
            raise ValueError("germline is required for staggered_score")

        scale = torch.exp(dsigma)
        extra_const = (1.0 - scale) * score.sum(dim=-1)

        out = score * scale[..., None]
        out = out.scatter_add(-1, germline[..., None], extra_const[..., None])
        return out

    def sample_limit(self, *batch_dims, germline=None):
        """The terminal/absorbing distribution is exactly the supplied germline."""
        if germline is None:
            raise ValueError("germline is required for sampling initialization")
        if batch_dims and tuple(batch_dims) != tuple(germline.shape):
            raise ValueError(
                f"batch_dims {tuple(batch_dims)} do not match germline shape "
                f"{tuple(germline.shape)}"
            )
        return germline.clone()

    def reverse_rate(self, i, score, germline=None):
        """Construct reverse-time rate = concrete_score * Q^T row."""
        self._require_germline(i, germline)
        normalized_rate = self.transp_rate(i, germline=germline).to(score) * score

        # Off-diagonal reverse rates must be non-negative.  The diagonal is then
        # set so each row sums to zero.
        current = i[..., None]
        normalized_rate = normalized_rate.scatter(
            -1, current, torch.zeros_like(current, dtype=normalized_rate.dtype)
        )
        diagonal = -normalized_rate.sum(dim=-1, keepdim=True)
        normalized_rate = normalized_rate.scatter(-1, current, diagonal)
        return normalized_rate

    def score_entropy(self, score, sigma, x, x0, germline=None):
        """DSE for all absorbing sites, including clean germline residues.

        Every x_t == g site contributes sum_{y != g} exp(log_score_y).
        Only absorbed mutations have a nonzero denoising target ratio.
        Non-germline sites have no reverse edges and contribute zero.
        """
        self._require_germline(x, germline)
        if x0.shape != x.shape:
            raise ValueError("x0 and x must have the same shape")

        at_germline = (x == germline)
        # Keep backward valid even when no site has been absorbed.
        entropy = score[..., 0] * 0.0
        if not at_germline.any():
            return entropy

        active_scores = score[at_germline].scatter(
            -1, germline[at_germline][..., None], -torch.inf
        )
        entropy = entropy.masked_scatter(
            at_germline, active_scores.exp().sum(dim=-1)
        )

        absorbed_mutation = at_germline & (x0 != germline)
        if absorbed_mutation.any():
            sigma = unsqueeze_as(sigma, x).expand_as(x)
            active_sigma = sigma[absorbed_mutation]
            # alpha / (1-alpha), without exp(sigma) overflow.
            ratio = torch.exp(-active_sigma) / (-torch.expm1(-active_sigma))
            target_log_score = score[absorbed_mutation].gather(
                -1, x0[absorbed_mutation][..., None]
            ).squeeze(-1)
            correction = torch.xlogy(ratio, ratio) - ratio * (1.0 + target_log_score)
            entropy = entropy + torch.zeros_like(entropy).masked_scatter(
                absorbed_mutation, correction
            )
        return entropy
