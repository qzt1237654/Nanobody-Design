"""Time-conditioned clean posterior to germline concrete-score conversion."""

import torch
import torch.nn.functional as F


GERMLINE_SCORE_PARAMETERIZATION = "germline_posterior_v1"


def germline_log_scores(logits, sigma, indices):
    """At x_i=g_i, s(y)=alpha/(1-alpha) * P_theta(x0_i=y|xt,g,t).

    The posterior includes the germline class, which represents staying germline.
    Thus sum_{y!=g} s(y) <= alpha/(1-alpha). The graph ignores off-diagonal
    scores at x_i!=g_i; every current-state self ratio is exactly one.
    This posterior is explicitly time-conditioned, not time-free RADD.
    """
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    sigma = sigma.to(dtype=dtype).reshape(-1, 1, 1)
    if not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("Germline scores require finite sigma > 0")
    log_ratio = -sigma - torch.log(-torch.expm1(-sigma))
    log_scores = F.log_softmax(logits.to(dtype), dim=-1) + log_ratio
    return log_scores.scatter(-1, indices[..., None], 0.0)
