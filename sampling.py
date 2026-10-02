"""Reverse sampling for VHH germline-absorbing SEDD."""

import torch

from catsample import sample_categorical
from model import utils as mutils


class EulerPredictor:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(
        self,
        score_fn,
        x,
        t,
        step_size,
        germline,
        attention_mask,
    ):
        sigma, dsigma = self.noise(t)
        score = score_fn(
            x,
            sigma,
            germline=germline,
            attention_mask=attention_mask,
        )

        reverse_rate = (
            step_size
            * dsigma[..., None]
            * self.graph.reverse_rate(x, score, germline=germline)
        )
        x_new = self.graph.sample_rate(x, reverse_rate)

        # Padding positions are not biological states and never change.
        return torch.where(attention_mask.bool(), x_new, germline)


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t, germline, attention_mask):
        sigma = self.noise(t)[0]
        score = score_fn(
            x,
            sigma,
            germline=germline,
            attention_mask=attention_mask,
        )

        # Exact one-site clean posterior at an absorbing position. Construct
        # the residual directly instead of subtracting staggered score terms.
        # The bounded posterior score head guarantees nonnegative mass.
        probs = score * torch.expm1(sigma)[..., None]
        probs = probs.scatter(-1, germline[..., None], 0.0)
        probs = probs.scatter(-1, germline[..., None], 1.0 - probs.sum(-1, keepdim=True))
        fixed = torch.nn.functional.one_hot(x, self.graph.dim).to(probs)
        probs = torch.where((x == germline)[..., None], probs, fixed)

        x_new = sample_categorical(probs)
        return torch.where(attention_mask.bool(), x_new, germline)


def get_sampling_fn(
    config,
    graph,
    noise,
    batch_dims,
    eps,
    device,
    germline,
    attention_mask,
):
    if config.sampling.predictor != "euler":
        raise ValueError(
            f"This VHH project only supports predictor='euler', "
            f"got {config.sampling.predictor!r}"
        )

    return get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        steps=config.sampling.steps,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=device,
        germline=germline,
        attention_mask=attention_mask,
    )


def get_pc_sampler(
    graph,
    noise,
    batch_dims,
    steps,
    denoise=True,
    eps=1e-5,
    device=torch.device("cpu"),
    germline=None,
    attention_mask=None,
):
    if germline is None:
        raise ValueError("germline is required for reverse sampling")
    if attention_mask is None:
        raise ValueError("attention_mask is required for reverse sampling")
    if tuple(germline.shape) != tuple(batch_dims):
        raise ValueError(
            f"batch_dims {tuple(batch_dims)} must match germline shape "
            f"{tuple(germline.shape)}"
        )
    if attention_mask.shape != germline.shape:
        raise ValueError("attention_mask and germline must have the same shape")
    if steps < 1:
        raise ValueError(f"sampling.steps must be >= 1, got {steps}")
    if not 0 < eps < 1:
        raise ValueError("sampling eps must be between 0 and 1")

    germline = germline.to(device)
    attention_mask = attention_mask.to(device)

    predictor = EulerPredictor(graph, noise)
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        score_fn = mutils.get_score_fn(model, train=False, sampling=True)

        # Terminal state of the forward process is the germline itself.
        x = graph.sample_limit(*batch_dims, germline=germline).to(device)
        x = torch.where(attention_mask.bool(), x, germline)

        timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
        dt = (1.0 - eps) / steps

        for idx in range(steps):
            t = timesteps[idx] * torch.ones(
                x.shape[0], 1, device=device
            )
            x = predictor.update_fn(
                score_fn,
                x,
                t,
                dt,
                germline=germline,
                attention_mask=attention_mask,
            )

        if denoise:
            t = timesteps[-1] * torch.ones(
                x.shape[0], 1, device=device
            )
            x = denoiser.update_fn(
                score_fn,
                x,
                t,
                germline=germline,
                attention_mask=attention_mask,
            )

        return torch.where(attention_mask.bool(), x, germline)

    return pc_sampler
