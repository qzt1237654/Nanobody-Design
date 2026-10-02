"""Finite-state oracles and end-to-end regression checks for the P1 fixes."""

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from catsample import sample_categorical
from graph_lib_germline import GermlineAbsorbing
from model import SEDD
from model.ema import ExponentialMovingAverage
from model.score_parameterization import germline_log_scores
from noise_lib import LogLinearNoise
from sampling import Denoiser, EulerPredictor, get_pc_sampler
import utils


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def reference_dse(log_scores, sigma, clean, current, germline):
    """Direct generator/transition enumeration, independent of production masks."""
    dim = log_scores.numel()
    Q = torch.zeros((dim, dim), dtype=torch.float64)
    for source in range(dim):
        if source != germline:
            Q[source, source] = -1
            Q[germline, source] = 1
    marginal = torch.matrix_exp(sigma * Q)[:, clean]
    result = log_scores.sum() * 0
    for y in range(dim):
        if y == current or Q[current, y] == 0:
            continue
        ratio = marginal[y] / marginal[current]
        result = result + Q[current, y] * (
            log_scores[y].exp() - ratio * log_scores[y]
            + torch.xlogy(ratio, ratio) - ratio
        )
    return result


@pytest.mark.parametrize("sigma", [0.001, 0.3, 2., 6.9])
def test_loss_and_gradient_match_enumerated_generator(sigma):
    graph = GermlineAbsorbing(3)
    for g in range(3):
        for clean in range(3):
            for current in {clean, g}:
                scores = torch.tensor([[[.2, -.3, .7]]], dtype=torch.float64, requires_grad=True)
                actual = graph.score_entropy(scores, torch.tensor([[sigma]], dtype=torch.float64),
                    torch.tensor([[current]]), torch.tensor([[clean]]), torch.tensor([[g]])).sum()
                expected = reference_dse(scores.reshape(-1), sigma, clean, current, g)
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
                ag = torch.autograd.grad(actual, scores, retain_graph=True)[0]
                eg = torch.autograd.grad(expected, scores)[0]
                torch.testing.assert_close(ag, eg, atol=1e-8, rtol=1e-8)


def test_conserved_sites_train_and_preserved_sites_do_not():
    graph = GermlineAbsorbing(20)
    g = torch.zeros((1, 3), dtype=torch.long)
    x0, xt = torch.tensor([[0, 1, 1]]), torch.tensor([[0, 0, 1]])
    scores = torch.zeros((1, 3, 20), dtype=torch.float64, requires_grad=True)
    loss = graph.score_entropy(scores, torch.tensor([[math.log(2)]], dtype=torch.float64), xt, x0, g)
    torch.testing.assert_close(loss, torch.tensor([[19., 18., 0.]], dtype=torch.float64))
    loss.sum().backward()
    assert scores.grad[0, 0, 1:].sum() == 19
    assert scores.grad[0, 2].abs().sum() == 0
    only_preserved = graph.score_entropy(scores[:, 2:], torch.tensor([[.5]]), xt[:, 2:], x0[:, 2:], g[:, 2:])
    assert only_preserved.requires_grad and only_preserved.item() == 0


def test_population_true_score_is_stationary():
    # Conservation supervision is necessary for the true mixed-population score.
    graph = GermlineAbsorbing(2)
    alpha, pi = .5, .1
    theta = torch.tensor(math.log(alpha*pi/(1-alpha*pi)), dtype=torch.float64, requires_grad=True)
    scores = torch.stack([theta * 0, theta]).reshape(1, 1, 2)
    g = torch.zeros((1, 1), dtype=torch.long)
    sigma = torch.tensor([[math.log(2)]], dtype=torch.float64)
    total = ((1-pi)*graph.score_entropy(scores, sigma, g, g, g)
             + pi*(1-alpha)*graph.score_entropy(scores, sigma, g, g+1, g)).sum()
    total.backward()
    assert abs(theta.grad.item()) < 1e-12


@pytest.mark.parametrize("scale", [0., 1., 100.])
def test_posterior_head_bounds_euler_and_denoiser(monkeypatch, scale):
    torch.manual_seed(8)
    graph, noise = GermlineAbsorbing(20), LogLinearNoise(.001)
    g = torch.randint(20, (4, 9))
    logits = torch.randn(4, 9, 20) * scale
    mask = torch.ones_like(g)
    mask[:, -1] = 0
    captured = []
    def capture(weights):
        # Validate through the actual categorical routine before recording.
        sample_categorical(weights)
        captured.append(weights)
        return weights.argmax(-1)
    monkeypatch.setattr("sampling.sample_categorical", capture)
    monkeypatch.setattr("graph_lib.sample_categorical", capture)
    def score_fn(x, sigma, **kwargs):
        return germline_log_scores(logits, sigma, x).exp()
    for t_value in [1., .5, .0088046875, .001]:
        t = torch.full((4, 1), t_value)
        dt = min(.0078046875, t_value*.5)
        EulerPredictor(graph, noise).update_fn(score_fn, g, t, dt, g, mask)
        Denoiser(graph, noise).update_fn(score_fn, g, t, g, mask)
    for probabilities in captured:
        assert probabilities.min() >= -1e-6
        torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(g, dtype=torch.float32))


def test_denoiser_matches_bayes_and_freezes_non_germline(monkeypatch):
    graph, noise = GermlineAbsorbing(3), LogLinearNoise(.001)
    prior = torch.tensor([.7, .2, .1], dtype=torch.float64)
    t = torch.tensor([[.4]], dtype=torch.float64)
    alpha = torch.exp(-noise(t)[0]).item()
    marginal = alpha * prior
    marginal[0] += 1-alpha
    x, g = torch.tensor([[0, 1]]), torch.tensor([[0, 0]])
    def score_fn(x, sigma, **kwargs):
        return (marginal / marginal[0]).expand(1, 2, 3)
    captured = []
    def capture(p):
        captured.append(p)
        return p.argmax(-1)
    monkeypatch.setattr("sampling.sample_categorical", capture)
    Denoiser(graph, noise).update_fn(score_fn, x, t, g, torch.ones_like(x))
    posterior = prior * torch.tensor([1., 1-alpha, 1-alpha])
    posterior /= posterior.sum()
    torch.testing.assert_close(captured[0][0, 0], posterior)
    torch.testing.assert_close(captured[0][0, 1], torch.tensor([0., 1., 0.], dtype=torch.float64))


@pytest.mark.parametrize("weights", [[-147., 148.], [float('nan'), 1.], [float('inf'), 1.], [0., 0.]])
def test_invalid_probabilities_raise(weights):
    with pytest.raises(ValueError):
        sample_categorical(torch.tensor([weights]))


def test_only_roundoff_negative_mass_is_tolerated():
    assert sample_categorical(torch.tensor([[-1e-8, 1.]])).item() == 1
    # Low-precision epsilon must not authorize percent-scale negative mass.
    for dtype in [torch.float16, torch.bfloat16]:
        with pytest.raises(ValueError, match='Negative'):
            sample_categorical(torch.tensor([[-.001, 1.001]], dtype=dtype))


def test_old_unbounded_first_euler_step_is_rejected():
    graph, noise = GermlineAbsorbing(20), LogLinearNoise(.001)
    g = torch.zeros((1, 1), dtype=torch.long)
    def raw_score(x, sigma, **kwargs):
        return torch.ones((*x.shape, 20))
    with pytest.raises(ValueError, match="Negative"):
        EulerPredictor(graph, noise).update_fn(raw_score, g, torch.ones((1, 1)), .999/128, g, torch.ones_like(g))


def tiny_model():
    cfg = OmegaConf.load(ROOT / 'configs/config.yaml')
    cfg.model.hidden_size = cfg.model.cond_dim = 32
    cfg.model.n_heads, cfg.model.n_blocks, cfg.model.dropout = 4, 1, 0.
    return SEDD(cfg)


def test_real_model_backward_sampling_and_padding():
    torch.manual_seed(42)
    model, graph, noise = tiny_model(), GermlineAbsorbing(20), LogLinearNoise(.001)
    g = torch.zeros((2, 32), dtype=torch.long)
    mask = torch.ones_like(g)
    mask[:, -4:] = 0
    x0 = g.clone()
    x0[:, :5] = 1
    import losses
    loss = losses.get_loss_fn(noise, graph, train=True)(model, x0, germline=g,
        t=torch.tensor([.3, .8]), perturbed_batch=g, attention_mask=mask).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.output_layer.linear.weight.grad.abs().sum() > 0
    result = get_pc_sampler(graph, noise, g.shape, 128, eps=.001, germline=g, attention_mask=mask)(model)
    assert torch.equal(result[:, -4:], g[:, -4:])
    # Bounded initialization does not force all positions to mutate at t=1.
    t = torch.ones((2, 1))
    from model.utils import get_score_fn
    score = get_score_fn(model, sampling=True)(g, noise(t)[0], germline=g, attention_mask=mask)
    leave = .999/128 * noise(t)[1] * score.scatter(-1, g[..., None], 0).sum(-1)
    assert leave.max() < .01


def test_oracle_sampler_recovers_population():
    torch.manual_seed(14)
    class Oracle(torch.nn.Module):
        def forward(self, x, sigma, **kwargs):
            alpha = torch.exp(-sigma).reshape(-1, 1)
            p_mutant = .2 * alpha
            ratio = p_mutant / (1-p_mutant)
            scores = torch.stack([torch.zeros_like(ratio), ratio.log()], -1)
            return scores.scatter(-1, x[..., None], 0.)
    g = torch.zeros((10000, 1), dtype=torch.long)
    sampled = get_pc_sampler(GermlineAbsorbing(2), LogLinearNoise(.001), g.shape,
        128, eps=.001, germline=g, attention_mask=torch.ones_like(g))(Oracle())
    assert abs(sampled.float().mean().item() - .2) < .015


def test_checkpoint_version_roundtrip_and_legacy_rejection(tmp_path):
    model = tiny_model()
    state = {'model': model, 'optimizer': torch.optim.AdamW(model.parameters()),
             'ema': ExponentialMovingAverage(model.parameters(), .99), 'step': 3}
    path = tmp_path / 'checkpoint.pth'
    utils.save_checkpoint(path, state)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint['score_parameterization'] == 'germline_posterior_v1'
    assert utils.restore_checkpoint(path, state, 'cpu')['step'] == 3
    del checkpoint['score_parameterization']
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='incompatible'):
        utils.restore_checkpoint(path, state, 'cpu')


def test_formal_training_sampling_and_resume(tmp_path):
    output = tmp_path / 'run'
    args = [sys.executable, str(ROOT / 'train.py'), f'data.tsv_path={ROOT / "real_vhh_200.tsv"}',
        'data.num_workers=0', 'training.batch_size=4', 'training.log_freq=1',
        'training.eval_freq=1', 'training.snapshot_freq=2', 'training.snapshot_freq_for_preemption=1',
        'sampling.batch_size=2', 'sampling.steps=8', 'model.hidden_size=32', 'model.cond_dim=32',
        'model.n_blocks=1', 'model.n_heads=4', 'model.dropout=0', f'hydra.run.dir={output}']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for final_step in [2, 3]:
        completed = subprocess.run(args + [f'training.n_iters={final_step}'], cwd=ROOT,
            capture_output=True, text=True, timeout=90, env=env)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        checkpoint = torch.load(output / f'checkpoint_{final_step}.pth', weights_only=False)
        assert checkpoint['step'] == final_step
        assert checkpoint['score_parameterization'] == 'germline_posterior_v1'
        sequences = (output / 'samples' / f'iter_{final_step}' / 'sample_0.txt').read_text().splitlines()
        assert len(sequences) == 2
        assert all(96 <= len(s) <= 98 and set(s) <= set('ACDEFGHIKLMNPQRSTVWY') for s in sequences)
