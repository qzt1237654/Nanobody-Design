import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None, attention_mask=None):
        """
        Batch shape: [B, L] int. D given from graph
        
        Args:
            model: score model
            batch: mature sequence x0 [B, L]
            cond: germline sequence [B, L] for germline_absorb, None otherwise
            t: timestep
            perturbed_batch: pre-computed x_t (optional)
            attention_mask: [B, L] 1=valid, 0=PAD (optional)
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        # Check if graph needs germline for forward corruption
        if perturbed_batch is None:
            if hasattr(graph, '__class__') and graph.__class__.__name__ == 'GermlineAbsorbing':
                if cond is None:
                    raise ValueError("GermlineAbsorbing graph requires germline (cond) for sample_transition")
                perturbed_batch = graph.sample_transition(batch, sigma[:, None], germline=cond)
            else:
                perturbed_batch = graph.sample_transition(batch, sigma[:, None])
            
            # Protect PAD tokens from corruption
            if attention_mask is not None:
                perturbed_batch = torch.where(
                    attention_mask.bool(),
                    perturbed_batch,
                    batch
                )

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        
        # Pass germline and attention_mask to score network if needed
        if hasattr(graph, '__class__') and graph.__class__.__name__ == 'GermlineAbsorbing':
            if cond is None:
                raise ValueError("GermlineAbsorbing graph requires germline (cond) for score network")
            log_score = log_score_fn(perturbed_batch, sigma, germline=cond, attention_mask=attention_mask)
        else:
            log_score = log_score_fn(perturbed_batch, sigma, attention_mask=attention_mask)
        
        # Pass germline to score_entropy if needed
        # Ensure log_score is float32 for graph operations (model may output bfloat16)
        if hasattr(graph, '__class__') and graph.__class__.__name__ == 'GermlineAbsorbing':
            loss = graph.score_entropy(log_score.float(), sigma[:, None], perturbed_batch, batch, germline=cond)
        else:
            loss = graph.score_entropy(log_score.float(), sigma[:, None], perturbed_batch, batch)
        
        # Zero out PAD positions - they should not contribute to loss
        if attention_mask is not None:
            if attention_mask.shape != loss.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} must match "
                    f"loss shape {tuple(loss.shape)}"
                )
            loss = loss * attention_mask.to(loss.dtype)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None, attention_mask=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond, attention_mask=attention_mask).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond, attention_mask=attention_mask).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn