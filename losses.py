import numpy as np
import torch
import torch.optim as optim

from model import utils as mutils


# ============================================================
# Denoising Score Entropy loss
# ============================================================

def get_loss_fn(
    noise,
    graph,
    train,
    sampling_eps=1e-3,
    lv=False,
):
    """
    Build the VHH germline-absorbing diffusion loss function.

    Args:
        noise:
            Noise schedule, currently LogLinearNoise.

        graph:
            GermlineAbsorbing graph.

        train:
            Whether the model is in training mode.

        sampling_eps:
            Minimum sampled diffusion time.
            t is sampled from [sampling_eps, 1].

        lv:
            Legacy likelihood-variance option.
            Not used in the current VHH project.
    """

    def loss_fn(
        model,
        batch,
        germline=None,
        t=None,
        perturbed_batch=None,
        attention_mask=None,
    ):
        """
        Compute Denoising Score Entropy loss.

        Args:
            model:
                DDiT / SEDD score model.

            batch:
                Mature VHH sequence x0.
                Shape: [B, L]

            germline:
                Corresponding germline sequence g.
                Shape: [B, L]

            t:
                Optional diffusion timestep.
                Shape: [B]

            perturbed_batch:
                Optional pre-computed noisy sequence x_t.
                Shape: [B, L]

            attention_mask:
                Valid-residue mask.
                1 = real amino acid
                0 = padding
                Shape: [B, L]

        Returns:
            Per-sequence loss.
            Shape: [B]
        """

        if germline is None:
            raise ValueError(
                "germline must be provided for VHH germline-absorbing diffusion"
            )

        if germline.shape != batch.shape:
            raise ValueError(
                f"germline shape {tuple(germline.shape)} must match "
                f"batch shape {tuple(batch.shape)}"
            )

        if attention_mask is not None and attention_mask.shape != batch.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} must match "
                f"batch shape {tuple(batch.shape)}"
            )

        # ----------------------------------------------------
        # 1. Sample diffusion time t
        # ----------------------------------------------------

        if t is None:
            if lv:
                raise NotImplementedError(
                    "Likelihood-variance sampling is not used "
                    "in the current VHH project."
                )

            t = (
                (1.0 - sampling_eps)
                * torch.rand(
                    batch.shape[0],
                    device=batch.device,
                )
                + sampling_eps
            )

        # ----------------------------------------------------
        # 2. Convert t -> sigma(t), d sigma / dt
        # ----------------------------------------------------

        sigma, dsigma = noise(t)

        # ----------------------------------------------------
        # 3. Forward corruption
        #
        # mature x0
        #     ↓
        # germline-absorbing forward process
        #     ↓
        # noisy sequence x_t
        # ----------------------------------------------------

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(
                batch,
                sigma[:, None],
                germline=germline,
            )

            # Padding is not a biological diffusion state.
            # Keep padded positions unchanged.
            if attention_mask is not None:
                perturbed_batch = torch.where(
                    attention_mask.bool(),
                    perturbed_batch,
                    batch,
                )

        # ----------------------------------------------------
        # 4. Score-network prediction
        #
        # S_theta(x_t, t | germline)
        # ----------------------------------------------------

        log_score_fn = mutils.get_score_fn(
            model,
            train=train,
            sampling=False,
        )

        log_score = log_score_fn(
            perturbed_batch,
            sigma,
            germline=germline,
            attention_mask=attention_mask,
        )

        # ----------------------------------------------------
        # 5. Denoising Score Entropy
        # ----------------------------------------------------

        # graph operations are kept in float32 for numerical
        # stability even when AMP is enabled.
        loss = graph.score_entropy(
            log_score.float(),
            sigma[:, None],
            perturbed_batch,
            batch,
            germline=germline,
        )

        # ----------------------------------------------------
        # 6. Remove padding positions from the objective
        # ----------------------------------------------------

        if attention_mask is not None:
            if attention_mask.shape != loss.shape:
                raise ValueError(
                    f"attention_mask shape "
                    f"{tuple(attention_mask.shape)} must match "
                    f"loss shape {tuple(loss.shape)}"
                )

            loss = loss * attention_mask.to(loss.dtype)

        # ----------------------------------------------------
        # 7. Continuous-time weighting
        #
        # d sigma / dt × score-entropy
        # ----------------------------------------------------

        loss = (
            dsigma[:, None]
            * loss
        ).sum(dim=-1)

        return loss

    return loss_fn


# ============================================================
# Optimizer
# ============================================================

def get_optimizer(config, params):
    """
    Create optimizer from config.
    """

    if config.optim.optimizer == "Adam":
        optimizer = optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(
                config.optim.beta1,
                config.optim.beta2,
            ),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )

    elif config.optim.optimizer == "AdamW":
        optimizer = optim.AdamW(
            params,
            lr=config.optim.lr,
            betas=(
                config.optim.beta1,
                config.optim.beta2,
            ),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )

    else:
        raise NotImplementedError(
            f"Optimizer "
            f"{config.optim.optimizer!r} "
            f"is not supported."
        )

    return optimizer


# ============================================================
# Optimization manager
# ============================================================

def optimization_manager(config):
    """
    Build optimizer-step function.

    Handles:
        - AMP GradScaler
        - learning-rate warmup
        - gradient clipping
        - optimizer.step()
    """

    def optimize_fn(
        optimizer,
        scaler,
        params,
        step,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
    ):
        # Convert generator to list because the parameters
        # are used by gradient clipping.
        params = list(params)

        # AMP: restore gradients to their true scale before
        # clipping.
        scaler.unscale_(optimizer)

        # ----------------------------------------------------
        # Learning-rate warmup
        # ----------------------------------------------------

        if warmup > 0:
            warmup_ratio = np.minimum(
                step / warmup,
                1.0,
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = (
                    lr * warmup_ratio
                )

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=grad_clip,
            )

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


# ============================================================
# One training / validation step
# ============================================================

def get_step_fn(
    noise,
    graph,
    train,
    optimize_fn,
    accum,
):
    """
    Build one training or validation step.

    training:
        mature
          ↓
        forward corruption
          ↓
        x_t
          ↓
        DDiT
          ↓
        DSE loss
          ↓
        backward
          ↓
        AdamW
          ↓
        EMA

    validation:
        Uses EMA parameters temporarily.
    """

    if accum < 1:
        raise ValueError(
            f"training.accum must be >= 1, got {accum}"
        )

    loss_fn = get_loss_fn(
        noise,
        graph,
        train,
    )

    accum_iter = 0
    total_loss = None

    def step_fn(
        state,
        batch,
        germline=None,
        attention_mask=None,
    ):
        nonlocal accum_iter
        nonlocal total_loss

        if germline is None:
            raise ValueError(
                "germline must be provided to step_fn"
            )

        model = state["model"]

        # ====================================================
        # TRAINING
        # ====================================================

        if train:
            optimizer = state["optimizer"]
            scaler = state["scaler"]

            loss = loss_fn(
                model,
                batch,
                germline=germline,
                attention_mask=attention_mask,
            ).mean()

            # Divide by accumulation factor so the total
            # accumulated gradient has the correct scale.
            scaled_loss = loss / accum

            scaler.scale(
                scaled_loss
            ).backward()

            accum_iter += 1

            if total_loss is None:
                total_loss = loss.detach()
            else:
                total_loss = (
                    total_loss
                    + loss.detach()
                )

            # ------------------------------------------------
            # Only update model after `accum` micro-batches
            # ------------------------------------------------

            if accum_iter == accum:
                state["step"] += 1

                optimize_fn(
                    optimizer,
                    scaler,
                    model.parameters(),
                    step=state["step"],
                )

                # Update EMA after optimizer update.
                state["ema"].update(
                    model.parameters()
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # Average loss across accumulated
                # micro-batches.
                loss = (
                    total_loss
                    / accum
                )

                accum_iter = 0
                total_loss = None

            else:
                # run_train.py ignores this value until an
                # optimizer step actually occurs.
                loss = scaled_loss.detach()

        # ====================================================
        # VALIDATION
        # ====================================================

        else:
            ema = state["ema"]

            # Evaluate with EMA weights, but restore current
            # training weights afterwards.
            ema.store(
                model.parameters()
            )

            ema.copy_to(
                model.parameters()
            )

            try:
                with torch.no_grad():
                    loss = loss_fn(
                        model,
                        batch,
                        germline=germline,
                        attention_mask=attention_mask,
                    ).mean()

            finally:
                ema.restore(
                    model.parameters()
                )

        return loss

    return step_fn