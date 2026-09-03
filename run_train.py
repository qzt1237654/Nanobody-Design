"""Formal VHH-only training loop for germline-absorbing SEDD."""

import datetime
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import graph_lib
import losses
import noise_lib
import sampling
import utils
from data_vhh_real import decode_sequence, get_vhh_dataloaders
from model import SEDD
from model.ema import ExponentialMovingAverage


# ============================================================
# Distributed setup
# ============================================================

def setup(rank, world_size, port):
    """
    Initialize distributed training only when world_size > 1.

    Single-GPU training does not need a process group.
    """
    if world_size == 1:
        return

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    backend = (
        "nccl"
        if torch.cuda.is_available() and dist.is_nccl_available()
        else "gloo"
    )

    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )


def cleanup():
    """
    Destroy process group if distributed training was initialized.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# Multiprocessing entry
# ============================================================

def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


# ============================================================
# Data iterator helper
# ============================================================

def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator

    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


# ============================================================
# Main training function
# ============================================================

def _run(rank, world_size, cfg):

    # --------------------------------------------------------
    # Basic configuration validation
    # --------------------------------------------------------

    if cfg.graph.type != "germline_absorb":
        raise ValueError(
            "run_train.py is VHH-only and requires germline_absorb"
        )

    if cfg.tokens != 20:
        raise ValueError(
            f"VHH vocabulary must contain exactly 20 states, "
            f"got {cfg.tokens}"
        )

    if cfg.model.length != cfg.data.max_length:
        raise ValueError(
            f"model.length ({cfg.model.length}) must equal "
            f"data.max_length ({cfg.data.max_length})"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        if rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"Worker rank {rank} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )

        torch.cuda.set_device(rank)

        device = torch.device(
            f"cuda:{rank}"
        )

        ddp_device_ids = [rank]

    else:

        device = torch.device("cpu")
        ddp_device_ids = None

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    work_dir = cfg.work_dir

    sample_dir = os.path.join(
        work_dir,
        "samples",
    )

    checkpoint_dir = os.path.join(
        work_dir,
        "checkpoints",
    )

    checkpoint_meta_path = os.path.join(
        work_dir,
        "checkpoints-meta",
        "checkpoint.pth",
    )

    if rank == 0:

        utils.makedirs(
            sample_dir
        )

        utils.makedirs(
            checkpoint_dir
        )

        utils.makedirs(
            os.path.dirname(
                checkpoint_meta_path
            )
        )

        logger = utils.get_logger(
            os.path.join(
                work_dir,
                "logs",
            )
        )

    def mprint(message):
        if rank == 0:
            logger.info(message)

    # --------------------------------------------------------
    # Log configuration
    # --------------------------------------------------------

    mprint(work_dir)
    mprint(cfg)
    mprint(f"Device: {device}")

    if device.type == "cuda":

        props = torch.cuda.get_device_properties(
            rank
        )

        mprint(
            f"GPU {rank}: "
            f"{props.name}, "
            f"memory="
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )

    # ========================================================
    # Graph and noise schedule
    # ========================================================

    graph = graph_lib.get_graph(
        cfg,
        device,
    )

    noise = noise_lib.get_noise(
        cfg
    ).to(device)

    # ========================================================
    # Model
    # ========================================================

    score_model = SEDD(
        cfg
    ).to(device)

    # Only wrap with DDP when multiple processes are used.
    if world_size > 1:

        score_model = DDP(
            score_model,
            device_ids=ddp_device_ids,
            static_graph=True,
        )

    num_parameters = sum(
        p.numel()
        for p in score_model.parameters()
    )

    mprint(
        f"Number of model parameters: "
        f"{num_parameters:,}"
    )

    # ========================================================
    # EMA
    # ========================================================

    ema = ExponentialMovingAverage(
        score_model.parameters(),
        decay=cfg.training.ema,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = losses.get_optimizer(
        cfg,
        score_model.parameters(),
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # ========================================================
    # AMP scaler
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    # ========================================================
    # Training state
    # ========================================================

    state = {
        "optimizer": optimizer,
        "scaler": scaler,
        "model": score_model,
        "ema": ema,
        "step": 0,
    }

    # --------------------------------------------------------
    # Restore checkpoint if present
    # --------------------------------------------------------

    state = utils.restore_checkpoint(
        checkpoint_meta_path,
        state,
        device,
    )

    initial_step = int(
        state["step"]
    )

    # ========================================================
    # Dataset
    # ========================================================

    mprint(
        "Loading VHH mature/germline pairs..."
    )

    train_loader, eval_loader = get_vhh_dataloaders(
        tsv_path=cfg.data.tsv_path,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        train_ratio=cfg.data.train_ratio,
        num_workers=cfg.data.num_workers,
        distributed=(world_size > 1),
        seed=cfg.data.seed,
    )

    train_iter = iter(
        train_loader
    )

    eval_iter = iter(
        eval_loader
    )

    # ========================================================
    # Training / evaluation functions
    # ========================================================

    optimize_fn = losses.optimization_manager(
        cfg
    )

    train_step_fn = losses.get_step_fn(
        noise,
        graph,
        train=True,
        optimize_fn=optimize_fn,
        accum=cfg.training.accum,
    )

    eval_step_fn = losses.get_step_fn(
        noise,
        graph,
        train=False,
        optimize_fn=optimize_fn,
        accum=cfg.training.accum,
    )

    # ========================================================
    # Snapshot sampling setup
    # ========================================================

    sampling_fn = None
    sample_attention_mask = None

    if cfg.training.snapshot_sampling:

        sample_batch = next(
            iter(eval_loader)
        )

        n_sample = min(
            int(cfg.sampling.batch_size),
            sample_batch["germline"].shape[0],
        )

        sample_germline = (
            sample_batch["germline"][:n_sample]
            .to(device)
        )

        sample_attention_mask = (
            sample_batch["attention_mask"][:n_sample]
            .to(device)
        )

        sampling_fn = sampling.get_sampling_fn(
            config=cfg,
            graph=graph,
            noise=noise,
            batch_dims=tuple(
                sample_germline.shape
            ),
            eps=cfg.noise.eps,
            device=device,
            germline=sample_germline,
            attention_mask=sample_attention_mask,
        )

    # ========================================================
    # Training initialization
    # ========================================================

    num_train_steps = int(
        cfg.training.n_iters
    )

    mprint(
        f"Starting training at optimizer step "
        f"{initial_step}."
    )

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    if rank == 0:

        pbar = tqdm(
            total=num_train_steps,
            initial=initial_step,
            desc="Training",
            file=sys.stdout,
            dynamic_ncols=True,
            ascii=True,
        )

    # ========================================================
    # Recovery accumulators
    # ========================================================

    overall_correct_accum = 0
    overall_total_accum = 0

    mutation_correct_accum = 0
    mutation_total_accum = 0

    # ========================================================
    # Main training loop
    # ========================================================

    while state["step"] < num_train_steps:

        previous_step = int(
            state["step"]
        )

        # ----------------------------------------------------
        # Get training batch
        # ----------------------------------------------------

        data_batch, train_iter = _next_batch(
            train_iter,
            train_loader,
        )

        batch = data_batch["mature"].to(
            device,
            non_blocking=True,
        )

        germline = data_batch["germline"].to(
            device,
            non_blocking=True,
        )

        attention_mask = data_batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Formal training step
        # ----------------------------------------------------

        loss = train_step_fn(
            state,
            batch,
            germline=germline,
            attention_mask=attention_mask,
        )

        # During gradient accumulation,
        # optimizer may not update yet.
        if state["step"] == previous_step:
            continue

        step = int(
            state["step"]
        )

        if rank == 0:
            pbar.update(1)

        # ====================================================
        # Recovery metrics
        # ====================================================

        with torch.no_grad():

            # -----------------------------------------------
            # Random diffusion timestep
            # -----------------------------------------------

            t = (
                torch.rand(
                    batch.shape[0],
                    device=device,
                )
                * 0.999
                + 0.001
            )

            sigma, _ = noise(
                t
            )

            # -----------------------------------------------
            # Forward corruption
            # -----------------------------------------------

            perturbed = graph.sample_transition(
                batch,
                sigma[:, None],
                germline=germline,
            )

            if attention_mask is not None:

                perturbed = torch.where(
                    attention_mask.bool(),
                    perturbed,
                    batch,
                )

            # -----------------------------------------------
            # Model score
            # -----------------------------------------------

            log_score = score_model(
                perturbed,
                sigma,
                germline=germline,
                attention_mask=attention_mask,
            )

            # Score-argmax heuristic prediction.
            pred = log_score.argmax(
                dim=-1
            )

            # -----------------------------------------------
            # Valid positions
            # -----------------------------------------------

            if attention_mask is not None:

                valid_mask = (
                    attention_mask.bool()
                )

            else:

                valid_mask = torch.ones_like(
                    batch,
                    dtype=torch.bool,
                )

            # ===============================================
            # Overall recovery
            # ===============================================

            overall_correct = (
                (pred == batch)
                & valid_mask
            )

            overall_correct_accum += (
                overall_correct.sum().item()
            )

            overall_total_accum += (
                valid_mask.sum().item()
            )

            # ===============================================
            # Mutation recovery
            #
            # Only positions that:
            #   1. are valid
            #   2. mature != germline
            #   3. were actually absorbed to germline
            # ===============================================

            mutation_target_mask = (
                valid_mask
                & (batch != germline)
                & (perturbed == germline)
            )

            mutation_correct = (
                (pred == batch)
                & mutation_target_mask
            )

            mutation_correct_accum += (
                mutation_correct.sum().item()
            )

            mutation_total_accum += (
                mutation_target_mask.sum().item()
            )

        # ====================================================
        # Training log
        # ====================================================

        if step % cfg.training.log_freq == 0:

            reduced_loss = (
                loss.detach().clone()
            )

            if dist.is_initialized():

                dist.all_reduce(
                    reduced_loss
                )

                reduced_loss /= (
                    world_size
                )

            avg_overall_recovery = (
                overall_correct_accum
                / max(
                    overall_total_accum,
                    1,
                )
            )

            avg_mutation_recovery = (
                mutation_correct_accum
                / max(
                    mutation_total_accum,
                    1,
                )
            )

            # ===============================================
            # IMPORTANT:
            # Obtain the REAL current LR from optimizer.
            #
            # This fixes:
            # NameError: current_lr is not defined
            # ===============================================

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            log_msg = (
                f"step: {step}, "
                f"lr: {current_lr:.3e}, "
                f"loss: {reduced_loss.item():.5e}, "
                f"overall_recovery_rate: "
                f"{avg_overall_recovery:.4f}, "
                f"mutation_recovery_rate: "
                f"{avg_mutation_recovery:.4f}, "
                f"mutation_targets: "
                f"{mutation_total_accum}"
            )

            mprint(
                log_msg
            )

            # -----------------------------------------------
            # Progress bar
            # -----------------------------------------------

            if rank == 0:

                pbar.set_postfix(
                    {
                        "lr": f"{current_lr:.2e}",
                        "loss": (
                            f"{reduced_loss.item():.4f}"
                        ),
                        "overall_rec": (
                            f"{avg_overall_recovery:.3f}"
                        ),
                        "mutation_rec": (
                            f"{avg_mutation_recovery:.3f}"
                        ),
                    }
                )

            # -----------------------------------------------
            # Reset recovery accumulators
            # -----------------------------------------------

            overall_correct_accum = 0
            overall_total_accum = 0

            mutation_correct_accum = 0
            mutation_total_accum = 0

        # ====================================================
        # Preemption / recovery checkpoint
        # ====================================================

        if (
            step
            % cfg.training.snapshot_freq_for_preemption
            == 0
            and rank == 0
        ):

            utils.save_checkpoint(
                checkpoint_meta_path,
                state,
            )

        # ====================================================
        # Evaluation
        # ====================================================

        if step % cfg.training.eval_freq == 0:

            eval_data, eval_iter = _next_batch(
                eval_iter,
                eval_loader,
            )

            eval_batch = eval_data[
                "mature"
            ].to(
                device,
                non_blocking=True,
            )

            eval_germline = eval_data[
                "germline"
            ].to(
                device,
                non_blocking=True,
            )

            eval_attention_mask = eval_data[
                "attention_mask"
            ].to(
                device,
                non_blocking=True,
            )

            # -----------------------------------------------
            # Formal eval loss
            # -----------------------------------------------

            eval_loss = eval_step_fn(
                state,
                eval_batch,
                germline=eval_germline,
                attention_mask=eval_attention_mask,
            )

            if dist.is_initialized():

                dist.all_reduce(
                    eval_loss
                )

                eval_loss /= world_size

            # ===============================================
            # Eval recovery
            # ===============================================

            with torch.no_grad():

                t = (
                    torch.rand(
                        eval_batch.shape[0],
                        device=device,
                    )
                    * 0.999
                    + 0.001
                )

                sigma_eval, _ = noise(
                    t
                )

                perturbed_eval = (
                    graph.sample_transition(
                        eval_batch,
                        sigma_eval[:, None],
                        germline=eval_germline,
                    )
                )

                if eval_attention_mask is not None:

                    perturbed_eval = torch.where(
                        eval_attention_mask.bool(),
                        perturbed_eval,
                        eval_batch,
                    )

                log_score_eval = score_model(
                    perturbed_eval,
                    sigma_eval,
                    germline=eval_germline,
                    attention_mask=eval_attention_mask,
                )

                pred_eval = (
                    log_score_eval.argmax(
                        dim=-1
                    )
                )

                if eval_attention_mask is not None:

                    eval_valid_mask = (
                        eval_attention_mask.bool()
                    )

                else:

                    eval_valid_mask = (
                        torch.ones_like(
                            eval_batch,
                            dtype=torch.bool,
                        )
                    )

                # -------------------------------------------
                # Eval overall recovery
                # -------------------------------------------

                eval_overall_correct = (
                    (pred_eval == eval_batch)
                    & eval_valid_mask
                )

                eval_overall_recovery = (
                    eval_overall_correct
                    .sum()
                    .float()
                    /
                    eval_valid_mask
                    .sum()
                    .clamp_min(1)
                    .float()
                )

                # -------------------------------------------
                # Eval mutation recovery
                # -------------------------------------------

                eval_mutation_mask = (
                    eval_valid_mask
                    & (
                        eval_batch
                        != eval_germline
                    )
                    & (
                        perturbed_eval
                        == eval_germline
                    )
                )

                eval_mutation_correct = (
                    (pred_eval == eval_batch)
                    & eval_mutation_mask
                )

                eval_mutation_targets = (
                    eval_mutation_mask.sum()
                )

                if eval_mutation_targets.item() > 0:

                    eval_mutation_recovery = (
                        eval_mutation_correct
                        .sum()
                        .float()
                        /
                        eval_mutation_targets
                        .float()
                    )

                else:

                    eval_mutation_recovery = (
                        torch.tensor(
                            float("nan"),
                            device=eval_batch.device,
                        )
                    )

            mprint(
                f"step: {step}, "
                f"eval_loss: "
                f"{eval_loss.item():.5e}, "
                f"eval_overall_recovery_rate: "
                f"{eval_overall_recovery.item():.4f}, "
                f"eval_mutation_recovery_rate: "
                f"{eval_mutation_recovery.item():.4f}, "
                f"eval_mutation_targets: "
                f"{eval_mutation_targets.item()}"
            )

        # ====================================================
        # Formal checkpoint / snapshot
        # ====================================================

        snapshot_now = (
            step
            % cfg.training.snapshot_freq
            == 0
            or step == num_train_steps
        )

        if snapshot_now:

            if rank == 0:

                utils.save_checkpoint(
                    os.path.join(
                        checkpoint_dir,
                        f"checkpoint_{step}.pth",
                    ),
                    state,
                )

                # Keep recovery checkpoint current.
                utils.save_checkpoint(
                    checkpoint_meta_path,
                    state,
                )

            # =================================================
            # Reverse diffusion sampling
            # =================================================

            if cfg.training.snapshot_sampling:

                mprint(
                    f"Generating VHH samples "
                    f"at step {step}"
                )

                this_sample_dir = os.path.join(
                    sample_dir,
                    f"iter_{step}",
                )

                utils.makedirs(
                    this_sample_dir
                )

                # Sampling uses EMA parameters.
                ema.store(
                    score_model.parameters()
                )

                ema.copy_to(
                    score_model.parameters()
                )

                try:

                    sample = sampling_fn(
                        score_model
                    )

                finally:

                    ema.restore(
                        score_model.parameters()
                    )

                sequences = [
                    decode_sequence(
                        sample[i],
                        sample_attention_mask[i],
                    )
                    for i in range(
                        sample.shape[0]
                    )
                ]

                file_name = os.path.join(
                    this_sample_dir,
                    f"sample_{rank}.txt",
                )

                with open(
                    file_name,
                    "w",
                    encoding="utf-8",
                ) as handle:

                    for sequence in sequences:

                        handle.write(
                            sequence + "\n"
                        )

            if dist.is_initialized():
                dist.barrier()

    # ========================================================
    # Finish
    # ========================================================

    if rank == 0:
        pbar.close()

    mprint(
        f"Training finished at optimizer step "
        f"{state['step']}."
    )