"""Formal VHH-only training loop for germline-absorbing SEDD."""

import datetime
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import graph_lib
import losses
import noise_lib
import sampling
import utils
from data_vhh_real import decode_sequence, get_vhh_dataloaders
from model import SEDD
from model.ema import ExponentialMovingAverage


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
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
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _run(rank, world_size, cfg):
    if cfg.graph.type != "germline_absorb":
        raise ValueError("run_train.py is VHH-only and requires germline_absorb")
    if cfg.tokens != 20:
        raise ValueError(f"VHH vocabulary must contain exactly 20 states, got {cfg.tokens}")
    if cfg.model.length != cfg.data.max_length:
        raise ValueError(
            f"model.length ({cfg.model.length}) must equal "
            f"data.max_length ({cfg.data.max_length})"
        )

    if torch.cuda.is_available():
        if rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"Worker rank {rank} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        ddp_device_ids = [rank]
    else:
        device = torch.device("cpu")
        ddp_device_ids = None

    work_dir = cfg.work_dir
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_path = os.path.join(
        work_dir, "checkpoints-meta", "checkpoint.pth"
    )

    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_path))
        logger = utils.get_logger(os.path.join(work_dir, "logs"))

    def mprint(message):
        if rank == 0:
            logger.info(message)

    mprint(work_dir)
    mprint(cfg)
    mprint(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(rank)
        mprint(
            f"GPU {rank}: {props.name}, "
            f"memory={props.total_memory / (1024 ** 3):.2f} GB"
        )

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    score_model = SEDD(cfg).to(device)
    score_model = DDP(
        score_model,
        device_ids=ddp_device_ids,
        static_graph=True,
    )

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of model parameters: {num_parameters:,}")

    ema = ExponentialMovingAverage(
        score_model.parameters(),
        decay=cfg.training.ema,
    )

    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    optimizer.zero_grad(set_to_none=True)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    state = {
        "optimizer": optimizer,
        "scaler": scaler,
        "model": score_model,
        "ema": ema,
        "step": 0,
    }
    state = utils.restore_checkpoint(
        checkpoint_meta_path,
        state,
        device,
    )
    initial_step = int(state["step"])

    mprint("Loading VHH mature/germline pairs...")
    train_loader, eval_loader = get_vhh_dataloaders(
        tsv_path=cfg.data.tsv_path,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        train_ratio=cfg.data.train_ratio,
        num_workers=cfg.data.num_workers,
        distributed=(world_size > 1),
        seed=cfg.data.seed,
    )
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    optimize_fn = losses.optimization_manager(cfg)
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

    sampling_fn = None
    sample_attention_mask = None
    if cfg.training.snapshot_sampling:
        sample_batch = next(iter(eval_loader))
        n_sample = min(
            int(cfg.sampling.batch_size),
            sample_batch["germline"].shape[0],
        )
        sample_germline = sample_batch["germline"][:n_sample].to(device)
        sample_attention_mask = sample_batch["attention_mask"][:n_sample].to(device)

        sampling_fn = sampling.get_sampling_fn(
            config=cfg,
            graph=graph,
            noise=noise,
            batch_dims=tuple(sample_germline.shape),
            eps=cfg.noise.eps,
            device=device,
            germline=sample_germline,
            attention_mask=sample_attention_mask,
        )

    num_train_steps = int(cfg.training.n_iters)
    mprint(f"Starting training at optimizer step {initial_step}.")

    while state["step"] < num_train_steps:
        previous_step = int(state["step"])

        data_batch, train_iter = _next_batch(train_iter, train_loader)
        batch = data_batch["mature"].to(device, non_blocking=True)
        germline = data_batch["germline"].to(device, non_blocking=True)
        attention_mask = data_batch["attention_mask"].to(
            device, non_blocking=True
        )

        loss = train_step_fn(
            state,
            batch,
            germline=germline,
            attention_mask=attention_mask,
        )

        # During gradient accumulation there may be no optimizer update yet.
        if state["step"] == previous_step:
            continue

        step = int(state["step"])

        if step % cfg.training.log_freq == 0:
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss)
            reduced_loss /= world_size
            mprint(
                f"step: {step}, training_loss: {reduced_loss.item():.5e}"
            )

        if (
            step % cfg.training.snapshot_freq_for_preemption == 0
            and rank == 0
        ):
            utils.save_checkpoint(checkpoint_meta_path, state)

        if step % cfg.training.eval_freq == 0:
            eval_data, eval_iter = _next_batch(eval_iter, eval_loader)
            eval_batch = eval_data["mature"].to(device, non_blocking=True)
            eval_germline = eval_data["germline"].to(device, non_blocking=True)
            eval_attention_mask = eval_data["attention_mask"].to(
                device, non_blocking=True
            )

            eval_loss = eval_step_fn(
                state,
                eval_batch,
                germline=eval_germline,
                attention_mask=eval_attention_mask,
            )
            dist.all_reduce(eval_loss)
            eval_loss /= world_size
            mprint(
                f"step: {step}, evaluation_loss: {eval_loss.item():.5e}"
            )

        snapshot_now = (
            step % cfg.training.snapshot_freq == 0
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
                # Keep the interruption checkpoint current at every formal snapshot.
                utils.save_checkpoint(checkpoint_meta_path, state)

            if cfg.training.snapshot_sampling:
                mprint(f"Generating VHH samples at step {step}")
                this_sample_dir = os.path.join(sample_dir, f"iter_{step}")
                utils.makedirs(this_sample_dir)

                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                try:
                    sample = sampling_fn(score_model)
                finally:
                    ema.restore(score_model.parameters())

                sequences = [
                    decode_sequence(sample[i], sample_attention_mask[i])
                    for i in range(sample.shape[0])
                ]

                file_name = os.path.join(
                    this_sample_dir,
                    f"sample_{rank}.txt",
                )
                with open(file_name, "w", encoding="utf-8") as handle:
                    for sequence in sequences:
                        handle.write(sequence + "\n")

            dist.barrier()

    mprint(f"Training finished at optimizer step {state['step']}.")
