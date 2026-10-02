import torch

import os
import logging
from omegaconf import OmegaConf, open_dict


def unwrap_model(model):
    """Extract the base model from DDP wrapper if present."""
    return model.module if hasattr(model, "module") else model


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        validate_checkpoint_parameterization(loaded_state, state['model'])
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        
        model_to_load = unwrap_model(state['model'])
        model_to_load.load_state_dict(loaded_state['model'], strict=True)
        
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        
        if 'scaler' in loaded_state and 'scaler' in state:
            state['scaler'].load_state_dict(loaded_state['scaler'])
            logging.info("Restored AMP GradScaler state")
        
        return state


def save_checkpoint(ckpt_dir, state):
    model_to_save = unwrap_model(state['model'])
    
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': model_to_save.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step'],
        'score_parameterization': getattr(model_to_save, 'score_parameterization', 'raw'),
    }
    
    if 'scaler' in state:
        saved_state['scaler'] = state['scaler'].state_dict()
    
    torch.save(saved_state, ckpt_dir)


def validate_checkpoint_parameterization(checkpoint, model):
    expected = getattr(unwrap_model(model), 'score_parameterization', 'raw')
    actual = checkpoint.get('score_parameterization', 'raw')
    if actual != expected:
        raise ValueError(
            f"Checkpoint score parameterization {actual!r} is incompatible with "
            f"{expected!r}. Start a new training run after the germline DSE fix; "
            "legacy raw-score weights cannot be resumed as posterior logits."
        )
