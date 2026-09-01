import hydra
from omegaconf import DictConfig, open_dict
import torch.multiprocessing as mp
import random
import run_train


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):

    with open_dict(cfg):
        cfg.work_dir = (
            hydra.core.hydra_config.HydraConfig
            .get()
            .runtime
            .output_dir
        )

        cfg.wandb_name = cfg.work_dir.split("/")[-1]

    port = random.randint(10000, 20000)

    ngpus = cfg.ngpus

    mp.spawn(
        run_train.run_multiprocess,
        args=(ngpus, cfg, port),
        nprocs=ngpus,
        join=True
    )


if __name__ == "__main__":
    main()