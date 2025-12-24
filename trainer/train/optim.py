import torch
from diffusers.optimization import get_scheduler
from .config import TrainConfig, log

def build_optimizer(param_groups, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            param_groups,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(
            param_groups,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            param_groups,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    lrs = [pg["lr"] for pg in opt.param_groups]
    log(f"STATUS optimizer_param_group_lrs={lrs}")
    return opt

def build_scheduler(cfg: TrainConfig, optimizer, dataset_len: int):
    steps_per_epoch = (dataset_len + cfg.batch_size - 1) // cfg.batch_size
    updates_per_epoch = (steps_per_epoch + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
    num_training_steps = updates_per_epoch * cfg.epochs

    sched = get_scheduler(
        name=cfg.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=cfg.num_cycles,
    )

    log(
        f"STATUS training_plan steps_per_epoch={steps_per_epoch} "
        f"updates_per_epoch={updates_per_epoch} total_updates={num_training_steps}"
    )
    return sched, num_training_steps
