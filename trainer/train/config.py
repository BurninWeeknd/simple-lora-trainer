import argparse
from dataclasses import dataclass
import torch

def log(msg: str) -> None:
    print(msg, flush=True)

def resolve_dtype(precision: str) -> torch.dtype:
    p = (precision or "").lower()
    if p in ("fp16", "float16"):
        return torch.float16
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32

def parse_target_modules(s: str) -> list[str] | None:
    s = (s or "").strip()
    if not s:
        return None
    raw = [t.strip() for t in s.replace(" ", ",").split(",")]
    out = [t for t in raw if t]
    return out or None

@dataclass
class TrainConfig:
    model_type: str
    base_model: str
    dataset: str
    caption_ext: str
    resolution: int
    batch_size: int
    epochs: int
    shuffle: bool
    lora_rank: int
    lora_alpha: float
    unet_lr: float
    clip_lr: float
    precision: str
    output: str

    gradient_checkpointing: bool = False
    grad_accum_steps: int = 1
    repeats: int = 1
    save_every_epochs: int = 0
    seed: int = 0
    log_every: int = 10

    do_inference: bool = False
    inference_prompt: str = ""
    inference_steps: int = 20
    inference_images: int = 2

    clip_skip: int = 0

    scheduler_type: str = "constant"
    warmup_steps: int = 0
    num_cycles: int = 1

    prepend_token: str | None = None
    append_token: str | None = None

    cache_latents: bool = False
    bucket_enabled: bool = False
    bucket_min_res: int = 512
    bucket_max_res: int = 1536
    bucket_step: int = 64

    optimizer: str = "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    memorize_first_token: bool = False
    lora_dropout: float = 0.0
    momentum: float = 0.9
    nesterov: bool = False
    target_modules: list[str] | None = None
    use_xformers: bool = False
    cpu_offload: bool = False

def log_train_config(cfg: TrainConfig) -> None:
    log("===== TRAIN CONFIG =====")
    log(f"model_type={cfg.model_type}")
    log(f"base_model={cfg.base_model}")
    log(f"precision={cfg.precision}")
    log(f"device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log(f"resolution={cfg.resolution}")
    log(f"batch_size={cfg.batch_size}")
    log(f"grad_accum_steps={cfg.grad_accum_steps}")
    log(f"effective_batch_size={cfg.batch_size * cfg.grad_accum_steps}")
    log(f"epochs={cfg.epochs}")
    log(f"repeats={cfg.repeats}")
    log(f"optimizer={cfg.optimizer}")
    log(f"unet_lr={cfg.unet_lr}")
    log(f"clip_lr={cfg.clip_lr}")
    log(f"weight_decay={cfg.weight_decay}")
    log(f"betas=({cfg.beta1}, {cfg.beta2})")
    log(f"epsilon={cfg.epsilon}")
    log(f"momentum={cfg.momentum}")
    log(f"nesterov={cfg.nesterov}")
    log(f"scheduler={cfg.scheduler_type}")
    log(f"warmup_steps={cfg.warmup_steps}")
    log(f"num_cycles={cfg.num_cycles}")
    log(f"lora_rank={cfg.lora_rank}")
    log(f"lora_alpha={cfg.lora_alpha}")
    log(f"lora_dropout={cfg.lora_dropout}")
    log(f"clip_skip={cfg.clip_skip}")
    log(f"train_clip={cfg.clip_lr > 0}")
    log(f"cache_latents={cfg.cache_latents}")
    log(f"bucket_enabled={cfg.bucket_enabled}")
    if cfg.bucket_enabled:
        log(f"bucket_min={cfg.bucket_min_res} max={cfg.bucket_max_res} step={cfg.bucket_step}")
    log(f"prepend_token={cfg.prepend_token}")
    log(f"append_token={cfg.append_token}")
    log(f"memorize_first_token={cfg.memorize_first_token}")
    log(f"do_inference={cfg.do_inference}")
    log("===== END CONFIG =====")

def build_arg_parser(default_resolution: int):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["sd", "sdxl"], required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--caption_ext", default=".txt")
    ap.add_argument("--prepend_token", default="")
    ap.add_argument("--append_token", default="")
    ap.add_argument("--resolution", type=int, default=default_resolution)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--cache_latents", action="store_true")
    ap.add_argument("--bucket", action="store_true")
    ap.add_argument("--bucket_min_res", type=int, default=512)
    ap.add_argument("--bucket_max_res", type=int, default=1536)
    ap.add_argument("--bucket_step", type=int, default=64)
    ap.add_argument("--scheduler_type", default="constant")
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--num_cycles", type=int, default=1)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=8.0)
    ap.add_argument("--unet_lr", type=float, default=1e-4)
    ap.add_argument("--clip_lr", type=float, default=0.0)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--output", required=True)
    ap.add_argument("--save_every_epochs", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--do_inference", action="store_true")
    ap.add_argument("--inference_prompt", default="")
    ap.add_argument("--inference_steps", type=int, default=20)
    ap.add_argument("--inference_images", type=int, default=2)
    ap.add_argument("--clip_skip", type=int, default=0)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--epsilon", type=float, default=1e-8)
    ap.add_argument("--memorize_first_token", action="store_true")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--nesterov", action="store_true")
    ap.add_argument("--target_modules", default="", help="Comma-separated list, e.g. to_q,to_k,to_v,to_out.0")
    ap.add_argument("--use_xformers", action="store_true", help="Enable xFormers memory efficient attention")
    ap.add_argument("--cpu_offload", action="store_true", help="Offload frozen components (VAE/text encoders) to CPU to save VRAM")
    return ap

def cfg_from_args(args) -> TrainConfig:
    return TrainConfig(
        model_type=args.model_type,
        base_model=args.base_model,
        dataset=args.dataset,
        caption_ext=args.caption_ext,
        prepend_token=args.prepend_token.strip() or None,
        append_token=args.append_token.strip() or None,
        resolution=args.resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=args.shuffle,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        unet_lr=args.unet_lr,
        clip_lr=args.clip_lr,
        precision=args.precision,
        output=args.output,
        save_every_epochs=args.save_every_epochs,
        repeats=args.repeats,
        grad_accum_steps=args.grad_accum_steps,
        do_inference=args.do_inference,
        inference_prompt=args.inference_prompt,
        inference_steps=args.inference_steps,
        inference_images=args.inference_images,
        clip_skip=args.clip_skip,
        scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
        num_cycles=args.num_cycles,
        gradient_checkpointing=args.gradient_checkpointing,
        cache_latents=args.cache_latents,
        bucket_enabled=args.bucket,
        bucket_min_res=args.bucket_min_res,
        bucket_max_res=args.bucket_max_res,
        bucket_step=args.bucket_step,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        memorize_first_token=args.memorize_first_token,
        lora_dropout=args.lora_dropout,
        momentum=args.momentum,
        nesterov=args.nesterov,
        target_modules=parse_target_modules(args.target_modules),
        use_xformers=args.use_xformers,
        cpu_offload=args.cpu_offload,
    )
