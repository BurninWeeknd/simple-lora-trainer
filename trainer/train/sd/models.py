import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from ..config import TrainConfig, log

def load_sd_models(cfg: TrainConfig, device, dtype):
    tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        cfg.base_model,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        cfg.base_model,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model,
        subfolder="unet",
        torch_dtype=dtype,
    ).to(device)

    scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        log("STATUS gradient_checkpointing=ENABLED")
    else:
        log("STATUS gradient_checkpointing=DISABLED")

    return tokenizer, text_encoder, vae, unet, scheduler
