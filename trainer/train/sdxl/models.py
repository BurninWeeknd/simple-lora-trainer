import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from ..config import log

def load_sdxl_components(base_model: str, device, dtype):
    log(f"STATUS loading UNet (device={device})")
    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", torch_dtype=dtype
    ).to(device)

    log(f"STATUS loading VAE (device={device})")
    vae = AutoencoderKL.from_pretrained(
        base_model, subfolder="vae", torch_dtype=dtype
    ).to(device)

    log(f"STATUS loading text_encoder_1 (device={device})")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)

    log(f"STATUS loading text_encoder_2 (device={device})")
    text_encoder_2 = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)

    log("STATUS loading tokenizer_1")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")

    log("STATUS loading tokenizer_2")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")

    log("STATUS all SDXL components loaded")
    return unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2

def load_sdxl_scheduler(base_model: str):
    return DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
