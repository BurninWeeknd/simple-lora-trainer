from pathlib import Path
import torch
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

@torch.no_grad()
def decode_latents_to_pil(vae: AutoencoderKL, latents: torch.Tensor):
    latents = latents / 0.18215
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()
    images = []
    for i in range(image.shape[0]):
        images.append(Image.fromarray((image[i] * 255).astype("uint8")))
    return images

@torch.no_grad()
def run_inference_preview_in_memory(
    *,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDPMScheduler,
    output_dir: Path,
    prompt: str,
    steps: int,
    num_images: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    clip_skip: int,
    guidance_scale: float = 7.5,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    unet.eval()
    vae.eval()
    text_encoder.eval()

    scheduler.set_timesteps(steps, device=device)

    text_in = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids.to(device)

    uncond_in = tokenizer(
        [""],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids.to(device)

    if clip_skip > 0:
        cond_out = text_encoder(text_in, output_hidden_states=True)
        uncond_out = text_encoder(uncond_in, output_hidden_states=True)
        idx = -(clip_skip + 1)
        cond = cond_out.hidden_states[idx]
        uncond = uncond_out.hidden_states[idx]
    else:
        cond = text_encoder(text_in)[0]
        uncond = text_encoder(uncond_in)[0]

    h = w = 512
    latent_h = h // 8
    latent_w = w // 8

    for i in range(num_images):
        gen = torch.Generator(device=device).manual_seed(seed + i)

        latents = torch.randn(
            (1, 4, latent_h, latent_w),
            generator=gen,
            device=device,
            dtype=dtype,
        )

        for t in scheduler.timesteps:
            latent_in = latents
            if hasattr(scheduler, "scale_model_input"):
                latent_in = scheduler.scale_model_input(latent_in, t)

            noise_uncond = unet(latent_in, t, encoder_hidden_states=uncond).sample
            noise_text = unet(latent_in, t, encoder_hidden_states=cond).sample
            noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            latents = scheduler.step(noise, t, latents).prev_sample

        imgs = decode_latents_to_pil(vae, latents)
        imgs[0].save(output_dir / f"img_{i}.png")
