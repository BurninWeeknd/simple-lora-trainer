from pathlib import Path
import torch
from PIL import Image

def decode_latents_to_pil(vae, latents):
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in images]

def make_add_time_ids(batch_size: int, bucket_res: int, dtype: torch.dtype):
    h = w = int(bucket_res)
    t = torch.tensor([h, w, 0, 0, h, w], dtype=dtype)
    return t.unsqueeze(0).repeat(batch_size, 1)

@torch.no_grad()
def run_sdxl_inference_preview(
    *,
    unet,
    vae,
    scheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    output_dir: Path,
    steps: int,
    seed: int,
    dtype,
    resolution: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    unet_device = next(unet.parameters()).device
    vae_device = next(vae.parameters()).device

    torch.manual_seed(seed)

    latents = torch.randn(
        (1, unet.config.in_channels, resolution // 8, resolution // 8),
        device=unet_device,
        dtype=dtype,
    )

    scheduler.set_timesteps(steps, device=unet_device)

    time_ids = make_add_time_ids(batch_size=1, bucket_res=resolution, dtype=dtype).to(unet_device)

    for t in scheduler.timesteps:
        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds.to(unet_device),
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds.to(unet_device),
                "time_ids": time_ids,
            },
        ).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    imgs = decode_latents_to_pil(vae, latents.to(vae_device))
    imgs[0].save(output_dir / "preview.png")
