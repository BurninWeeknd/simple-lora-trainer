from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import TrainConfig
from ..data import image_transform, apply_caption_options, parse_caption_tags

class SDTrainStep:
    def __init__(
        self,
        *,
        cfg: TrainConfig,
        dataset,
        cached_latents_by_bucket,
        tokenizer,
        text_encoder,
        vae,
        unet,
        scheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.cached = cached_latents_by_bucket
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype

    def __call__(self, batch_indices: list[int], bucket_res: int) -> torch.Tensor:
        tfm_bucket = image_transform(bucket_res)

        images = []
        captions = []

        for idx in batch_indices:
            img_path, cap_path = self.dataset[idx]
            text = Path(cap_path).read_text(encoding="utf-8").strip()
            text = apply_caption_options(text, self.cfg)
            images.append(tfm_bucket(Image.open(img_path).convert("RGB")))
            captions.append(text)

        pixel = torch.stack(images).to(device=self.device, dtype=self.dtype)

        te_device = next(self.text_encoder.parameters()).device
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).input_ids.to(te_device)

        train_clip = (self.cfg.clip_lr is not None) and (float(self.cfg.clip_lr) > 0.0)

        if train_clip:
            if self.cfg.clip_skip > 0:
                out = self.text_encoder(tokens, output_hidden_states=True)
                idx = -(self.cfg.clip_skip + 1)
                enc = out.hidden_states[idx]
            else:
                enc = self.text_encoder(tokens)[0]
        else:
            with torch.no_grad():
                if self.cfg.clip_skip > 0:
                    out = self.text_encoder(tokens, output_hidden_states=True)
                    layer_idx = -(self.cfg.clip_skip + 1)
                    enc = out.hidden_states[layer_idx]
                else:
                    enc = self.text_encoder(tokens)[0]

        if self.cfg.cache_latents:
            assert self.cached is not None
            bucket_cache = self.cached[bucket_res]
            latents = torch.cat([bucket_cache[i] for i in batch_indices], dim=0).to(device=self.device, dtype=self.dtype)
        else:
            with torch.no_grad():
                latents = self.vae.encode(pixel).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        t = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.size(0),),
            device=self.device,
        ).long()

        noisy = self.scheduler.add_noise(latents, noise, t)

        unet_device = next(self.unet.parameters()).device
        enc_unet = enc.to(unet_device)

        pred = self.unet(noisy.to(unet_device), t.to(unet_device), encoder_hidden_states=enc_unet).sample
        loss = F.mse_loss(pred.float().to(self.device), noise.float())

        return loss
