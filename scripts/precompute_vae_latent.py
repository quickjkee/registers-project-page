#!/usr/bin/env python3
"""Encode the cat image into the SD-VAE latent and save it, so the main
diffusion-attention script doesn't need a working `diffusers` install.
Run inside the pinned venv: /tmp/vae_env/bin/python precompute_vae_latent.py
"""
import os, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL

HERE = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(HERE, "..", "static", "images", "candidates", "cat.jpg")
out = os.path.join(HERE, "..", "static", "images", "attention_diff", "cat_sdvae_latent.pt")
os.makedirs(os.path.dirname(out), exist_ok=True)

im = Image.open(img_path).convert("RGB")
w, h = im.size; m = min(w, h)
im = im.crop(((w-m)//2, (h-m)//2, (w-m)//2+m, (h-m)//2+m)).resize((256, 256), Image.BICUBIC)
x = torch.from_numpy(np.asarray(im)/255.0).permute(2, 0, 1).unsqueeze(0).float()*2 - 1

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").eval()
with torch.no_grad():
    lat = vae.encode(x).latent_dist.sample().mul_(0.18215)
torch.save(lat.cpu(), out)
print("saved latent", tuple(lat.shape), "->", out)
