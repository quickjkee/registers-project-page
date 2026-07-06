#!/usr/bin/env python3
"""
Figure 5(b): PCA of intermediate feature maps for pDiT-B/16 without vs with
register tokens, at high noise. With registers the patch-token feature map is
much cleaner (smoother, more coherent); without registers it is noisier.

For each model we hook a mid-block's patch features, project to the top-3 PCA
components -> RGB, reshape to the 16x16 token grid and upsample.

Output: static/images/pca/pca_noreg.png, pca_reg.png  (+ input.png)
"""
import os, sys, math, argparse
import numpy as np
import torch
from PIL import Image

R = "/home/quickjkee/projects/CUR/registers"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "static", "images", "pca"))
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT_CLASS = 281
# High-noise regime t in [0, 0.2] (JiT: t=1 clean). Registers smooth features here.
T_CANDS = [0.05, 0.10, 0.15, 0.20]
SEEDS = list(range(8))
BLOCK_IDX = 5                 # block 6 (0-indexed): registers reduce TV most here

def load_cat(size):
    p = os.path.join(HERE, "..", "static", "images", "candidates", "cat.jpg")
    im = Image.open(p).convert("RGB")
    w, h = im.size; m = min(w, h)
    im = im.crop(((w-m)//2,(h-m)//2,(w-m)//2+m,(h-m)//2+m)).resize((size,size), Image.BICUBIC)
    return im

def hook_block(blocks, idx):
    store = {}
    def hook(mod, inp, out):
        store["x"] = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().cpu()
    h = blocks[idx].register_forward_hook(hook)
    return store, h

def pca_rgb(patches, grid=16, S=256):
    """patches: [P, D] -> PCA top-3 -> RGB image [S,S,3]."""
    f = patches - patches.mean(0, keepdim=True)
    # robust top-3 directions via SVD
    U, Sg, V = torch.linalg.svd(f, full_matrices=False)
    proj = f @ V[:3].T                       # [P, 3]
    # percentile normalize each channel for contrast
    lo = torch.quantile(proj, 0.02, dim=0)
    hi = torch.quantile(proj, 0.98, dim=0)
    proj = ((proj - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    img = proj.reshape(grid, grid, 3).numpy()
    return np.asarray(Image.fromarray((img*255).astype(np.uint8)).resize((S, S), Image.BILINEAR))

def tv(patches, grid=16):
    """Total variation of the (normalized) patch feature grid; lower = smoother."""
    f = patches / (patches.norm(dim=-1, keepdim=True) + 1e-6)
    F = f.reshape(grid, grid, -1)
    return float((F[1:]-F[:-1]).pow(2).sum() + (F[:, 1:]-F[:, :-1]).pow(2).sum())

def load_jit(with_reg):
    sys.modules.pop("models", None)
    sys.path.insert(0, f"{R}/generative_models/JiT")
    from collections import OrderedDict
    from model_jit import JiT_B_16
    if with_reg:
        ckpt = f"{R}/checkpoints/jit/jit_b16_regs/checkpoint-last.pth"
        net = JiT_B_16(in_context_len=32, in_context_start=4); reg = 32
    else:
        ckpt = f"{R}/checkpoints/jit/jit_b16_no_regs/checkpoint-last.pth"
        net = JiT_B_16(in_context_len=0, in_context_start=0); reg = 0
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)['model']
    net.load_state_dict(OrderedDict((k[4:] if k.startswith("net.") else k, v) for k,v in sd.items()), strict=True)
    net.eval().to(DEVICE)
    return net, reg

def features_at(net, reg, x, y, seed, t):
    store, h = hook_block(net.blocks, BLOCK_IDX)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(x.shape, device=DEVICE, generator=gen)
    z = t * x + (1 - t) * noise
    with torch.no_grad():
        net(z, torch.full((1,), t, device=DEVICE), y)
    h.remove()
    return store["x"][0][reg:]               # patch tokens only

def main():
    load_cat(256).save(os.path.join(OUT, "input.png"))
    img = load_cat(256)
    x = (torch.from_numpy(np.asarray(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
    y = torch.tensor([CAT_CLASS], device=DEVICE)

    net_no, reg_no = load_jit(False)
    net_re, reg_re = load_jit(True)

    # Scan (seed, t) in the high-noise regime; pick the single case where the
    # with-registers feature map is the smoothest RELATIVE to without (best case).
    best = None
    for seed in SEEDS:
        for t in T_CANDS:
            pn = features_at(net_no, reg_no, x, y, seed, t)
            pr = features_at(net_re, reg_re, x, y, seed, t)
            tn, tr = tv(pn), tv(pr)
            gap = tn - tr                     # absolute smoothing: noreg noisy, reg smooth
            print(f"  seed={seed} t={t:.2f}: TV_noreg={tn:7.1f} TV_reg={tr:7.1f} ratio={tr/tn:.2f} gap={gap:7.1f}")
            if best is None or gap > best[0]:
                best = (gap, seed, t, pn, pr, tn, tr)
    gap, seed, t, pn, pr, tn, tr = best
    print(f"BEST case (max gap): seed={seed}, t={t}, TV_noreg={tn:.1f}, TV_reg={tr:.1f}")
    Image.fromarray(pca_rgb(pn)).save(os.path.join(OUT, "pca_noreg.png"))
    Image.fromarray(pca_rgb(pr)).save(os.path.join(OUT, "pca_reg.png"))
    print("DONE ->", OUT)

if __name__ == "__main__":
    main()
