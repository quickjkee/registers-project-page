#!/usr/bin/env python3
"""
PCA feature maps for the interactive slider in the "why registers help" section.
Raw (no smoothing) 16x16 token grids for pDiT-B/16 WITH and WITHOUT registers,
over all blocks x a range of high-noise timesteps.

Output: static/images/pca/{reg,noreg}_b<NN>_t<TT>.png  (NN=block, TT=time index)
        static/images/pca/input.png, manifest_pca.json
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image

R = "/home/quickjkee/projects/CUR/registers"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "static", "images", "pca"))
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMES = [0.05, 0.07, 0.10, 0.12, 0.15]          # high-noise range (JiT: t=1 clean)
BLOCKS = [5, 6, 7]                              # 1-indexed blocks to export
SEED = 0
S = 256
# (image file, ImageNet class) — clear subject on a natural/textured background
IMAGES = [("cat",       281), ("bellpepper", 945), ("vase", 883)]

def load_img(name, size):
    im = Image.open(os.path.join(HERE, "..", f"static/images/candidates/{name}.jpg")).convert("RGB")
    w, h = im.size; m = min(w, h)
    return im.crop(((w-m)//2,(h-m)//2,(w-m)//2+m,(h-m)//2+m)).resize((size,size), Image.BICUBIC)

@torch.no_grad()
def pca_rgb_per_map(feat, grid=16, eps=1e-8):
    """Separate PCA + separate RGB scaling for THIS feature map only.
    feat: [256, D] patch tokens. Matches visualization.ipynb."""
    mean = feat.mean(0, keepdim=True)
    Xc = feat - mean
    U, Sg, Vt = torch.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:3].T                                  # [256, 3]
    zmin = Z.min(0, keepdim=True).values
    zmax = Z.max(0, keepdim=True).values
    rgb = ((Z - zmin) / (zmax - zmin + eps)).clamp(0, 1)
    im = Image.fromarray((rgb.view(grid, grid, 3).numpy()*255).astype(np.uint8))
    return im.resize((S, S), Image.NEAREST)            # nearest, no smoothing

def load_jit(with_reg):
    sys.modules.pop("models", None); sys.path.insert(0, f"{R}/generative_models/JiT")
    from collections import OrderedDict
    from model_jit import JiT_B_16
    if with_reg:
        ck = f"{R}/checkpoints/jit/jit_b16_regs/checkpoint-last.pth"; net = JiT_B_16(in_context_len=32, in_context_start=4)
    else:
        ck = f"{R}/checkpoints/jit/jit_b16_no_regs/checkpoint-last.pth"; net = JiT_B_16(in_context_len=0, in_context_start=0)
    sd = torch.load(ck, map_location="cpu", weights_only=False)['model']
    net.load_state_dict(OrderedDict((k[4:] if k.startswith("net.") else k, v) for k,v in sd.items()), strict=True)
    return net.eval().to(DEVICE)

def features_for(net, x, cls, t):
    """Patch features [256,D] per block at timestep t for image x, class cls."""
    y = torch.tensor([cls], device=DEVICE)
    store = {}; hs = []
    for i, b in enumerate(net.blocks):
        def mk(i):
            def h(m, inp, o): store[i] = (o[0] if isinstance(o, (tuple, list)) else o).detach().float().cpu()
            return h
        hs.append(b.register_forward_hook(mk(i)))
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    noise = torch.randn(x.shape, device=DEVICE, generator=gen)
    z = t * x + (1 - t) * noise
    with torch.no_grad():
        net(z, torch.full((1,), t, device=DEVICE), y)
    for h in hs: h.remove()
    return {b: store[b][0][max(0, store[b][0].shape[0]-256):] for b in store}

def main():
    net_no = load_jit(False); net_re = load_jit(True)
    n = 0
    for name, cls in IMAGES:
        load_img(name, S).save(os.path.join(OUT, f"input_{name}.png"))
        x = (torch.from_numpy(np.asarray(load_img(name, 256))/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
        for ti, t in enumerate(TIMES, start=1):
            fno = features_for(net_no, x, cls, t)
            fre = features_for(net_re, x, cls, t)
            for b in BLOCKS:                     # 1-indexed -> 0-indexed b-1
                pca_rgb_per_map(fno[b-1]).save(os.path.join(OUT, f"{name}_noreg_b{b:02d}_t{ti:02d}.png"))
                pca_rgb_per_map(fre[b-1]).save(os.path.join(OUT, f"{name}_reg_b{b:02d}_t{ti:02d}.png"))
                n += 2
        print(f"{name}: done")
    print(f"rendered {n} cells (per-map PCA)")

    json.dump({"images": [a for a, _ in IMAGES], "blocks": BLOCKS, "timesteps": TIMES},
              open(os.path.join(OUT, "manifest_pca.json"), "w"), indent=2)
    print("DONE ->", OUT)

if __name__ == "__main__":
    main()
