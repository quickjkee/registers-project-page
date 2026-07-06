#!/usr/bin/env python3
"""
Dump PCA feature maps for ALL blocks x several timesteps, for pDiT-B/16 without
and with registers, into contact-sheet grids so you can pick the best cell.

Output: /tmp/pca_matrix_noreg.png and /tmp/pca_matrix_reg.png
(rows = blocks 1..12, cols = timesteps)
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

R = "/home/quickjkee/projects/CUR/registers"
HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT_CLASS = 281
TIMES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]   # JiT: t=1 clean (low t = high noise)
SEED = 0

def load_cat(size):
    im = Image.open(os.path.join(HERE, "..", "static/images/candidates/cat.jpg")).convert("RGB")
    w, h = im.size; m = min(w, h)
    return im.crop(((w-m)//2,(h-m)//2,(w-m)//2+m,(h-m)//2+m)).resize((size,size), Image.BICUBIC)

def pca_rgb(patches, grid=16, S=150, smooth=False):
    f = patches - patches.mean(0, keepdim=True)
    U, Sg, V = torch.linalg.svd(f, full_matrices=False)
    proj = f @ V[:3].T
    lo = torch.quantile(proj, 0.02, dim=0); hi = torch.quantile(proj, 0.98, dim=0)
    proj = ((proj - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    img = Image.fromarray((proj.reshape(grid, grid, 3).numpy()*255).astype(np.uint8))
    if smooth:
        # LIGHT smoothing: bilinear to a small intermediate (each token ~4px),
        # then nearest to final — soft token edges, not a heavy blur.
        img = img.resize((grid*4, grid*4), Image.BILINEAR)
        return img.resize((S, S), Image.NEAREST)
    return img.resize((S, S), Image.NEAREST)

def load_jit(with_reg):
    sys.modules.pop("models", None); sys.path.insert(0, f"{R}/generative_models/JiT")
    from collections import OrderedDict
    from model_jit import JiT_B_16
    if with_reg:
        ck = f"{R}/checkpoints/jit/jit_b16_regs/checkpoint-last.pth"; net = JiT_B_16(in_context_len=32, in_context_start=4); reg = 32
    else:
        ck = f"{R}/checkpoints/jit/jit_b16_no_regs/checkpoint-last.pth"; net = JiT_B_16(in_context_len=0, in_context_start=0); reg = 0
    sd = torch.load(ck, map_location="cpu", weights_only=False)['model']
    net.load_state_dict(OrderedDict((k[4:] if k.startswith("net.") else k, v) for k,v in sd.items()), strict=True)
    return net.eval().to(DEVICE), reg

def font(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception: return ImageFont.load_default()

def build(with_reg, out):
    net, reg = load_jit(with_reg)
    depth = len(net.blocks)
    img = load_cat(256)
    x = (torch.from_numpy(np.asarray(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    # capture all blocks at once per timestep
    cells = {}   # (block, ti) -> PIL
    for ti, t in enumerate(TIMES):
        store = {}
        hs = []
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
        for b in range(depth):
            feat = store[b][0]
            drop = max(0, feat.shape[0] - 256)   # registers present only after insertion
            cells[(b, ti)] = pca_rgb(feat[drop:], smooth=SMOOTH)
    # lay out: rows = blocks, cols = timesteps; +1 row/col for labels
    cs = 150; lab = 60
    W = lab + cs*len(TIMES); H = lab + cs*depth
    sheet = Image.new("RGB", (W, H), (255,255,255)); d = ImageDraw.Draw(sheet)
    for ti, t in enumerate(TIMES):
        d.text((lab + ti*cs + cs//2 - 20, 20), f"t={t}", fill=(0,0,0), font=font(20))
    for b in range(depth):
        d.text((8, lab + b*cs + cs//2 - 10), f"blk {b+1}", fill=(0,0,0), font=font(18))
        for ti in range(len(TIMES)):
            sheet.paste(cells[(b, ti)], (lab + ti*cs, lab + b*cs))
    sheet.save(out)
    print("saved", out, sheet.size)

if __name__ == "__main__":
    for SMOOTH in (False, True):
        tag = "smooth" if SMOOTH else "raw"
        build(False, f"/tmp/pca_{tag}_noreg.png")
        torch.cuda.empty_cache()
        build(True, f"/tmp/pca_{tag}_reg.png")
        torch.cuda.empty_cache()
