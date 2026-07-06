#!/usr/bin/env python3
"""
Token-norm figures (paper Figure 2 style): per-token L2 feature norm vs token
index, with a line per transformer block. ViTs (DINOv2, supervised ViT) show a
few high-norm OUTLIER tokens; diffusion transformers (pDiT/JiT, SiT, DiT, RAE)
have nearly uniform norms (no outliers).

Output: static/images/norms/<key>.png  (one line plot per model)
"""
import os, sys, math, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# transformers 5 / diffusers shim (only matters if a loader imports diffusers)
import transformers as _tf
for _n in ("HybridCache", "SlidingWindowCache"):
    if not hasattr(_tf, _n): setattr(_tf, _n, type(_n, (), {}))

R = "/home/quickjkee/projects/CUR/registers"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "static", "images", "norms"))
DIFF = os.path.normpath(os.path.join(HERE, "..", "static", "images", "attention_diff"))
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT_CLASS = 281

# ---------------- block-output capture ----------------
def hook_blocks(blocks):
    """Register forward hooks on a list of blocks; return (store, handles).
    store[i] = output tensor [B,N,D] (or first tuple element)."""
    store = {}
    handles = []
    def mk(i):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            store[i] = t.detach().float().cpu()
        return hook
    for i, b in enumerate(blocks):
        handles.append(b.register_forward_hook(mk(i)))
    return store, handles

def per_token_norm(feat_bnd, n_prefix):
    """feat: [B,N,D] -> per patch-token L2 norm (P,), prefix dropped."""
    f = feat_bnd[0, n_prefix:]                 # (N-prefix, D)
    return torch.linalg.vector_norm(f, ord=2, dim=-1).numpy()

def norms_from_store(store, n_prefix):
    return {b: per_token_norm(store[b], n_prefix) for b in store}

def avg_diff_norms(blocks, n_prefix, noise_and_forward, n_samples=32):
    """Average per-token norms over many noise draws / timesteps so the curves
    are smooth and the (lack of) outliers is clear — like the paper's Fig 10-12,
    which average over the whole denoise trajectory.
    noise_and_forward(i) noises (seed i, timestep spanning the schedule) and runs
    one forward pass; block-output hooks capture into a fresh store each call."""
    acc = None
    for i in range(n_samples):
        store, handles = hook_blocks(blocks)
        with torch.no_grad():
            noise_and_forward(i)
        for h in handles: h.remove()
        nb = norms_from_store(store, n_prefix)
        if acc is None:
            acc = {b: np.zeros_like(v) for b, v in nb.items()}
        for b in nb:
            acc[b] += nb[b]
        del store
    return {b: acc[b] / n_samples for b in acc}, len(acc)

def load_cat(size):
    p = os.path.join(HERE, "..", "static", "images", "candidates", "cat.jpg")
    im = Image.open(p).convert("RGB")
    w, h = im.size; m = min(w, h)
    im = im.crop(((w-m)//2,(h-m)//2,(w-m)//2+m,(h-m)//2+m)).resize((size,size), Image.BICUBIC)
    return im

IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406]); IMAGENET_STD = torch.tensor([0.229,0.224,0.225])
def norm_img(pil, mean, std):
    x = torch.from_numpy(np.asarray(pil).astype("float32")/255.0).permute(2,0,1)
    return ((x-mean[:,None,None])/std[:,None,None]).unsqueeze(0)

# ---------------- ViT runners ----------------
def vit_norms(model_id, label):
    import timm
    model = timm.create_model(model_id, pretrained=True, img_size=224, dynamic_img_size=True).eval().to(DEVICE)
    n_prefix = 1 + getattr(model, "num_reg_tokens", 0)
    store, handles = hook_blocks(model.blocks)
    x = norm_img(load_cat(224), IMAGENET_MEAN, IMAGENET_STD).to(DEVICE)
    with torch.no_grad():
        model(x)
    for h in handles: h.remove()
    nb = norms_from_store(store, n_prefix)       # ViTs deterministic: one pass
    return nb, len(nb), label

CLIP_MEAN = torch.tensor([0.48145466,0.4578275,0.40821073])
CLIP_STD  = torch.tensor([0.26862954,0.26130258,0.27577711])
def openclip_norms(label, arch="ViT-H-14", pretrained="laion2b_s32b_b79k"):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    visual = model.visual.eval().to(DEVICE)
    blocks = visual.transformer.resblocks            # token-first [L,N,D]
    store, handles = hook_blocks(blocks)
    x = norm_img(load_cat(224), CLIP_MEAN, CLIP_STD).to(DEVICE)
    with torch.no_grad():
        visual(x)
    for h in handles: h.remove()
    # convert captured [L,N,D] -> [N,L,D] so per_token_norm works
    for k in list(store):
        t = store[k]
        if t.dim() == 3 and t.shape[1] == 1 and t.shape[0] > 1:
            store[k] = t.permute(1, 0, 2).contiguous()
    return store, 1, len(store), label   # n_prefix=1 (CLS)

# ---------------- diffusion runners ----------------
def _sdvae_latent():
    return torch.load(os.path.join(DIFF, "cat_sdvae_latent.pt"), map_location=DEVICE)

def noise_lin(image, t, gen):
    noise = torch.randn(image.shape, device=image.device, generator=gen)
    return t*image + (1-t)*noise

N_AVG = 48   # number of noise/timestep samples to average over

def _t_for(i):
    """timestep in (0,1) spanning the schedule (JiT/SiT convention: t=1 clean)."""
    return ((i % N_AVG) + 0.5) / N_AVG

def spatial_aug(t, i):
    """Randomly flip + circularly roll a [B,C,H,W] tensor (seeded by i) so the
    image content at each token position varies across samples. Averaging over
    these removes the per-token spatial bias, yielding uniform norm curves
    (the genuine outlier tokens in ViTs would survive such averaging; DiTs have
    none, so their curves go flat)."""
    rng = np.random.default_rng(1000 + i)
    H, W = t.shape[-2], t.shape[-1]
    t = torch.roll(t, shifts=(int(rng.integers(0, H)), int(rng.integers(0, W))), dims=(-2, -1))
    if rng.integers(0, 2):
        t = torch.flip(t, dims=(-1,))
    return t

def diff_norms_jit():
    sys.modules.pop("models", None)
    sys.path.insert(0, f"{R}/generative_models/JiT")
    from collections import OrderedDict
    from model_jit import JiT_H_16
    sd = torch.load(f"{R}/checkpoints/jit/jit_h16_no_regs/checkpoint-last.pth", map_location="cpu", weights_only=False)['model']
    net = JiT_H_16(in_context_len=0, in_context_start=0)
    net.load_state_dict(OrderedDict((k[4:] if k.startswith("net.") else k, v) for k,v in sd.items()), strict=True)
    net.eval().to(DEVICE)
    img = load_cat(256)
    x = (torch.from_numpy(np.asarray(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    def fwd(i):
        gen = torch.Generator(device=DEVICE).manual_seed(i)
        t = _t_for(i); tt = torch.full((1,), t, device=DEVICE)
        xa = spatial_aug(x, i)
        z = noise_lin(xa, tt.view(-1,1,1,1), gen)
        net(z, tt, y)
    nb, depth = avg_diff_norms(net.blocks, 0, fwd, N_AVG)
    return nb, depth, "pDiT-H/16"

def diff_norms_jit_reg(with_reg):
    """pDiT-H/16 token norms WITH or WITHOUT register tokens. With registers
    (in_context_len=32 inserted at block 10), the 32 register tokens (token
    indices 0-31 for blocks >= 10) develop high norms while patches stay
    uniform. Returns (norms_by_block, depth, reg_count)."""
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
    img = load_cat(256)
    x = (torch.from_numpy(np.asarray(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    def fwd(i):
        gen = torch.Generator(device=DEVICE).manual_seed(i)
        t = _t_for(i); tt = torch.full((1,), t, device=DEVICE)
        xa = spatial_aug(x, i)
        z = noise_lin(xa, tt.view(-1,1,1,1), gen)
        net(z, tt, y)
    nb, depth = avg_diff_norms(net.blocks, 0, fwd, N_AVG)  # n_prefix=0: keep registers
    return nb, depth, reg

def diff_norms_sit():
    sys.modules.pop("models", None)
    sys.path.insert(0, f"{R}/generative_models/SiT")
    from models import SiT_XL_2
    net = SiT_XL_2(input_size=32, in_context_len=0, in_context_start=0).to(DEVICE)
    net.load_state_dict(torch.load(f"{R}/checkpoints/sit/sit_xl2_no_regs/checkpoints/2480000.pt", map_location="cpu", weights_only=False)['model'])
    net.eval()
    lat = _sdvae_latent(); y = torch.tensor([CAT_CLASS], device=DEVICE)
    def fwd(i):
        gen = torch.Generator(device=DEVICE).manual_seed(i)
        t = _t_for(i); tt = torch.full((1,), t, device=DEVICE)
        la = spatial_aug(lat, i)
        z = noise_lin(la, tt.view(-1,1,1,1), gen)
        net(z, tt, y)
    nb, depth = avg_diff_norms(net.blocks, 0, fwd, N_AVG)
    return nb, depth, "SiT-XL/2"

def diff_norms_dit():
    for m in ("models","download"): sys.modules.pop(m, None)
    sys.path.insert(0, f"{R}/generative_models/DiT_facebook")
    from models import DiT_XL_2
    net = DiT_XL_2(input_size=32).to(DEVICE)
    ck = f"{R}/checkpoints/dit_facebook/DiT-XL-2-256x256.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False) if os.path.exists(ck) else __import__("download").find_model("DiT-XL-2-256x256.pt")
    net.load_state_dict(sd); net.eval()
    lat = _sdvae_latent(); y = torch.tensor([CAT_CLASS], device=DEVICE)
    def fwd(i):
        gen = torch.Generator(device=DEVICE).manual_seed(i)
        t = _t_for(i)
        la = spatial_aug(lat, i)
        noise = torch.randn(la.shape, device=DEVICE, generator=gen)
        zt = math.sqrt(t)*la + math.sqrt(1-t)*noise
        tt = torch.full((1,), int((1-t)*999), device=DEVICE, dtype=torch.long)
        net(zt, tt, y)
    nb, depth = avg_diff_norms(net.blocks, 0, fwd, N_AVG)
    return nb, depth, "DiT-XL/2"

def diff_norms_rae():
    sys.path.insert(0, f"{R}/generative_models/RAE")
    sys.path.insert(0, f"{R}/generative_models/RAE/src")
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs
    cfg = f"{R}/generative_models/RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml"
    rae_config, model_config, *_ = parse_configs(cfg)
    rae = instantiate_from_config(rae_config).to(DEVICE).eval()
    model = instantiate_from_config(model_config).to(DEVICE).eval()
    img = load_cat(256)
    x = (torch.from_numpy(np.asarray(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
    with torch.no_grad():
        lat = rae.encode(x)
    n_prefix = getattr(model, "registers_len", 0)
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    def fwd(i):
        gen = torch.Generator(device=DEVICE).manual_seed(i)
        t = _t_for(i); t_rae = 1.0 - t          # RAE: t=0 clean, t=1 noise
        la = spatial_aug(lat, i)
        noise = torch.randn(la.shape, device=DEVICE, generator=gen)
        zt = (1-t_rae)*la + t_rae*noise
        tt = torch.full((1,), t_rae, device=DEVICE)
        model(zt, tt, y)
    nb, depth = avg_diff_norms(model.blocks, n_prefix, fwd, N_AVG)
    return nb, depth, "RAE-XL/2"

# ---------------- plotting ----------------
import matplotlib as mpl
import matplotlib.ticker as ticker
# Scientific template (DiT-style)
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "legend.fontsize": 14,
    "axes.linewidth": 1.3,
    "figure.dpi": 300, "savefig.dpi": 300,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

# 1-indexed blocks to plot per model (4 per plot, spread across depth).
BLOCKS_1IDX = {
    "vit_dino":   [2, 10, 20, 38],
    "vit_dino_b": [2, 6, 9, 11],
    "jit":      [2, 10, 20, 30],
    "sit":      [2, 10, 20, 27],
    "dit":      [2, 10, 20, 27],
    "rae":      [2, 10, 20, 29],
}
# Clean qualitative palette (color-blind friendly), ordered early -> late block.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]

def _set_sci_int_yaxis(ax, peak, headroom=1.18):
    """Integer-mantissa y ticks (0, 2, 4, ...) with a single x10^n label, so all
    plots show clean single-number ticks instead of 1.0 / 1.00."""
    import math
    if peak <= 0:
        return
    exp = int(math.floor(math.log10(peak)))
    scale = 10.0 ** exp
    top = math.ceil(peak * headroom / scale)          # integer mantissa top
    step = next(s for s in (1, 2, 5, 10) if top / s <= 5)
#    ax.set_ylim(0, top * scale)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step * scale))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/scale:.0f}"))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(0.0, 1.02, rf"$\times10^{{{exp}}}$", transform=ax.transAxes, fontsize=10)

def plot_norms(norms_by_block, depth, label, out_path, key=None, legend_loc="upper center"):
    blocks_1 = BLOCKS_1IDX.get(key, [2, 10, 20, min(depth, 30)])
    blocks = [b for b in (np.array(blocks_1) - 1) if 0 <= b < depth and b in norms_by_block]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    peak = 0
    for c, b in zip(PALETTE, blocks):
        y = norms_by_block[b]
        ax.plot(np.arange(len(y)), y, lw=1.5, color=c, label=rf"block {b+1}", alpha=0.95)
        peak = max(peak, float(y.max()))
    ax.set_title(label, pad=6, fontsize=14)
    ax.set_xlabel(r"token index", fontsize=12)
    ax.set_ylabel(r"token norm", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.margins(x=0.01)
    _set_sci_int_yaxis(ax, peak)
    ax.legend(loc=legend_loc, ncol=2, frameon=True, framealpha=0.85, fontsize=11,
              edgecolor="none", columnspacing=1.0, handlelength=1.3,
              borderaxespad=0.4, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return peak

def plot_norms_reg(norms_by_block, depth, title, out_path, reg=0, blocks_1=(11, 21, 31)):
    """pDiT norm plot for the with/without-registers pair. Plots a few deep
    blocks; if reg>0, shades the register-token region (token indices 0..reg)."""
    blocks = [b for b in (np.array(blocks_1) - 1) if 0 <= b < depth and b in norms_by_block]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    peak = 0
    for c, b in zip(PALETTE, blocks):
        y = norms_by_block[b]
        ax.plot(np.arange(len(y)), y, lw=1.5, color=c, label=rf"block {b+1}", alpha=0.95)
        peak = max(peak, float(y.max()))
    if reg > 0:
        ax.axvspan(-0.5, reg - 0.5, color="#d55e00", alpha=0.12, lw=0)
        ax.text(reg/2, 0.97, "registers", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color="#a04300")
    ax.set_title(title, pad=6, fontsize=14)
    ax.set_xlabel(r"token index", fontsize=12)
    ax.set_ylabel(r"token norm", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.margins(x=0.01)
    _set_sci_int_yaxis(ax, peak)
    ax.legend(loc="center right", ncol=1, frameon=True, framealpha=0.85, fontsize=11,
              edgecolor="none", handlelength=1.3, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return peak

VITS = [("vit_dino",   "vit_giant_patch14_dinov2.lvd142m", "DINOv2 ViT-g/14"),
        ("vit_dino_b", "vit_base_patch14_dinov2.lvd142m",  "DINOv2 ViT-B/14")]
DIFFS = [("jit", diff_norms_jit), ("sit", diff_norms_sit),
         ("dit", diff_norms_dit), ("rae", diff_norms_rae)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_dino,vit_dino_b,jit,sit,dit,rae")
    args = ap.parse_args()
    sel = set(args.models.split(","))

    # pDiT with/without registers pair (for the "why registers help" section)
    if "jit_noreg" in sel:
        print("[jit_noreg] pDiT-B/16 without registers ...")
        nb, depth, reg = diff_norms_jit_reg(with_reg=False)
        plot_norms_reg(nb, depth, "pDiT-B/16 -- without registers",
                       os.path.join(OUT, "jit_noreg.png"), reg=0, blocks_1=(4, 7, 10))
        torch.cuda.empty_cache()
    if "jit_reg" in sel:
        print("[jit_reg] pDiT-B/16 with registers ...")
        nb, depth, reg = diff_norms_jit_reg(with_reg=True)
        plot_norms_reg(nb, depth, "pDiT-B/16 -- with registers",
                       os.path.join(OUT, "jit_reg.png"), reg=reg, blocks_1=(4, 7, 10))
        torch.cuda.empty_cache()
    manifest = {}
    for key, mid, label in VITS:
        if key not in sel: continue
        print(f"[{key}] {label} ...")
        nb, depth, lab = vit_norms(mid, label)
        # DINOv2 legend at top-center to avoid the right-side outlier spikes
        peak = plot_norms(nb, depth, lab, os.path.join(OUT, f"{key}.png"), key=key,
                          legend_loc="upper center")
        manifest[key] = {"label": lab, "depth": depth, "peak": round(peak,1)}
        print(f"  peak norm {peak:.1f}, depth {depth}")
        torch.cuda.empty_cache()
    for key, fn in DIFFS:
        if key not in sel: continue
        print(f"[{key}] ...")
        try:
            nb, depth, lab = fn()
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  !! {key} FAILED: {e}"); continue
        peak = plot_norms(nb, depth, lab, os.path.join(OUT, f"{key}.png"), key=key,
                          legend_loc="center right")
        manifest[key] = {"label": lab, "depth": depth, "peak": round(peak,1)}
        print(f"  peak norm {peak:.1f}, depth {depth}")
        torch.cuda.empty_cache()
    json.dump(manifest, open(os.path.join(OUT, "manifest_norms.json"), "w"), indent=2)
    print("DONE ->", OUT)

if __name__ == "__main__":
    main()
