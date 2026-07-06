#!/usr/bin/env python3
"""
Attention maps (layers x timesteps) for diffusion transformers: JiT/pDiT, SiT,
RAE, DiT. For each model we noise the input image at several diffusion
timesteps, run a forward pass while capturing per-block self-attention (via an
SDPA monkeypatch), and render a heatmap of how much each patch token is
attended (mean over heads and over query tokens).

Output: static/images/attention_diff/<model>_L<li>_T<ti>.png  (+ input.png,
manifest_diff.json with the layer indices and timestep values per model).

The point of the figure: unlike ViTs, these DiTs do NOT show high-norm
background outliers in their attention maps.
"""
import os, sys, math, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# diffusers 0.36 + transformers 5.x: diffusers.loaders.peft imports symbols that
# moved/renamed in transformers 5. Shim the missing names BEFORE importing
# diffusers anywhere (we only need the VAE, which doesn't use them).
import transformers as _tf
for _name in ("HybridCache", "SlidingWindowCache"):
    if not hasattr(_tf, _name):
        setattr(_tf, _name, type(_name, (), {}))

R = "/home/quickjkee/projects/CUR/registers"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "static", "images", "attention_diff"))
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- attention capture ----------------------
_captured = []
_orig_sdpa = F.scaled_dot_product_attention

def _sdpa_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    d = q.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(d)
    attn = (q.float() @ k.float().transpose(-2, -1)) * s
    if attn_mask is not None and not isinstance(attn_mask, bool):
        attn = attn + attn_mask
    attn = attn.softmax(-1)
    _captured.append(attn[0].detach().to("cpu"))        # keep heads -> (H,N,N)
    return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale)

class capture:
    def __enter__(self):
        _captured.clear(); F.scaled_dot_product_attention = _sdpa_capture; return self
    def __exit__(self, *a):
        F.scaled_dot_product_attention = _orig_sdpa

# ---------------------- rendering ----------------------
def turbo(x):
    x = np.clip(x, 0, 1)
    return np.stack([np.clip(1.4*x-0.2,0,1), np.clip(1.6*x*(1-0.7*x)+0.1,0,1), np.clip(1.0-1.3*x,0,1)], -1)

def head_to_grid(attn_hnn, head, n_prefix, grid):
    """attn_hnn: (H,N,N) for one sample. For the given head, map = mean over
    query rows of attention received by each patch token. Drop prefix tokens."""
    a = attn_hnn[head, n_prefix:, n_prefix:]      # patches x patches
    received = a.mean(0)                          # (P,)
    gh, gw = grid
    received = received[:gh*gw]
    return received.reshape(gh, gw).numpy()

def best_head_grid(attn_hnn, n_prefix, grid, mask, border_max=0.05):
    """Pick the head whose patch-attention best concentrates on the object,
    among heads that are NOT border-biased. Returns (heat, head, focus, border).
    Score = object_focus - 2*max(0, border) so border-heavy heads are penalized.
    """
    H = attn_hnn.shape[0]
    best = (-1e9, 0, None, 1.0)
    for h in range(H):
        g = head_to_grid(attn_hnn, h, n_prefix, grid)
        f = object_focus(g, mask)
        bf = border_frac(g, grid)
        score = f - 2.0 * max(0.0, bf)
        if score > best[0]:
            best = (score, h, g, bf)
    _, hd, g, bf = best
    return g, hd, object_focus(g, mask), bf

def render(img_rgb, heat, S=256, gamma=1.6, lo_pct=40, hi_pct=99.5):
    from PIL import Image as I
    lo, hi = np.percentile(heat, lo_pct), np.percentile(heat, hi_pct)
    h0 = np.clip((heat - lo) / (hi - lo + 1e-6), 0, 1) ** gamma
    h = np.asarray(I.fromarray((h0*255).astype(np.uint8)).resize((S, S), I.BILINEAR)).astype(float)/255.0
    if img_rgb.shape[0] != S:
        img_rgb = np.asarray(I.fromarray((img_rgb*255).astype(np.uint8)).resize((S,S), I.BICUBIC)).astype(float)/255.0
    gray = img_rgb.mean(-1, keepdims=True)
    bg = np.repeat(0.28 + 0.40*gray, 3, -1)
    a = np.clip(h*1.1, 0, 1)[..., None]
    rgb = np.clip(bg*(1-a) + turbo(h)*a, 0, 1)
    return I.fromarray((rgb*255).astype(np.uint8))

def object_mask_grid(grid=(16, 16)):
    """Binary mask (gh,gw) of the main object (cat) via DeepLabV3, downsampled
    to the token grid. Falls back to a center prior if segmentation fails."""
    gh, gw = grid
    try:
        import torchvision
        from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
        w = DeepLabV3_ResNet101_Weights.DEFAULT
        seg = deeplabv3_resnet101(weights=w).eval().to(DEVICE)
        im = load_cat(256)
        x = w.transforms()(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = seg(x)["out"][0]            # (C,H,W)
        pred = out.argmax(0).cpu().numpy()    # VOC classes; cat=8, dog=12, etc.
        fg = (pred != 0).astype(float)        # any non-background
        from PIL import Image as I
        m = np.asarray(I.fromarray((fg * 255).astype(np.uint8)).resize((gw, gh), I.BILINEAR)) / 255.0
        m = (m > 0.4).astype(float)
        if m.sum() >= 3:
            print(f"  object mask: {int(m.sum())}/{gh*gw} tokens (DeepLab)")
            return m
    except Exception as e:
        print("  object mask: deeplab failed, using center prior:", e)
    yy, xx = np.mgrid[0:gh, 0:gw]
    m = (((xx - gw/2)/(gw*0.32))**2 + ((yy - gh/2)/(gh*0.36))**2) < 1
    return m.astype(float)

def _border_mask(grid, width=1):
    gh, gw = grid
    m = np.zeros((gh, gw), dtype=float)
    m[:width, :] = 1; m[-width:, :] = 1; m[:, :width] = 1; m[:, -width:] = 1
    return m

def object_focus(heat, mask):
    """Concentration of attention on the object, corrected for the mask's area
    so a uniform map scores ~0. = (mass inside mask / total) - (mask area frac).
    Positive => map is peaked on the object beyond chance; higher is better."""
    h = np.clip(heat, 0, None)
    tot = h.sum() + 1e-9
    inside = (h * mask).sum() / tot
    area = mask.mean() + 1e-9                # fraction of grid the object covers
    return float(inside - area)

def border_frac(heat, grid, width=1):
    """Fraction of attention mass on the outer border ring (above its area)."""
    h = np.clip(heat, 0, None)
    bm = _border_mask(grid, width)
    frac = (h * bm).sum() / (h.sum() + 1e-9)
    return float(frac - bm.mean())           # >0 means border-biased

def load_cat(size):
    p = os.path.join(HERE, "..", "static", "images", "candidates", "cat.jpg")
    im = Image.open(p).convert("RGB")
    w, h = im.size; m = min(w, h)
    im = im.crop(((w-m)//2, (h-m)//2, (w-m)//2+m, (h-m)//2+m)).resize((size, size), Image.BICUBIC)
    return im

# ---------------------- model runners ----------------------
# Each returns: attn_per_t = list over timesteps of [list over blocks of (N,N) tensor],
#               n_prefix, grid (gh,gw)
TIMESTEPS = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]   # high -> low noise (1=clean)
CAT_CLASS = 281                          # ImageNet tabby cat

def noise_lin(image, t, gen):
    noise = torch.randn(image.shape, device=image.device, generator=gen)
    return t * image + (1 - t) * noise   # JiT/SiT flow convention (t=1 -> image)

def run_jit():
    sys.path.insert(0, f"{R}/generative_models/JiT")
    from collections import OrderedDict
    from model_jit import JiT_H_16
    sd = torch.load(f"{R}/checkpoints/jit/jit_h16_no_regs/checkpoint-last.pth", map_location="cpu", weights_only=False)['model']
    net = JiT_H_16(in_context_len=0, in_context_start=0)
    nsd = OrderedDict((k[4:] if k.startswith("net.") else k, v) for k, v in sd.items())
    net.load_state_dict(nsd, strict=True); net.eval().to(DEVICE)
    img = load_cat(256)
    x = (torch.from_numpy(np.asarray(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    per_t = []
    for t in TIMESTEPS:
        tt = torch.full((1,), t, device=DEVICE)
        z = noise_lin(x, tt.view(-1,1,1,1), gen)
        with torch.no_grad(), capture():
            net(z, tt, y)
        per_t.append([a for a in _captured])   # one (H,N,N) per block
    return per_t, 0, (16,16)

def _sdvae_latent():
    p = os.path.join(OUT, "cat_sdvae_latent.pt")
    if not os.path.exists(p):
        raise FileNotFoundError("Run precompute_vae_latent.py in the venv first: " + p)
    return torch.load(p, map_location=DEVICE)

def run_sit():
    sys.modules.pop("models", None)
    sys.path.insert(0, f"{R}/generative_models/SiT")
    from models import SiT_XL_2
    net = SiT_XL_2(input_size=32, in_context_len=0, in_context_start=0).to(DEVICE)
    sd = torch.load(f"{R}/checkpoints/sit/sit_xl2_no_regs/checkpoints/2480000.pt", map_location="cpu", weights_only=False)['model']
    net.load_state_dict(sd); net.eval()
    lat = _sdvae_latent()
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    per_t = []
    for t in TIMESTEPS:
        tt = torch.full((1,), t, device=DEVICE)
        z = noise_lin(lat, tt.view(-1,1,1,1), gen)
        with torch.no_grad(), capture():
            net(z, tt, y)
        per_t.append([a for a in _captured])
    return per_t, 0, (16,16)

def run_dit():
    # SiT also has a top-level `models` module; drop any cached one so DiT's
    # own models.py is imported from its directory.
    for m in ("models", "download"):
        sys.modules.pop(m, None)
    sys.path.insert(0, f"{R}/generative_models/DiT_facebook")
    from models import DiT_XL_2
    net = DiT_XL_2(input_size=32).to(DEVICE)
    ckpt = f"{R}/checkpoints/dit_facebook/DiT-XL-2-256x256.pt"
    if not os.path.exists(ckpt):
        from download import find_model          # downloads if missing
        sd = find_model("DiT-XL-2-256x256.pt")
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(sd); net.eval()
    lat = _sdvae_latent()
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    # DiT uses discrete timesteps 0..999 (DDPM). Map our fractions to that range.
    per_t = []
    for t in TIMESTEPS:
        noise = torch.randn(lat.shape, device=DEVICE, generator=gen)
        # forward-noise via fraction (approx): blend, then pass discrete t
        zt = math.sqrt(t)*lat + math.sqrt(1-t)*noise
        tt = torch.full((1,), int((1-t)*999), device=DEVICE, dtype=torch.long)
        with torch.no_grad(), capture():
            net(zt, tt, y)
        per_t.append([a for a in _captured])
    return per_t, 0, (16,16)

def run_rae():
    sys.path.insert(0, f"{R}/generative_models/RAE")
    sys.path.insert(0, f"{R}/generative_models/RAE/src")
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs
    # Original XL config from the repo. instantiate_from_config loads the ckpt
    # baked into the config (`ckpt:`), so we must NOT override weights here.
    cfg = f"{R}/generative_models/RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml"
    rae_config, model_config, *_ = parse_configs(cfg)
    rae = instantiate_from_config(rae_config).to(DEVICE).eval()
    model = instantiate_from_config(model_config).to(DEVICE).eval()
    img = load_cat(256)
    x = (torch.from_numpy(np.asarray(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)*2-1)
    with torch.no_grad():
        lat = rae.encode(x)
    y = torch.tensor([CAT_CLASS], device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    n_prefix = getattr(model, "registers_len", 0)
    noise = torch.randn(lat.shape, device=DEVICE, generator=gen)
    per_t = []
    for t in TIMESTEPS:
        # RAE Linear transport: xt = (1-t)*latent + t*noise, t in [0,1] passed
        # directly to the model. NOTE t=0 is clean, t=1 is pure noise here
        # (opposite of the JiT convention). We invert so our shared TIMESTEPS
        # axis keeps "lower = more noise" consistent across models.
        t_rae = 1.0 - t
        zt = (1 - t_rae) * lat + t_rae * noise
        tt = torch.full((1,), t_rae, device=DEVICE)
        with torch.no_grad(), capture():
            model(zt, tt, y)
        per_t.append([a for a in _captured])
    return per_t, n_prefix, (16,16)

RUNNERS = {"jit": ("pDiT (JiT-H/16)", run_jit),
           "sit": ("SiT-XL/2", run_sit),
           "dit": ("DiT-XL/2", run_dit),
           "rae": ("RAE (DiT-DH-XL)", run_rae)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="jit,sit,dit,rae")
    ap.add_argument("--nlayers", type=int, default=5, help="max layers to keep")
    ap.add_argument("--focus-thr", type=float, default=0.10,
                    help="min object-concentration (above-chance) of the best head to keep a layer")
    ap.add_argument("--border-max", type=float, default=0.03,
                    help="max above-chance border mass allowed (reject border-focused maps)")
    args = ap.parse_args()

    load_cat(256).save(os.path.join(OUT, "input.png"))
    img_disp = np.asarray(load_cat(256)).astype(float)/255.0
    mask = object_mask_grid((16, 16))

    manifest = {"timesteps": [], "models": {}}
    chosen_ts = None
    for key in args.models.split(","):
        key = key.strip()
        label, fn = RUNNERS[key]
        print(f"[{key}] {label} ...")
        try:
            per_t, n_prefix, grid = fn()
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  !! {key} FAILED: {e}"); continue
        depth = len(per_t[0])

        # For every (block, timestep), pick the HEAD that most attends to the
        # object (instead of averaging heads, which blends in background/sink
        # heads). Score by that head's object concentration.
        heats = {}   # (b,ti) -> grid heat (best head)
        heads = {}   # (b,ti) -> chosen head index
        focus = np.zeros((depth, len(TIMESTEPS)))
        bord = np.zeros((depth, len(TIMESTEPS)))
        for ti in range(len(TIMESTEPS)):
            for b in range(depth):
                h, hd, f, bf = best_head_grid(per_t[ti][b], n_prefix, grid, mask)
                heats[(b, ti)] = h
                heads[(b, ti)] = hd
                focus[b, ti] = f
                bord[b, ti] = bf

        # Pick nlayers layers with a STRICT check, evaluated per layer (mean
        # over timesteps): high object-focus AND low border bias. Reject any
        # layer that is border-biased or below the focus threshold.
        mfocus = focus.mean(1)
        mbord = bord.mean(1)
        ok = [b for b in range(depth)
              if mfocus[b] >= args.focus_thr and mbord[b] <= args.border_max]
        ok.sort(key=lambda b: mfocus[b] - 1.5 * max(0.0, mbord[b]), reverse=True)
        keep_layers = sorted(ok[:args.nlayers])
        if len(keep_layers) < args.nlayers:    # relax border if too few pass
            extra = sorted([b for b in range(depth) if b not in keep_layers],
                           key=lambda b: mfocus[b], reverse=True)
            keep_layers = sorted(keep_layers + extra[:args.nlayers - len(keep_layers)])
        keep_t = list(range(len(TIMESTEPS)))

        for col, b in enumerate(keep_layers, start=1):
            for row, ti in enumerate(keep_t, start=1):
                render(img_disp, heats[(b, ti)]).save(os.path.join(OUT, f"{key}_L{col:02d}_T{row:02d}.png"))
        manifest["models"][key] = {"label": label, "depth": depth,
                                   "layers": [int(b+1) for b in keep_layers], "nlayers": len(keep_layers),
                                   "timesteps": [TIMESTEPS[ti] for ti in keep_t], "ntimesteps": len(keep_t),
                                   "n_prefix": int(n_prefix),
                                   "focus": [round(float(focus[b].mean()), 3) for b in keep_layers],
                                   "border": [round(float(bord[b].mean()), 3) for b in keep_layers],
                                   "heads": [[int(heads[(b, ti)]) for ti in keep_t] for b in keep_layers]}
        # use the first model's kept timesteps as the shared axis label set
        if chosen_ts is None:
            chosen_ts = [TIMESTEPS[ti] for ti in keep_t]
        print(f"  {key}: depth={depth}, kept layers={[b+1 for b in keep_layers]} "
              f"x {len(keep_t)} timesteps={[round(TIMESTEPS[ti],2) for ti in keep_t]}")
        torch.cuda.empty_cache()
    manifest["timesteps"] = chosen_ts or TIMESTEPS

    with open(os.path.join(OUT, "manifest_diff.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("DONE ->", OUT)

if __name__ == "__main__":
    main()
