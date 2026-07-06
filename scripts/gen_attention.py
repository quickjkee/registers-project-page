#!/usr/bin/env python3
"""
Generate real CLS->patch attention maps per transformer block for several
ViT-family models, overlaid on an input image.

Outputs: static/images/attention/{model}_block{NN}.png  and  input.png

Models (all ~24-block ViT-L):
  vit       timm  vit_large_patch16_224.augreg_in21k_ft_in1k
  deit      timm  deit3_large_patch16_224.fb_in22k_ft_in1k
  openclip  open_clip ViT-L-14 (laion2b_s32b_b82k)
  dino      timm  vit_large_patch14_dinov2.lvd142m

Method: monkeypatch torch.nn.functional.scaled_dot_product_attention to
capture attention weights, run a forward pass, and for each block take the
attention FROM the class/register-free CLS token TO the patch tokens,
averaged over heads, reshaped to the patch grid, upsampled to image size.
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "static", "images", "attention"))
CAND = os.path.normpath(os.path.join(HERE, "..", "static", "images", "candidates"))
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------- attention capture -----------------------
_captured = []           # list of attn weight tensors, one per SDPA call
_orig_sdpa = F.scaled_dot_product_attention

def _sdpa_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                  scale=None, **kw):
    # q,k: (B, heads, N, d). Recompute weights for capture, but still return
    # the real SDPA output so the forward pass is unchanged.
    d = q.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(d)
    attn = (q.float() @ k.float().transpose(-2, -1)) * s
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(-1)
    _captured.append(attn.detach().to("cpu"))
    return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale)

# --------------------------- helpers -----------------------------
def to_grid_map(attn, n_prefix, grid_hw):
    """attn: (B, heads, N, N) or (heads, N, N). CLS row -> patches, avg heads."""
    if attn.dim() == 4:
        attn = attn[0]                     # drop batch -> (heads, N, N)
    a = attn.mean(0)                       # (N, N) avg over heads
    cls_row = a[0, n_prefix:]              # CLS -> patch tokens
    gh, gw = grid_hw
    cls_row = cls_row[:gh * gw]
    return cls_row.reshape(gh, gw).numpy()

def turbo(x):
    x = np.clip(x, 0, 1)
    r = np.clip(1.4 * x - 0.2, 0, 1)
    g = np.clip(1.6 * x * (1 - 0.7 * x) + 0.1, 0, 1)
    b = np.clip(1.0 - 1.3 * x, 0, 1)
    return np.stack([r, g, b], -1)

def background_grid(pil, gh, gw):
    """Return a (gh, gw) float {0,1} mask that is 1 on BACKGROUND cells and 0 on
    the main object, using DeepLabV3 semantic segmentation (VOC: class 0 = bg).
    A grid cell counts as object if >30% of its pixels are foreground."""
    from torchvision.models.segmentation import deeplabv3_resnet101
    import torchvision.transforms.functional as TF
    seg = deeplabv3_resnet101(weights="DEFAULT").eval().to(DEVICE)
    x = TF.to_tensor(pil)
    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = seg(x)["out"][0].argmax(0).cpu().numpy()       # HxW class ids
    fg = (pred != 0).astype(float)
    H, W = fg.shape
    obj = np.zeros((gh, gw))
    for r in range(gh):
        for c in range(gw):
            cell = fg[r*H//gh:(r+1)*H//gh, c*W//gw:(c+1)*W//gw]
            obj[r, c] = cell.mean()
    return (obj <= 0.30).astype(float)                        # 1 where background

def persistent_outlier_mask(heats, bg_mask=None, z_thresh=4.0, min_frac=0.33,
                            persist_frac=0.4):
    """Outlier tokens are STATIC artifacts: high-norm spikes that recur in the
    same location across many layers (not transient object attention). Given the
    list of per-layer heatmaps, flag a token if it is a per-layer spike (robust
    median/MAD z-score AND a fraction of that layer's peak) in at least
    persist_frac of the layers. If bg_mask is given, restrict outliers to
    BACKGROUND cells (never mark the main object). Returns (gh, gw) float {0,1}."""
    gh, gw = heats[0].shape
    counts = np.zeros(gh * gw)
    for heat in heats:
        flat = heat.flatten().astype(float)
        med = np.median(flat)
        mad = np.median(np.abs(flat - med)) + 1e-9
        z = (flat - med) / (1.4826 * mad)
        mx = flat.max() + 1e-9
        counts += ((z >= z_thresh) & (flat >= min_frac * mx)).astype(float)
    persist_min = max(3, int(round(persist_frac * len(heats))))
    mask = (counts >= persist_min).astype(float).reshape(gh, gw)
    if bg_mask is not None:
        mask = mask * bg_mask                                  # background only
    return mask

def render(img_rgb, heat, S=256, gamma=2.2, lo_pct=55, hi_pct=99.5,
           outlier_mask=None):
    """img_rgb: HxWx3 float [0,1]; heat: ghxgw float.
    Normal attention is drawn with the turbo colormap (original look); tokens in
    outlier_mask are recoloured red, with intensity gated by the CURRENT layer's
    attention so a static artifact only lights up red on the layers where it is
    actually active.
    """
    from PIL import Image as I
    gh, gw = heat.shape
    # contrast stretch on the raw (low-res) map BEFORE upsampling
    lo = np.percentile(heat, lo_pct)
    hi = np.percentile(heat, hi_pct)
    h0 = np.clip((heat - lo) / (hi - lo + 1e-6), 0, 1)
    h0 = h0 ** gamma                                   # suppress mid, keep spikes
    h = I.fromarray((h0 * 255).astype(np.uint8)).resize((S, S), I.BILINEAR)
    h = np.asarray(h).astype(float) / 255.0
    # match overlay base to render size S
    if img_rgb.shape[0] != S:
        img_rgb = np.asarray(I.fromarray((img_rgb * 255).astype(np.uint8)).resize((S, S), I.BICUBIC)).astype(float) / 255.0
    gray = img_rgb.mean(-1, keepdims=True)
    bg = np.repeat(0.28 + 0.40 * gray, 3, -1)          # slightly darker base -> hotspots pop
    col = turbo(h)
    if outlier_mask is not None and outlier_mask.any():
        m = I.fromarray((outlier_mask * 255).astype(np.uint8)).resize((S, S), I.BILINEAR)
        m = (np.asarray(m).astype(float) / 255.0)[..., None]
        red = np.array([0.93, 0.11, 0.11])
        col = col * (1 - m) + red * m                  # outlier tokens -> red
    a = np.clip(h * 1.15, 0, 1)[..., None]             # alpha = current attention
    rgb = np.clip(bg * (1 - a) + col * a, 0, 1)
    # No burned-in label: the model name + layer are shown in the HTML caption.
    return I.fromarray((rgb * 255).astype(np.uint8))

def load_image(path, S=224):
    im = Image.open(path).convert("RGB")
    # center crop to square then resize
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    return im.resize((S, S), Image.BICUBIC)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

def normalize(pil, mean, std, S):
    x = torch.from_numpy(np.asarray(pil).astype("float32") / 255.0).permute(2, 0, 1)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return x.unsqueeze(0)

# --------------------------- models ------------------------------
def run_timm(name, model_id, pil224, mean, std, S, label, blocks_dir):
    import timm
    try:
        model = timm.create_model(model_id, pretrained=True,
                                  img_size=pil224.size[0], dynamic_img_size=True
                                  ).eval().to(DEVICE)
    except TypeError:
        model = timm.create_model(model_id, pretrained=True).eval().to(DEVICE)
    n_prefix = 1 + getattr(model, "num_reg_tokens", 0)  # cls (+ regs if any)
    # patch grid
    patch = model.patch_embed.patch_size[0]
    grid = (pil224.size[1] // patch, pil224.size[0] // patch)
    x = normalize(pil224, mean, std, S).to(DEVICE)
    _captured.clear()
    F.scaled_dot_product_attention = _sdpa_capture
    with torch.no_grad():
        model(x)
    F.scaled_dot_product_attention = _orig_sdpa
    print(f"  {name}: captured {len(_captured)} attn layers, grid={grid}, n_prefix={n_prefix}")
    return _captured.copy(), n_prefix, grid

def run_openclip(pil224, S, label, arch="ViT-H-14", pretrained="laion2b_s32b_b79k"):
    import open_clip
    import torch.nn as nn
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    visual = model.visual.eval().to(DEVICE)
    patch = 14
    grid = (pil224.size[1] // patch, pil224.size[0] // patch)
    x = normalize(pil224, CLIP_MEAN, CLIP_STD, S).to(DEVICE)

    # open_clip uses nn.MultiheadAttention. Wrap its forward to force weight
    # output (per-head) and capture it. open_clip calls attn with batch_first?
    # ViT residual blocks call self.attn(x, x, x, need_weights=False, ...).
    _captured.clear()
    orig_fwd = nn.MultiheadAttention.forward
    def patched(self, query, key, value, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = False
        out, w = orig_fwd(self, query, key, value, **kw)
        if w is not None:
            _captured.append(w.detach().to("cpu"))   # (B, heads, N, N)
        return out, w
    nn.MultiheadAttention.forward = patched
    try:
        with torch.no_grad():
            visual(x)
    finally:
        nn.MultiheadAttention.forward = orig_fwd
    print(f"  openclip: captured {len(_captured)} attn layers, grid={grid}")
    return _captured.copy(), 1, grid

# ----------------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--start", type=int, default=10, help="first layer to export (1-indexed)")
    ap.add_argument("--end", type=int, default=32, help="last layer to export (capped at depth)")
    ap.add_argument("--only", default=None, help="comma-separated model keys to (re)generate; default all")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    S14, S16 = 224, 224
    pil16 = load_image(args.image, S16)
    pil14 = load_image(args.image, 224)  # /14 at 224 -> 16x16 grid

    # save input (256px for display)
    Image.open(args.image).convert("RGB").resize((256, 256), Image.BICUBIC).save(
        os.path.join(OUT, "input.png"))

    img16 = np.asarray(pil16).astype(float) / 255.0
    img14 = np.asarray(pil14).astype(float) / 255.0

    # Background mask (16x16): outliers are only allowed on background cells, so
    # the main object (e.g. the cat) is never marked red.
    print("[seg] computing background mask...")
    bg16 = background_grid(pil14, 16, 16)
    print(f"  background cells: {int(bg16.sum())}/256")

    # ViT-Huge / giant class models (patch-14 -> 16x16 grid at 224).
    jobs = [
        # --- without registers: H/giant ViT family ---
        ("vit", "vit_large_patch14_dinov2.lvd142m", pil14, IMAGENET_MEAN, IMAGENET_STD, 224, "DINOv2-L/14", img14, "timm"),
        ("deit", "deit3_huge_patch14_224.fb_in22k_ft_in1k", pil14, IMAGENET_MEAN, IMAGENET_STD, 224, "DeiT-III-H/14", img14, "timm"),
        ("dino", "vit_giant_patch14_dinov2.lvd142m", pil14, IMAGENET_MEAN, IMAGENET_STD, 224, "DINOv2-g/14", img14, "timm"),
        ("openclip", "ViT-H-14:laion2b_s32b_b79k", pil14, CLIP_MEAN, CLIP_STD, 224, "OpenCLIP-H/14", img14, "openclip"),
        # --- with registers: DINOv2 family (S/B/L/g), all trained with 4 reg tokens ---
        ("reg_dino_s", "vit_small_patch14_reg4_dinov2.lvd142m", pil14, IMAGENET_MEAN, IMAGENET_STD, 224, "DINOv2-S/14 (reg)", img14, "timm"),
        ("reg_dino_b", "vit_base_patch14_reg4_dinov2.lvd142m", pil14, IMAGENET_MEAN, IMAGENET_STD, 224, "DINOv2-B/14 (reg)", img14, "timm"),
        ("reg_dino_l", "vit_large_patch14_reg4_dinov2.lvd142m", pil14, IMAGENET_MEAN, IMAGENET_STD, 224, "DINOv2-L/14 (reg)", img14, "timm"),
        ("reg_dino_g", "vit_giant_patch14_reg4_dinov2.lvd142m", pil14, IMAGENET_MEAN, IMAGENET_STD, 224, "DINOv2-g/14 (reg)", img14, "timm"),
    ]

    import json
    # When regenerating a subset, start from the existing manifest so untouched
    # models keep their entries.
    manifest = {"models": {}}
    mpath = os.path.join(OUT, "manifest.json")
    if only and os.path.exists(mpath):
        manifest = json.load(open(mpath))
        manifest.setdefault("models", {})

    for name, mid, pil, mean, std, S, label, img, kind in jobs:
        if only and name not in only:
            continue
        print(f"[{name}] loading...")
        try:
            if kind == "timm":
                caps, n_prefix, grid = run_timm(name, mid, pil, mean, std, S, label, OUT)
            else:
                arch, pre = (mid.split(":", 1) if mid and ":" in mid
                             else ("ViT-H-14", "laion2b_s32b_b79k"))
                caps, n_prefix, grid = run_openclip(pil, S, label, arch=arch, pretrained=pre)
        except Exception as e:
            print(f"  !! {name} FAILED: {e}")
            continue
        L = len(caps)

        # Always emit N = end-start+1 maps so every model shares one slider,
        # regardless of depth. Sample N layer indices evenly across the model's
        # [start, min(end, depth)] range (shallow models repeat some layers).
        N = args.end - args.start + 1
        lo = max(1, args.start)
        hi = min(args.end, L)
        keep = [int(round(x)) for x in np.linspace(lo - 1, hi - 1, N)]  # 0-indexed

        # First pass: collect every exported layer's heatmap, then find the
        # static (persistent) outlier tokens. Only mark the without-registers row.
        heats = [to_grid_map(caps[bi], n_prefix, grid) for bi in keep]
        bgm = bg16 if grid == (16, 16) else None
        mask = (None if name.startswith("reg_")
                else persistent_outlier_mask(heats, bg_mask=bgm))
        # Second pass: render with the shared (static) outlier mask.
        for out_i, heat in enumerate(heats, start=1):
            im = render(img, heat, S=256, outlier_mask=mask)
            im.save(os.path.join(OUT, f"{name}_block{out_i:02d}.png"))
        manifest["models"][name] = {
            "label": label,
            "kept": len(keep),
            "layers": [int(b + 1) for b in keep],   # 1-indexed real layer ids
            "depth": L,
        }
        print(f"  {name}: depth={L}, {N} maps sampled from layers {lo}-{hi} "
              f"({keep[0]+1}..{keep[-1]+1})")
        torch.cuda.empty_cache()

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print("DONE ->", OUT, "| manifest written")

if __name__ == "__main__":
    main()
