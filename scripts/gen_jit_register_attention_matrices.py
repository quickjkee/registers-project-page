#!/usr/bin/env python3
"""
JiT register-attention matrices.

Starting from the JiT-only register-attention overlay script, this version keeps
EACH register token separate and renders one matrix/contact-sheet per register.

For a fixed chosen head h, block/layer b, timestep t, and register r, it uses:

    A[h, r, patch_start:patch_start + patch_tokens]

where A has shape (heads, tokens, tokens), rows are query tokens, and columns are
key tokens. The result is reshaped to the image patch grid, usually 16x16.

Outputs:
  out/input.png
  out/matrices/heat/register_01_head_0_heat_matrix.png
  out/matrices/overlay/register_01_head_0_overlay_matrix.png
  ... same for all registers ...
  out/register_attention_maps_head_0.npz
  out/manifest_register_attention_matrices.json

Matrix convention:
  rows    = timesteps
  columns = JiT blocks/layers
  cell    = attention from ONE register query token to spatial patch key tokens

Example:
  python gen_jit_register_attention_matrices.py \
    --image /path/to/cat.jpg \
    --jit-dir /home/quickjkee/projects/CUR/registers/generative_models/JiT \
    --ckpt /home/quickjkee/projects/CUR/registers/checkpoints/jit/jit_h16_regs/checkpoint-last.pth \
    --out ./jit_register_matrices \
    --class-id 281 \
    --registers 32 \
    --head 0 \
    --timesteps 0.10,0.25,0.40,0.55,0.70,0.85,0.95 \
    --patch-start 32 \
    --patch-tokens 256 \
    --grid 16x16

Important:
  If you load a no-register JiT checkpoint, the first 32 tokens are NOT registers.
  Use a checkpoint trained/configured with 32 registers and keep --registers 32.
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ORIG_SDPA = F.scaled_dot_product_attention
_CAPTURED = []


def _sdpa_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    """Monkeypatch target: saves attention probabilities for each SDPA call."""
    d = q.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(d)
    attn = (q.float() @ k.float().transpose(-2, -1)) * s
    if attn_mask is not None and not isinstance(attn_mask, bool):
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1)
    _CAPTURED.append(attn[0].detach().cpu())  # (heads, tokens, tokens), sample 0
    return _ORIG_SDPA(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


class CaptureAttention:
    def __enter__(self):
        _CAPTURED.clear()
        F.scaled_dot_product_attention = _sdpa_capture
        return self

    def __exit__(self, exc_type, exc, tb):
        F.scaled_dot_product_attention = _ORIG_SDPA


def center_crop_resize(path: str, size: int = 256) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return im.crop((left, top, left + m, top + m)).resize((size, size), Image.BICUBIC)


def image_to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x * 2.0 - 1.0


def noise_lin(image: torch.Tensor, t: float, seed: int) -> torch.Tensor:
    """JiT/SiT flow convention: t=1.0 is clean image, t=0.0 is pure noise."""
    try:
        gen = torch.Generator(device=image.device).manual_seed(seed)
        noise = torch.randn(image.shape, device=image.device, generator=gen)
    except RuntimeError:
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(image.shape, generator=gen).to(image.device)
    return t * image + (1.0 - t) * noise


def load_jit_model(
    jit_dir: str,
    ckpt_path: str,
    registers: int,
    in_context_start: int,
    strict: bool,
    arch: str,
):
    sys.path.insert(0, jit_dir)
    from model_jit import JiT_H_16, JiT_B_16  # imported after sys.path update

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    sd = OrderedDict((k[4:] if k.startswith("net.") else k, v) for k, v in sd.items())

    arch_l = arch.lower()
    if arch_l in {"b", "base", "jit_b_16"}:
        net = JiT_B_16(in_context_len=registers, in_context_start=in_context_start)
    elif arch_l in {"h", "huge", "jit_h_16"}:
        net = JiT_H_16(in_context_len=registers, in_context_start=in_context_start)
    else:
        raise ValueError("--arch must be B or H")

    missing, unexpected = net.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print("State-dict load warnings:")
        if missing:
            print("  missing keys:", missing[:20], "..." if len(missing) > 20 else "")
        if unexpected:
            print("  unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")
        print("If you expected register attention, make sure this is a register-trained checkpoint.")
    net.eval().to(DEVICE)
    return net


def parse_grid(value: str) -> Tuple[int, int]:
    s = value.lower().replace(" ", "")
    if "x" in s:
        a, b = s.split("x", 1)
        gh, gw = int(a), int(b)
    elif "," in s:
        a, b = s.split(",", 1)
        gh, gw = int(a), int(b)
    else:
        gh = gw = int(s)
    if gh <= 0 or gw <= 0:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return gh, gw


def parse_timesteps(value: str) -> List[float]:
    vals = [float(x) for x in value.replace(" ", "").split(",") if x]
    if not vals:
        raise argparse.ArgumentTypeError("--timesteps cannot be empty")
    for t in vals:
        if not (0.0 <= t <= 1.0):
            raise argparse.ArgumentTypeError("JiT timesteps should be in [0, 1]")
    return vals


def parse_head(value: str):
    if value in {"mean", "sum"}:
        return value
    try:
        v = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("head must be an integer, or optionally 'mean'/'sum'") from e
    if v < 0:
        raise argparse.ArgumentTypeError("head index must be >= 0")
    return v


def parse_layer_spec(spec: str, depth: int) -> List[int]:
    """Parse 1-based layer/block specs like 'all', '1-4,8,12'. Return 0-based indices."""
    spec = spec.strip().lower().replace(" ", "")
    if spec in {"all", "*", ""}:
        return list(range(depth))

    out = []
    for part in spec.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            out.extend(range(start - 1, end))
        else:
            out.append(int(part) - 1)

    uniq = []
    seen = set()
    for idx in out:
        if idx < 0 or idx >= depth:
            raise ValueError(f"Layer/block spec includes {idx + 1}, but valid 1-based range is 1..{depth}")
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)
    return uniq


def pick_head_register_values(reg_to_patch_hrp: torch.Tensor, head: Union[int, str]):
    """
    reg_to_patch_hrp: (heads, registers, patch_tokens)
    Return (registers, patch_tokens), selected head label.
    """
    if isinstance(head, int):
        if head >= reg_to_patch_hrp.shape[0]:
            raise ValueError(f"Requested --head {head}, but attention only has {reg_to_patch_hrp.shape[0]} heads")
        return reg_to_patch_hrp[head], head
    if head == "mean":
        return reg_to_patch_hrp.mean(dim=0), "mean"
    if head == "sum":
        return reg_to_patch_hrp.sum(dim=0), "sum"
    raise ValueError("head must be an integer, 'mean', or 'sum'")


def register_to_patch_grids(
    attn_hnn: torch.Tensor,
    registers: int,
    head: Union[int, str],
    patch_start: int,
    patch_tokens: int,
    grid: Tuple[int, int],
) -> Tuple[np.ndarray, Union[int, str]]:
    """
    Convert one layer's attention tensor into separate maps for each register.

    attn_hnn: (heads, tokens, tokens)
    Uses rows 0:registers as register query tokens and columns
    patch_start:patch_start+patch_tokens as spatial patch key tokens.

    Returns:
      maps: (registers, gh, gw)
      selected_head: int/'mean'/'sum'
    """
    if registers <= 0:
        raise ValueError("--registers must be > 0 for register attention")
    if attn_hnn.ndim != 3:
        raise ValueError(f"Expected attention shape (heads,tokens,tokens), got {tuple(attn_hnn.shape)}")
    if attn_hnn.shape[1] < registers or attn_hnn.shape[2] < patch_start + patch_tokens:
        raise ValueError(
            f"Attention shape is {tuple(attn_hnn.shape)}. Cannot use registers=0:{registers} "
            f"and patch keys={patch_start}:{patch_start + patch_tokens}. Adjust --patch-start/--patch-tokens."
        )

    gh, gw = grid
    if gh * gw != patch_tokens:
        raise ValueError(f"--grid {gh}x{gw} requires {gh * gw} patch tokens, but --patch-tokens={patch_tokens}")

    # Core extraction: register rows/query tokens -> spatial patch columns/key tokens.
    reg_to_patch = attn_hnn[:, :registers, patch_start : patch_start + patch_tokens]  # (H,R,P)
    values_rp, selected_head = pick_head_register_values(reg_to_patch, head)           # (R,P)
    maps = values_rp.numpy().reshape(registers, gh, gw)
    return maps, selected_head


def turbo(x: np.ndarray) -> np.ndarray:
    """Small turbo-like colormap; avoids needing matplotlib."""
    x = np.clip(x, 0, 1)
    return np.stack(
        [
            np.clip(1.4 * x - 0.2, 0, 1),
            np.clip(1.6 * x * (1 - 0.7 * x) + 0.1, 0, 1),
            np.clip(1.0 - 1.3 * x, 0, 1),
        ],
        axis=-1,
    )


def heat_bounds(values: np.ndarray, lo_pct: float, hi_pct: float) -> Tuple[float, float]:
    lo = float(np.percentile(values, lo_pct))
    hi = float(np.percentile(values, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def normalize_heat_with_bounds(heat: np.ndarray, lo: float, hi: float, gamma: float):
    h = np.clip((heat - lo) / (hi - lo + 1e-6), 0, 1)
    return h ** gamma


def render_overlay(
    img_rgb: np.ndarray,
    heat: np.ndarray,
    size: int,
    lo: float,
    hi: float,
    gamma: float,
) -> Image.Image:
    heat_norm = normalize_heat_with_bounds(heat, lo, hi, gamma)
    heat_img = Image.fromarray((heat_norm * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    h = np.asarray(heat_img).astype(np.float32) / 255.0

    if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
        img_rgb = np.asarray(
            Image.fromarray((img_rgb * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        ).astype(np.float32) / 255.0

    gray = img_rgb.mean(axis=-1, keepdims=True)
    bg = np.repeat(0.28 + 0.40 * gray, 3, axis=-1)
    alpha = np.clip(h * 1.1, 0, 1)[..., None]
    rgb = np.clip(bg * (1 - alpha) + turbo(h) * alpha, 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def render_heat_only(heat: np.ndarray, size: int, lo: float, hi: float, gamma: float) -> Image.Image:
    heat_norm = normalize_heat_with_bounds(heat, lo, hi, gamma)
    heat_img = Image.fromarray((heat_norm * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    h = np.asarray(heat_img).astype(np.float32) / 255.0
    rgb = turbo(h)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def font_default():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def make_matrix_image(
    cells: List[List[Image.Image]],
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    gap: int = 2,
    left_pad: int = 74,
    top_pad: int = 48,
    right_pad: int = 8,
    bottom_pad: int = 8,
) -> Image.Image:
    n_rows = len(cells)
    n_cols = len(cells[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("cannot make matrix image with no cells")

    cell_w, cell_h = cells[0][0].size
    w = left_pad + n_cols * cell_w + (n_cols - 1) * gap + right_pad
    h = top_pad + n_rows * cell_h + (n_rows - 1) * gap + bottom_pad
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = font_default()

    draw.text((6, 4), title, fill=(0, 0, 0), font=font)

    for c, label in enumerate(col_labels):
        x = left_pad + c * (cell_w + gap)
        draw.text((x + 2, top_pad - 18), label, fill=(0, 0, 0), font=font)

    for r, label in enumerate(row_labels):
        y = top_pad + r * (cell_h + gap)
        draw.text((6, y + max(0, cell_h // 2 - 6)), label, fill=(0, 0, 0), font=font)

    for r in range(n_rows):
        for c in range(n_cols):
            x = left_pad + c * (cell_w + gap)
            y = top_pad + r * (cell_h + gap)
            canvas.paste(cells[r][c], (x, y))
    return canvas


def safe_head_label(head) -> str:
    return str(head).replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--jit-dir", required=True, help="Path to generative_models/JiT")
    parser.add_argument("--ckpt", required=True, help="Path to JiT checkpoint-last.pth")
    parser.add_argument("--out", default="jit_register_matrices", help="Output directory")
    parser.add_argument("--class-id", type=int, default=281, help="ImageNet class id; 281 = tabby cat")
    parser.add_argument("--timesteps", type=parse_timesteps, default=parse_timesteps("0.10,0.25,0.40,0.55,0.70,0.85,0.95"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--registers", type=int, default=32, help="Number of register tokens at the start of the sequence")
    parser.add_argument("--in-context-start", type=int, default=4, help="JiT in_context_start constructor arg")
    parser.add_argument("--arch", default="B", choices=["B", "H", "b", "h"], help="JiT architecture constructor to use")
    parser.add_argument("--patch-start", type=int, default=None, help="First spatial patch key-token index. Default: --registers")
    parser.add_argument("--patch-tokens", type=int, default=256, help="Number of spatial patch tokens to render")
    parser.add_argument("--grid", type=parse_grid, default=parse_grid("16x16"), help="Spatial patch grid, e.g. 16x16")
    parser.add_argument("--layers", default="all", help="1-based block list/ranges, e.g. all or 1-8,12,20")
    parser.add_argument(
        "--head",
        type=parse_head,
        required=True,
        help="Chosen attention head index, e.g. --head 0. 'mean'/'sum' are allowed but less literal.",
    )
    parser.add_argument("--cell-size", type=int, default=96, help="Pixel size of each mini attention map in the matrix")
    parser.add_argument("--gap", type=int, default=2, help="Gap between matrix cells")
    parser.add_argument("--lo-pct", type=float, default=40.0, help="Low percentile for heat normalization")
    parser.add_argument("--hi-pct", type=float, default=99.5, help="High percentile for heat normalization")
    parser.add_argument("--gamma", type=float, default=1.6, help="Gamma applied after percentile normalization")
    parser.add_argument(
        "--norm",
        choices=["per-register", "global", "per-cell"],
        default="per-register",
        help="Normalization for matrix cells. per-register keeps cells comparable within one register matrix.",
    )
    parser.add_argument("--save-cells", action="store_true", help="Also save individual cell PNG files")
    parser.add_argument(
        "--non-strict-load",
        action="store_true",
        help="Load checkpoint with strict=False. Useful only for debugging mismatched checkpoints.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    heat_dir = os.path.join(args.out, "matrices", "heat")
    overlay_dir = os.path.join(args.out, "matrices", "overlay")
    os.makedirs(heat_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    patch_start = args.patch_start if args.patch_start is not None else args.registers
    gh, gw = args.grid
    if gh * gw != args.patch_tokens:
        raise ValueError(f"--grid {gh}x{gw} has {gh * gw} cells, but --patch-tokens={args.patch_tokens}")

    input_img = center_crop_resize(args.image, args.size)
    input_img.save(os.path.join(args.out, "input.png"))
    img_rgb = np.asarray(input_img).astype(np.float32) / 255.0

    net = load_jit_model(
        args.jit_dir,
        args.ckpt,
        registers=args.registers,
        in_context_start=args.in_context_start,
        strict=not args.non_strict_load,
        arch=args.arch,
    )
    x = image_to_tensor(input_img).to(DEVICE)
    y = torch.tensor([args.class_id], device=DEVICE)

    selected_layers = None
    selected_head = None
    maps = None  # (registers, timesteps, layers, gh, gw)
    captured_depth = None
    total_heads = None
    total_tokens = None

    for ti, t in enumerate(args.timesteps):
        tt = torch.full((1,), float(t), device=DEVICE)
        z = noise_lin(x, float(t), args.seed)
        with torch.no_grad(), CaptureAttention():
            net(z, tt, y)

        if not _CAPTURED:
            raise RuntimeError(
                "No attention maps were captured. The model may not be using "
                "torch.nn.functional.scaled_dot_product_attention."
            )

        if selected_layers is None:
            captured_depth = len(_CAPTURED)
            selected_layers = parse_layer_spec(args.layers, captured_depth)
            if not selected_layers:
                raise ValueError("No layers selected")
            total_heads = int(_CAPTURED[0].shape[0])
            total_tokens = int(_CAPTURED[0].shape[-1])
            maps = np.zeros((args.registers, len(args.timesteps), len(selected_layers), gh, gw), dtype=np.float32)
            print(
                f"Captured depth={captured_depth}, heads={total_heads}, tokens={total_tokens}; "
                f"using blocks={[i + 1 for i in selected_layers]}, head={args.head}, "
                f"register queries=0:{args.registers}, patch keys={patch_start}:{patch_start + args.patch_tokens}"
            )
        elif len(_CAPTURED) != captured_depth:
            raise RuntimeError(f"Captured {len(_CAPTURED)} layers at t={t}, expected {captured_depth}")

        for li, layer_idx in enumerate(selected_layers):
            layer_maps, h = register_to_patch_grids(
                _CAPTURED[layer_idx],
                registers=args.registers,
                head=args.head,
                patch_start=patch_start,
                patch_tokens=args.patch_tokens,
                grid=args.grid,
            )
            if selected_head is None:
                selected_head = h
            maps[:, ti, li] = layer_maps

        print(f"  timestep {t:.3f}: done")
        torch.cuda.empty_cache()

    assert maps is not None and selected_layers is not None
    head_label = safe_head_label(selected_head)

    npz_name = f"register_attention_maps_head_{head_label}.npz"
    np.savez_compressed(
        os.path.join(args.out, npz_name),
        maps=maps,
        timesteps=np.asarray(args.timesteps, dtype=np.float32),
        layers_1_based=np.asarray([i + 1 for i in selected_layers], dtype=np.int32),
        registers_1_based=np.arange(1, args.registers + 1, dtype=np.int32),
    )

    if args.norm == "global":
        global_lo, global_hi = heat_bounds(maps, args.lo_pct, args.hi_pct)
    else:
        global_lo, global_hi = None, None

    row_labels = [f"t={t:.2f}" for t in args.timesteps]
    col_labels = [f"B{i + 1:02d}" for i in selected_layers]

    outputs = []
    for r in range(args.registers):
        reg_maps = maps[r]  # (T,L,gh,gw)
        if args.norm == "per-register":
            reg_lo, reg_hi = heat_bounds(reg_maps, args.lo_pct, args.hi_pct)
        elif args.norm == "global":
            reg_lo, reg_hi = global_lo, global_hi
        else:
            reg_lo, reg_hi = None, None

        heat_cells = []
        overlay_cells = []
        for ti in range(len(args.timesteps)):
            heat_row = []
            overlay_row = []
            for li in range(len(selected_layers)):
                heat = reg_maps[ti, li]
                if args.norm == "per-cell":
                    lo, hi = heat_bounds(heat, args.lo_pct, args.hi_pct)
                else:
                    lo, hi = reg_lo, reg_hi
                heat_row.append(render_heat_only(heat, args.cell_size, lo, hi, args.gamma))
                overlay_row.append(render_overlay(img_rgb, heat, args.cell_size, lo, hi, args.gamma))
            heat_cells.append(heat_row)
            overlay_cells.append(overlay_row)

        title = f"Register {r + 1} / token {r}; head={selected_head}; rows=timesteps; cols=blocks"
        heat_matrix = make_matrix_image(heat_cells, row_labels, col_labels, title, gap=args.gap)
        overlay_matrix = make_matrix_image(overlay_cells, row_labels, col_labels, title, gap=args.gap)

        heat_name = f"register_{r + 1:02d}_head_{head_label}_heat_matrix.png"
        overlay_name = f"register_{r + 1:02d}_head_{head_label}_overlay_matrix.png"
        heat_matrix.save(os.path.join(heat_dir, heat_name))
        overlay_matrix.save(os.path.join(overlay_dir, overlay_name))

        item = {
            "register_0_based": r,
            "register_1_based": r + 1,
            "heat_matrix": os.path.join("matrices", "heat", heat_name),
            "overlay_matrix": os.path.join("matrices", "overlay", overlay_name),
        }
        outputs.append(item)

        if args.save_cells:
            cell_dir = os.path.join(args.out, "cells", f"register_{r + 1:02d}")
            os.makedirs(cell_dir, exist_ok=True)
            for ti, t in enumerate(args.timesteps):
                for li, layer_idx in enumerate(selected_layers):
                    heat = reg_maps[ti, li]
                    if args.norm == "per-cell":
                        lo, hi = heat_bounds(heat, args.lo_pct, args.hi_pct)
                    elif args.norm == "per-register":
                        lo, hi = reg_lo, reg_hi
                    else:
                        lo, hi = global_lo, global_hi
                    render_heat_only(heat, args.size, lo, hi, args.gamma).save(
                        os.path.join(cell_dir, f"r{r + 1:02d}_t{ti + 1:02d}_B{layer_idx + 1:02d}_heat.png")
                    )
                    render_overlay(img_rgb, heat, args.size, lo, hi, args.gamma).save(
                        os.path.join(cell_dir, f"r{r + 1:02d}_t{ti + 1:02d}_B{layer_idx + 1:02d}_overlay.png")
                    )

    manifest = {
        "device": DEVICE,
        "image": os.path.abspath(args.image),
        "checkpoint": os.path.abspath(args.ckpt),
        "architecture": args.arch.upper(),
        "class_id": args.class_id,
        "seed": args.seed,
        "num_captured_layers": captured_depth,
        "num_heads": total_heads,
        "num_tokens": total_tokens,
        "attention_direction": "single register query token -> spatial patch key tokens",
        "register_query_token_range": [0, args.registers],
        "patch_key_token_range": [patch_start, patch_start + args.patch_tokens],
        "grid": [gh, gw],
        "timesteps": [float(t) for t in args.timesteps],
        "layers_0_based": [int(i) for i in selected_layers],
        "layers_1_based": [int(i + 1) for i in selected_layers],
        "selected_head": selected_head,
        "normalization": args.norm,
        "maps_npz": npz_name,
        "maps_shape": list(maps.shape),
        "matrix_rows": "timesteps",
        "matrix_columns": "blocks/layers",
        "outputs": outputs,
    }

    manifest_name = "manifest_register_attention_matrices.json"
    with open(os.path.join(args.out, manifest_name), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {os.path.abspath(args.out)}")
    print(f"Raw maps: {npz_name} with shape {maps.shape} = registers x timesteps x layers x {gh} x {gw}")
    print(f"Manifest: {manifest_name}")


if __name__ == "__main__":
    main()
