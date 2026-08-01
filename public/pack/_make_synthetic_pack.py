#!/usr/bin/env python3
"""Generate a tiny synthetic EVS pack for UI smoke testing (no GPU).

Creates 4 temporal steps on an 8x6 soft-token grid with a moving "object"
blob so prune overlays read clearly. Step 0 is fully kept.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
OUT = ROOT / "pack.json"

T = 4
W, H = 8, 6  # [W', H'] width first
TOKENS = W * H
Q = 0.75
FPS = 2.0
CELL = 48


def idx(r: int, c: int) -> int:
    return r * W + c


def make_mask(t: int) -> list[bool]:
    """Step 0 all True; later steps keep a moving 2x2 blob + a few cells."""
    mask = [False] * TOKENS
    if t == 0:
        return [True] * TOKENS
    # Moving blob (simulates motion hotspot)
    cr = 1 + (t % 3)
    cc = 1 + t
    for dr in (0, 1):
        for dc in (0, 1):
            r, c = cr + dr, cc + dc
            if 0 <= r < H and 0 <= c < W:
                mask[idx(r, c)] = True
    # Extra keep along a vertical edge to look less empty at q=0.75
    for r in range(H):
        if r % 2 == 0:
            mask[idx(r, min(W - 1, cc + 2))] = True
    return mask


def make_dissim(t: int, mask: list[bool]) -> list[float]:
    if t == 0:
        return [255.0] * TOKENS
    out = [0.05] * TOKENS
    for i, kept in enumerate(mask):
        if kept:
            out[i] = 0.55 + 0.35 * ((i % 5) / 5.0)
    return out


def draw_frame(t: int, path: Path) -> None:
    img = Image.new("RGB", (W * CELL, H * CELL), (32, 48, 64))
    draw = ImageDraw.Draw(img)
    # Static background stripes
    for r in range(H):
        for c in range(W):
            x0, y0 = c * CELL, r * CELL
            shade = 40 + (c + r) % 3 * 12
            draw.rectangle(
                [x0, y0, x0 + CELL - 1, y0 + CELL - 1],
                fill=(shade, shade + 8, shade + 16),
            )
    # Moving object
    cr = 1 + (t % 3)
    cc = 1 + t
    for dr in (0, 1):
        for dc in (0, 1):
            r, c = cr + dr, min(W - 1, cc + dc)
            x0, y0 = c * CELL, r * CELL
            draw.ellipse(
                [x0 + 4, y0 + 4, x0 + CELL - 5, y0 + CELL - 5],
                fill=(220, 90, 40),
            )
    # Soft vignette label
    draw.text((8, 8), f"t={t}", fill=(240, 240, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)


def main() -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    frames: list[str] = []
    retention: list[list[bool]] = []
    dissim: list[list[float]] = []
    kept_total = 0
    for t in range(T):
        rel = f"frames/{t:03d}.jpg"
        draw_frame(t, ROOT / rel)
        frames.append(rel)
        m = make_mask(t)
        retention.append(m)
        dissim.append(make_dissim(t, m))
        kept_total += sum(1 for v in m if v)

    pack = {
        "id": "synthetic_smoke",
        "label": "Synthetic smoke (replace with _bake_evs.py)",
        "model": "qwen2_5_vl",
        "q": Q,
        "fps": FPS,
        "grid": [W, H],
        "num_frames": T,
        "tokens_per_frame": TOKENS,
        "kept_total": kept_total,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "frames": frames,
        "retention_mask": retention,
        "dissimilarity": dissim,
        "synthetic": True,
    }
    OUT.write_text(json.dumps(pack, indent=2) + "\n")
    print(f"wrote {OUT} ({T} steps, {W}x{H}, kept_total={kept_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
