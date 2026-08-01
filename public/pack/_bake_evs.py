#!/usr/bin/env python3
"""Bake an EVS visualization pack from a short video (GPU / MPS).

Uses Qwen2.5-VL's vision tower for embeds, then the exact
``compute_retention_mask`` from the nested vLLM fork via importlib
(file load — does not import the full vLLM package).

``video.mp4`` is the fluid source footage (no prune bake-in). Debug
``frames/*.jpg`` are sampled from that footage (one per EVS temporal step),
not from model tensors.

Example:
  python3 _bake_evs.py \\
    --video _src/clip.mp4 \\
    --q 0.75 \\
    --model Qwen/Qwen2.5-VL-3B-Instruct \\
    --id videomme_evs \\
    --videomme-id 4ZK-m01XSQ8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
EVS_PATH = Path(__file__).resolve().parents[2] / "vendor" / "evs.py"


def load_evs():
    if not EVS_PATH.is_file():
        raise FileNotFoundError(f"EVS module not found: {EVS_PATH}")
    spec = importlib.util.spec_from_file_location("hiprune_evs", EVS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {EVS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_dissimilarity(
    video_embeds: torch.Tensor,
    video_size_thw: tuple[int, int, int],
    spatial_merge_size: int,
) -> torch.Tensor:
    """Per-step dissimilarity map matching EVS (step 0 = 255 sentinel)."""
    T, H, W = map(int, video_size_thw)
    embeds = video_embeds.reshape(
        T,
        H // spatial_merge_size,
        W // spatial_merge_size,
        video_embeds.size(-1),
    )
    similarity = torch.nn.functional.cosine_similarity(
        embeds[1:, ...], embeds[:-1, ...], dim=-1
    )
    dissimilarity = 1 - similarity
    dissimilarity = torch.cat(
        [255 * torch.ones_like(embeds[:1, :, :, 0]), dissimilarity], dim=0
    )
    return dissimilarity  # (T, H', W')


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ffmpeg_prep(
    video_path: Path,
    out_mp4: Path,
    fps: float = 4.0,
    width: int = 640,
) -> Path:
    """Re-encode to a decoder-friendly clip for the vision tower."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_mp4),
    ]
    print("ffmpeg prep …", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp4


def save_temporal_jpegs_from_source_video(
    video_path: Path,
    num_temporal_steps: int,
    duration_s: float,
    out_dir: Path,
) -> list[str]:
    """Save one real RGB JPEG per EVS temporal step from the source MP4.

    These are debug stills only — the UI plays ``video.mp4``. Never export
    model tensors as images (that produced fake cyan-grid “pruning” frames).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.jpg"):
        p.unlink()

    rels: list[str] = []
    dur = max(duration_s, 0.1)
    for t in range(num_temporal_steps):
        # Center of each temporal bin along the clip
        ts = ((t + 0.5) / num_temporal_steps) * dur
        rel = f"frames/{t:03d}.jpg"
        out = out_dir.parent / rel
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        rels.append(rel)
    return rels


def run_vision(
    model_id: str,
    video_path: Path,
    device: torch.device,
    sample_fps: float = 2.0,
):
    """Return embeds, grid_thw, merge, temporal_patch, video_tensor (N,C,H,W)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    print(f"loading {model_id} dtype={dtype} …", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "fps": sample_fps,
                },
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )
    if not video_inputs:
        raise RuntimeError("process_vision_info returned no video")
    video_tensor = video_inputs[0]
    if not isinstance(video_tensor, torch.Tensor):
        raise RuntimeError(f"unexpected video_inputs type: {type(video_tensor)}")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    # Move tensors; keep non-tensors as-is
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")
    if pixel_values_videos is None or video_grid_thw is None:
        raise RuntimeError(
            "processor did not return pixel_values_videos / video_grid_thw"
        )

    vision = model.visual if hasattr(model, "visual") else model.model.visual
    merge = int(getattr(vision, "spatial_merge_size", 2))
    temporal_patch = int(
        getattr(
            getattr(model.config, "vision_config", model.config),
            "temporal_patch_size",
            2,
        )
    )

    print("running vision tower …", flush=True)
    with torch.inference_mode():
        embeds = vision(pixel_values_videos, grid_thw=video_grid_thw)

    if isinstance(embeds, (tuple, list)):
        embeds = embeds[0]
    grid = video_grid_thw[0].detach().cpu().tolist()
    T, H, W = int(grid[0]), int(grid[1]), int(grid[2])
    expected = T * (H // merge) * (W // merge)
    if embeds.shape[0] != expected:
        raise RuntimeError(
            f"embed length {embeds.shape[0]} != expected merged {expected} "
            f"for grid_thw={(T, H, W)} merge={merge}"
        )

    # Free model VRAM/MPS before writing pack
    embeds_cpu = embeds.float().cpu()
    del embeds, model, vision, inputs, pixel_values_videos
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return embeds_cpu, (T, H, W), merge, temporal_patch, video_tensor.cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video",
        type=Path,
        default=ROOT / "_src" / "clip.mp4",
    )
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--fps-play", type=float, default=4.0, help="Legacy pack fps field")
    ap.add_argument(
        "--sample-fps",
        type=float,
        default=4.0,
        help="Vision sample fps (higher = finer EVS temporal steps)",
    )
    ap.add_argument(
        "--prep-width",
        type=int,
        default=640,
        help="Width for ffmpeg vision prep (keeps aspect; 960+ may OOM on MPS)",
    )
    ap.add_argument("--id", type=str, default="videomme_evs")
    ap.add_argument("--label", type=str, default="Video-MME talking head — EVS")
    ap.add_argument(
        "--videomme-id",
        type=str,
        default="540LkURTR7g",
        help="Optional Video-MME clip id stored in pack.json",
    )
    ap.add_argument(
        "--skip-ffmpeg",
        action="store_true",
        help="Use --video as-is (already decoder-friendly)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for pack.json / video.mp4 / frames (default: this script's folder)",
    )
    args = ap.parse_args()

    out_root = (args.out_dir or ROOT).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.video.is_file():
        print(f"missing video: {args.video}", file=sys.stderr)
        return 1
    if not (0.0 <= args.q < 1.0):
        print("--q must be in [0, 1)", file=sys.stderr)
        return 1

    evs = load_evs()
    device = pick_device()
    print(
        f"device={device} model={args.model} q={args.q} "
        f"sample_fps={args.sample_fps} prep_width={args.prep_width} out={out_root}",
        flush=True,
    )

    tmpdir = tempfile.mkdtemp(prefix="evs_bake_")
    try:
        if args.skip_ffmpeg:
            bake_video = args.video
        else:
            bake_video = Path(tmpdir) / "clip_vision.mp4"
            ffmpeg_prep(
                args.video,
                bake_video,
                fps=args.sample_fps,
                width=args.prep_width,
            )

        embeds, grid_thw, merge, temporal_patch, video_tensor = run_vision(
            args.model, bake_video, device, sample_fps=args.sample_fps
        )
        T, H, W = grid_thw
        Hp, Wp = H // merge, W // merge
        tokens_per_frame = Hp * Wp
        print(
            f"grid_thw=({T},{H},{W}) soft=({Wp}x{Hp}) merge={merge} "
            f"temporal_patch={temporal_patch} raw_frames={tuple(video_tensor.shape)}",
            flush=True,
        )

        mask = evs.compute_retention_mask(
            embeds, grid_thw, spatial_merge_size=merge, q=args.q
        )
        dissim = compute_dissimilarity(embeds, grid_thw, merge)

        mask_thw = mask.view(T, Hp, Wp)
        retained = int(mask.sum().item())
        expected = evs.compute_retained_tokens_count(tokens_per_frame, T, args.q)
        if retained != expected:
            print(
                f"warn: kept {retained} != compute_retained_tokens_count {expected}",
                file=sys.stderr,
            )
        if not bool(mask_thw[0].all()):
            print("warn: step 0 is not fully kept", file=sys.stderr)

        # Encode UI video FIRST (raw footage only — never bake overlays into MP4)
        video_rel = "video.mp4"
        video_out = out_root / video_rel
        print(f"encoding fluid UI video → {video_out} …", flush=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(args.video),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                str(video_out),
            ],
            check=True,
            capture_output=True,
        )
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        meta = json.loads(probe.stdout)
        stream = (meta.get("streams") or [{}])[0]
        fmt = meta.get("format") or {}
        video_w = int(stream.get("width") or 0)
        video_h = int(stream.get("height") or 0)
        video_dur = float(stream.get("duration") or fmt.get("duration") or 0)

        frames_dir = out_root / "frames"
        print("writing debug JPEGs from source video (not model tensors) …", flush=True)
        frame_rels = save_temporal_jpegs_from_source_video(
            args.video,
            num_temporal_steps=T,
            duration_s=video_dur if video_dur > 0 else float(T),
            out_dir=frames_dir,
        )

        retention_mask = [[bool(v) for v in row] for row in mask_thw.view(T, -1).tolist()]
        dissimilarity = dissim.view(T, -1).tolist()

        pack = {
            "id": args.id,
            "label": args.label,
            "model": "qwen2_5_vl",
            "q": args.q,
            "fps": args.fps_play,
            "playback_rate": 1.0,
            "grid": [Wp, Hp],
            "num_frames": T,
            "tokens_per_frame": tokens_per_frame,
            "kept_total": retained,
            "spatial_merge_size": merge,
            "temporal_patch_size": temporal_patch,
            "video": video_rel,
            "video_duration": video_dur,
            "video_width": video_w,
            "video_height": video_h,
            "frames": frame_rels,
            "retention_mask": retention_mask,
            "dissimilarity": dissimilarity,
            "source_video": str(args.video),
            "videomme_id": args.videomme_id or None,
            "hf_model": args.model,
            "synthetic": False,
        }
        out = out_root / "pack.json"
        out.write_text(json.dumps(pack) + "\n")
        print(
            f"wrote {out} T={T} grid={Wp}x{Hp} kept={retained}/{T * tokens_per_frame} "
            f"video={video_w}x{video_h}@{video_dur:.2f}s",
            flush=True,
        )
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
