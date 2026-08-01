#!/usr/bin/env python3
"""Pack invariant checks for EVS demo (no GPU required for pack tests).

Ranking tests prove keep/prune follows EVS global top-k on pairwise
inter-frame dissimilarity (high cosine similarity → pruned).

Precomputed examples from ``examples.json`` are tested individually
(parametrized) so each baked video pack reports its own pass/fail.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "public" / "pack"
MANIFEST = PACK_ROOT / "examples.json"
LEGACY_PACK = PACK_ROOT / "pack.json"
# Prefer vendored copy (standalone repo); fall back to nested Hiprune vLLM fork.
EVS_PATH = ROOT / "vendor" / "evs.py"
if not EVS_PATH.is_file():
    EVS_PATH = ROOT.parents[0] / "vllm" / "vllm" / "multimodal" / "evs.py"


def load_evs():
    spec = importlib.util.spec_from_file_location("hiprune_evs", EVS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def example_cases() -> list[tuple[str, Path]]:
    """(example_id, pack.json path) for each precomputed demo video."""
    cases: list[tuple[str, Path]] = []
    if MANIFEST.is_file():
        data = json.loads(MANIFEST.read_text())
        for ex in data.get("examples", []):
            cases.append((ex["id"], PACK_ROOT / ex["pack"]))
    elif LEGACY_PACK.is_file():
        cases.append(("legacy", LEGACY_PACK))
    return cases


def pack_paths() -> list[Path]:
    return [p for _, p in example_cases()]


def ffprobe_video(path: Path) -> dict:
    """Return width, height, duration from ffprobe (requires ffmpeg)."""
    assert shutil.which("ffprobe"), "ffprobe not found (install ffmpeg)"
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    data = json.loads(out)
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(data["format"]["duration"]),
    }


def check_precomputed_video(pack: dict, pack_path: Path) -> None:
    """Assert baked MP4 exists and matches pack video_* metadata."""
    assert pack.get("video"), f"{pack_path}: missing video field"
    video_path = pack_path.parent / pack["video"]
    assert video_path.is_file(), f"missing precomputed video {video_path}"
    assert video_path.stat().st_size > 10_000, f"{video_path} looks empty"

    meta = ffprobe_video(video_path)
    assert meta["width"] == int(pack["video_width"]), (
        f"{pack_path}: width {meta['width']} != pack {pack['video_width']}"
    )
    assert meta["height"] == int(pack["video_height"]), (
        f"{pack_path}: height {meta['height']} != pack {pack['video_height']}"
    )
    pack_dur = float(pack["video_duration"])
    assert abs(meta["duration"] - pack_dur) < 0.15, (
        f"{pack_path}: duration {meta['duration']} != pack {pack_dur}"
    )


def check_pack(pack_path: Path) -> None:
    assert pack_path.is_file(), f"missing {pack_path}"
    pack = json.loads(pack_path.read_text())
    T = pack["num_frames"]
    Wp, Hp = pack["grid"]
    tokens = pack["tokens_per_frame"]
    assert tokens == Wp * Hp
    assert len(pack["frames"]) == T
    assert len(pack["retention_mask"]) == T
    assert len(pack["dissimilarity"]) == T
    assert pack["grid"][0] == Wp and pack["grid"][1] == Hp

    pack_dir = pack_path.parent
    for rel in pack["frames"]:
        assert (pack_dir / rel).is_file(), f"missing frame {rel} under {pack_dir}"

    if pack.get("video"):
        assert (pack_dir / pack["video"]).is_file(), f"missing {pack['video']}"
        assert float(pack.get("video_duration") or 0) > 0
        assert int(pack.get("video_width") or 0) > 0
        assert int(pack.get("video_height") or 0) > 0

    for t, row in enumerate(pack["retention_mask"]):
        assert len(row) == tokens, f"mask[{t}] len"
        assert len(pack["dissimilarity"][t]) == tokens

    assert all(pack["retention_mask"][0]), "step 0 must be fully kept"

    kept_total = sum(sum(1 for v in row if v) for row in pack["retention_mask"])
    assert kept_total == pack["kept_total"]

    evs = load_evs()
    expected = evs.compute_retained_tokens_count(tokens, T, pack["q"])
    if not pack.get("synthetic", False):
        assert kept_total == expected, f"kept {kept_total} != EVS budget {expected}"
    else:
        if T > 1:
            assert not all(pack["retention_mask"][1]), "synthetic step1 should prune"


def check_topk_from_dissimilarity(pack: dict, pack_path: Path) -> None:
    """Mask must equal global stable top-k on flattened dissimilarity (EVS rule)."""
    T = pack["num_frames"]
    tokens = pack["tokens_per_frame"]
    kept_total = pack["kept_total"]
    dis = [v for row in pack["dissimilarity"] for v in row]
    mask = [bool(v) for row in pack["retention_mask"] for v in row]
    assert len(dis) == T * tokens
    assert len(mask) == T * tokens

    # Stable descending argsort (matches torch.argsort(..., descending=True, stable=True))
    order = sorted(range(len(dis)), key=lambda i: (-dis[i], i))
    top = set(order[:kept_total])
    got = {i for i, v in enumerate(mask) if v}
    assert top == got, f"{pack_path}: retention_mask != global top-k on dissimilarity"


def check_step_ordering(pack: dict, pack_path: Path) -> None:
    """For t>0, kept dissim >= pruned dissim within each step (global top-k consequence)."""
    T = pack["num_frames"]
    tokens = pack["tokens_per_frame"]
    for t in range(1, T):
        dis = pack["dissimilarity"][t]
        mask = pack["retention_mask"][t]
        kept = [dis[i] for i in range(tokens) if mask[i]]
        pruned = [dis[i] for i in range(tokens) if not mask[i]]
        if not kept or not pruned:
            continue
        assert min(kept) + 1e-9 >= max(pruned), (
            f"{pack_path} step {t}: kept_min={min(kept)} < pruned_max={max(pruned)}"
        )
        # Equivalent cosine: high similarity → pruned
        sim_kept = [1.0 - d for d in kept]
        sim_pruned = [1.0 - d for d in pruned]
        assert max(sim_kept) <= min(sim_pruned) + 1e-9, (
            f"{pack_path} step {t}: kept cos sim not <= pruned cos sim"
        )


_EXAMPLE_CASES = example_cases()
_EXAMPLE_IDS = [c[0] for c in _EXAMPLE_CASES]


@pytest.mark.parametrize("example_id,pack_path", _EXAMPLE_CASES, ids=_EXAMPLE_IDS)
def test_precomputed_pack_invariants(example_id: str, pack_path: Path):
    """Each baked example: shapes, budget, step-0 keep, assets on disk."""
    assert example_id
    check_pack(pack_path)


@pytest.mark.parametrize("example_id,pack_path", _EXAMPLE_CASES, ids=_EXAMPLE_IDS)
def test_precomputed_retention_global_topk(example_id: str, pack_path: Path):
    """Each baked pack's mask == global top-k on stored dissimilarity."""
    assert example_id
    pack = json.loads(pack_path.read_text())
    check_topk_from_dissimilarity(pack, pack_path)
    check_step_ordering(pack, pack_path)


@pytest.mark.parametrize("example_id,pack_path", _EXAMPLE_CASES, ids=_EXAMPLE_IDS)
def test_precomputed_video_matches_pack(example_id: str, pack_path: Path):
    """Each baked video.mp4 exists and matches pack width/height/duration."""
    assert example_id
    pack = json.loads(pack_path.read_text())
    check_precomputed_video(pack, pack_path)


def test_evs_module_loads():
    evs = load_evs()
    assert hasattr(evs, "compute_retention_mask")
    assert hasattr(evs, "compute_retained_tokens_count")
    assert evs.compute_retained_tokens_count(100, 4, 0.75) == max(100, int(400 * 0.25))


def test_evs_topk_synthetic():
    """Tiny fake embeds: highest-dissim cells kept; most similar pruned."""
    import torch

    evs = load_evs()
    # T=2, pre-merge H=W=2, merge=1 → 2x2 soft tokens per step
    # Frame0 embeds; frame1 nearly identical on (0,0)/(0,1), different on (1,0)/(1,1)
    T, H, W, C = 2, 2, 2, 8
    merge = 1
    embeds = torch.zeros(T * H * W, C)
    # step 0
    embeds[0] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    embeds[1] = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
    embeds[2] = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])
    embeds[3] = torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0])
    # step 1: (0,0) and (0,1) almost same as step0 → high similarity → prune
    embeds[4] = embeds[0] * 0.99
    embeds[5] = embeds[1] * 0.99
    # (1,0) and (1,1) orthogonal-ish → high dissim → keep
    embeds[6] = torch.tensor([0, 0, 0, 0, 1.0, 0, 0, 0])
    embeds[7] = torch.tensor([0, 0, 0, 0, 0, 1.0, 0, 0])

    q = 0.5  # keep max(4, int(8*0.5))=4 → all of step0 (forced) uses budget; wait
    # retained = max(tokens_per_frame, int(total*(1-q))) = max(4, int(8*0.5)) = max(4,4)=4
    # Step0 sentinel forces all 4 step0 tokens into top-k, so step1 may all be pruned.
    # Use lower q so some step1 tokens survive: q=0.25 → keep max(4, int(8*0.75))=6
    q = 0.25
    mask = evs.compute_retention_mask(
        embeds, (T, H, W), spatial_merge_size=merge, q=q
    )
    assert mask.shape[0] == T * H * W
    mask_thw = mask.view(T, H, W)
    assert bool(mask_thw[0].all()), "step 0 fully kept"

    # Dissimilarity for step1
    e0 = embeds[:4].view(H, W, C)
    e1 = embeds[4:].view(H, W, C)
    sim = torch.nn.functional.cosine_similarity(e1, e0, dim=-1)
    dis = 1 - sim
    # The two highest-dissim step1 cells should be preferred over the two low-dissim ones
    flat_dis = dis.view(-1)
    # Among step1 indices 4..7 in global mask, kept ones must have higher dis than pruned
    step1_kept = []
    step1_pruned = []
    for i in range(4):
        if mask[4 + i]:
            step1_kept.append(float(flat_dis[i]))
        else:
            step1_pruned.append(float(flat_dis[i]))
    assert step1_kept, "expected some step1 keeps with q=0.25"
    assert step1_pruned, "expected some step1 prunes"
    assert min(step1_kept) >= max(step1_pruned) - 1e-5

    # Explicitly: high-sim cells (0,0) and (0,1) should be pruned before low-sim
    assert float(sim[0, 0]) > 0.9 and float(sim[0, 1]) > 0.9
    assert not bool(mask_thw[1, 0, 0]) or not bool(mask_thw[1, 0, 1])
    # At least one of the high-change bottom cells kept
    assert bool(mask_thw[1, 1, 0]) or bool(mask_thw[1, 1, 1])


def test_manifest_keys_unique():
    if not MANIFEST.is_file():
        return
    data = json.loads(MANIFEST.read_text())
    keys = [ex["key"] for ex in data["examples"]]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))
    assert _EXAMPLE_CASES, "expected precomputed examples in examples.json"


if __name__ == "__main__":
    test_evs_module_loads()
    test_manifest_keys_unique()
    for _id, p in example_cases():
        check_pack(p)
        pack = json.loads(p.read_text())
        check_topk_from_dissimilarity(pack, p)
        check_step_ordering(pack, p)
        check_precomputed_video(pack, p)
    test_evs_topk_synthetic()
    print("ok")