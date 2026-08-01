#!/usr/bin/env python3
"""Pack invariant checks for EVS demo (no GPU)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "public" / "pack"
MANIFEST = PACK_ROOT / "examples.json"
LEGACY_PACK = PACK_ROOT / "pack.json"
EVS_PATH = ROOT / "vendor" / "evs.py"


def load_evs():
    spec = importlib.util.spec_from_file_location("hiprune_evs", EVS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pack_paths() -> list[Path]:
    paths: list[Path] = []
    if MANIFEST.is_file():
        data = json.loads(MANIFEST.read_text())
        for ex in data.get("examples", []):
            rel = ex["pack"]
            paths.append(PACK_ROOT / rel)
    elif LEGACY_PACK.is_file():
        paths.append(LEGACY_PACK)
    return paths


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


def test_pack_invariants():
    paths = pack_paths()
    assert paths, "no packs found (examples.json or pack.json)"
    for p in paths:
        check_pack(p)


def test_evs_module_loads():
    evs = load_evs()
    assert hasattr(evs, "compute_retention_mask")
    assert hasattr(evs, "compute_retained_tokens_count")
    assert evs.compute_retained_tokens_count(100, 4, 0.75) == max(100, int(400 * 0.25))


def test_manifest_keys_unique():
    if not MANIFEST.is_file():
        return
    data = json.loads(MANIFEST.read_text())
    keys = [ex["key"] for ex in data["examples"]]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


if __name__ == "__main__":
    test_evs_module_loads()
    test_manifest_keys_unique()
    test_pack_invariants()
    print("ok")
