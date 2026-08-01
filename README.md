# EVS Temporal Token Pruning Demo

Standalone recording demo for **NVIDIA Efficient Video Sampling (EVS)** in vLLM.

Plays the **real source video fluidly**, with a transparent soft-token patch
overlay synced to EVS temporal steps (so pruning reads as continuous, not
blocky stills):

1. Original video  
2. Inter-frame dissimilarity heatmap overlay  
3. Keep / prune overlay  

This visualizes **vLLM EVS**, not HiPrune.

## Run

```bash
npm install
npm run dev
```

Open http://127.0.0.1:5183/

Multi-example packs under `public/pack/examples/` (see `examples.json`):

| Key | Example |
|-----|---------|
| `1` | Video-MME hurdles (`4ZK-m01XSQ8`) |
| `2` | Video-MME talking head (`540LkURTR7g`, fixed background) |
| `3` | Video-MME laptop wipe (`7iXM5aq53Ts`, fixed overhead desk) |
| `4` | Video-MME gym band (`21q-lDikdBg`, seated resistance-band exercise) |
| `5` | Video-MME makeup (`6NVr0cNiHPM`, fixed pink backdrop, hand/face motion) |
| `6` | Video-MME egg flip (`1sTQOxXFO44`, colorful kitchen motion) |
| `7` | Video-MME outdoor workout (`6EIrArTyLVU`, grass/trees, outdoor bars) |

**Controls:** press a number to cue that example (start frame, no overlay); press the same number again to play with the EVS prune overlay. `Space` / `Enter` pauses and resumes (keeps overlay + time). While paused, hover a patch to see pairwise cosine similarity vs the previous temporal step. `Esc` freezes back to cue.

## Temporal steps vs source video

The **UI plays the full source video fluidly** (`public/pack/video.mp4`).  
EVS itself operates on a coarser temporal soft-token grid (`temporal_patch_size = 2`):
pack `num_frames` = that `T`. The prune/heat overlay updates as video time maps onto
those EVS steps (`time → floor(t/duration * T)`), so the video looks continuous while
the mask follows the real pruning schedule.

Step **0 is always fully kept** (EVS forces this).

## Re-bake (GPU / MPS)

Each example is a folder under `public/pack/examples/`. Example:

```bash
cd public/pack
# 1 — hurdles
python3 _bake_evs.py \
  --video _src/clip_hurdles.mp4 \
  --out-dir examples/1_hurdles \
  --q 0.75 --sample-fps 2 --prep-width 640 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --id hurdles_evs --videomme-id 4ZK-m01XSQ8 \
  --label "Video-MME hurdles — EVS"

# 2 — talking head (fixed background)
python3 _bake_evs.py \
  --video _src/clip.mp4 \
  --out-dir examples/2_talking_head \
  --q 0.75 --sample-fps 4 --prep-width 640 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --id talking_head_evs --videomme-id 540LkURTR7g \
  --label "Video-MME talking head — EVS"

# 3 — laptop wipe (fixed overhead)
python3 _bake_evs.py \
  --video _src/clip_laptop.mp4 \
  --out-dir examples/3_laptop \
  --q 0.75 --sample-fps 2 --prep-width 640 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --id laptop_evs --videomme-id 7iXM5aq53Ts \
  --label "Video-MME laptop wipe — EVS"

# 4 — gym resistance band (fixed camera)
python3 _bake_evs.py \
  --video _src/clip_gym.mp4 \
  --out-dir examples/4_gym \
  --q 0.75 --sample-fps 2 --prep-width 640 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --id gym_evs --videomme-id 21q-lDikdBg \
  --label "Video-MME gym band — EVS"

# 5 — makeup (fixed backdrop, hand motion)
python3 _bake_evs.py \
  --video _src/clip_makeup.mp4 \
  --out-dir examples/5_makeup \
  --q 0.75 --sample-fps 2 --prep-width 640 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --id makeup_evs --videomme-id 6NVr0cNiHPM \
  --label "Video-MME makeup — EVS"

# 6 — egg flip (colorful kitchen)
python3 _bake_evs.py \
  --video _src/clip_egg.mp4 \
  --out-dir examples/6_egg \
  --q 0.75 --sample-fps 2 --prep-width 640 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --id egg_evs --videomme-id 1sTQOxXFO44 \
  --label "Video-MME egg flip — EVS"

# 7 — outdoor workout (nature / grass)
python3 _bake_evs.py \
  --video _src/clip_outdoor.mp4 \
  --out-dir examples/7_outdoor \
  --q 0.75 --sample-fps 2 --prep-width 640 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --id outdoor_evs --videomme-id 6EIrArTyLVU \
  --label "Video-MME outdoor workout — EVS"
```

Requires: `torch`, `transformers`, `qwen-vl-utils`, `ffmpeg`. Debug `frames/*.jpg` are sampled from the source MP4 (not model tensors).

The bake script loads [`vllm/vllm/multimodal/evs.py`](../vllm/vllm/multimodal/evs.py) via **importlib** (file path) so it uses the exact `compute_retention_mask` without importing the full vLLM package.

## Synthetic pack

```bash
cd public/pack
python3 _make_synthetic_pack.py
```

## Record a GIF / video

1. Load the page (cue: step 0, no overlay).  
2. Start screen capture.  
3. Press `Space`.  
4. Capture one full cycle (original → heat → prune).

## Tests

```bash
cd evs_demo
python3 tests/test_pack_invariants.py
# or: python3 -m pytest tests/ -v
```

## Pack schema

See `src/lib/types.ts` (`EvsPack`). Important fields:

- `grid`: `[W', H']` (width first, after spatial merge)  
- `num_frames`: `T` EVS temporal steps  
- `retention_mask`: `[T][W'*H']` booleans  
- `dissimilarity`: `[T][W'*H']` floats (step 0 = high sentinel)  
