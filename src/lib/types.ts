/** Precomputed NVIDIA EVS pack — masks per temporal soft-token step. */

export interface EvsPack {
  id: string;
  label: string;
  model: "qwen2_5_vl";
  q: number;
  /** Soft-token grid after spatial merge: [W', H'] (width first). */
  grid: [number, number];
  /** T = EVS temporal steps (== video_grid_thw[0]). */
  num_frames: number;
  tokens_per_frame: number;
  kept_total: number;
  spatial_merge_size: number;
  temporal_patch_size: number;
  /** Fluid source video under /pack/ (preferred). */
  video?: string;
  video_duration?: number;
  video_width?: number;
  video_height?: number;
  /** HTML video playbackRate (1 = realtime). */
  playback_rate?: number;
  /** Legacy: one JPEG per temporal step (fallback if no video). */
  frames: string[];
  /** Playback fps over steps when falling back to stills. */
  fps: number;
  /** [T][W'*H'] row-major (row = y, col = x). */
  retention_mask: boolean[][];
  /** [T][W'*H']; step 0 is a high sentinel. */
  dissimilarity: number[][];
}

/** OverlayCanvas-compatible metadata for a single temporal step. */
export interface StepMetadata {
  method: "evs";
  grid: [number, number];
  num_tokens: number;
  retention: number;
  pruned: number[];
  kept: number[];
}

export function maskToMetadata(pack: EvsPack, t: number): StepMetadata {
  const step = Math.max(0, Math.min(pack.num_frames - 1, t));
  const mask = pack.retention_mask[step] ?? [];
  const kept: number[] = [];
  const pruned: number[] = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) kept.push(i);
    else pruned.push(i);
  }
  const retention =
    pack.tokens_per_frame > 0 ? kept.length / pack.tokens_per_frame : 0;
  return {
    method: "evs",
    grid: pack.grid,
    num_tokens: pack.tokens_per_frame,
    retention,
    pruned,
    kept,
  };
}

export function stepKeepCount(pack: EvsPack, t: number): number {
  const mask = pack.retention_mask[t];
  if (!mask) return 0;
  let n = 0;
  for (const v of mask) if (v) n += 1;
  return n;
}

export function videoUrl(pack: EvsPack): string | null {
  if (!pack.video) return null;
  return `/pack/${pack.video}`;
}

/** Resolve video URL when pack.json lives under an example subdirectory. */
export function videoUrlForBase(pack: EvsPack, baseUrl: string): string | null {
  if (!pack.video) return null;
  const base = baseUrl.replace(/\/$/, "");
  if (pack.video.startsWith("http") || pack.video.startsWith("/")) {
    return pack.video;
  }
  return `${base}/${pack.video}`;
}

/** Map continuous video time → EVS temporal step index. */
export function timeToStep(
  currentTime: number,
  duration: number,
  numSteps: number
): number {
  if (numSteps <= 1) return 0;
  if (!(duration > 0)) return 0;
  const t = Math.min(1, Math.max(0, currentTime / duration));
  return Math.min(numSteps - 1, Math.floor(t * numSteps));
}

export function stepToTime(
  step: number,
  duration: number,
  numSteps: number
): number {
  if (numSteps <= 0) return 0;
  return (Math.max(0, Math.min(numSteps - 1, step)) / numSteps) * duration;
}
