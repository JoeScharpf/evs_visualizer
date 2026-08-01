import { useCallback, useEffect, useRef } from "react";
import type { StepMetadata } from "./lib/types";

const KEEP = "rgb(212,175,55)";
/** Semi-transparent so the live video still reads underneath. */
const PRUNED_FILL = "rgba(0,0,0,0.62)";

export type HeatPalette = "jet" | "magma";

function heatColorJet(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  let r = 0;
  let g = 0;
  let b = 0;
  if (x < 0.25) {
    const u = x / 0.25;
    r = 0;
    g = Math.round(40 + 215 * u);
    b = 255;
  } else if (x < 0.5) {
    const u = (x - 0.25) / 0.25;
    r = 0;
    g = 255;
    b = Math.round(255 * (1 - u));
  } else if (x < 0.75) {
    const u = (x - 0.5) / 0.25;
    r = Math.round(255 * u);
    g = 255;
    b = 0;
  } else {
    const u = (x - 0.75) / 0.25;
    r = 255;
    g = Math.round(255 * (1 - 0.85 * u));
    b = 0;
  }
  return `rgb(${r},${g},${b})`;
}

function heatColorMagma(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  const stops: Array<[number, number, number, number]> = [
    [0.0, 0, 0, 4],
    [0.25, 80, 18, 123],
    [0.5, 183, 55, 121],
    [0.75, 251, 135, 97],
    [1.0, 252, 253, 191],
  ];
  let i = 0;
  while (i < stops.length - 2 && x > stops[i + 1][0]) i += 1;
  const [t0, r0, g0, b0] = stops[i];
  const [t1, r1, g1, b1] = stops[i + 1];
  const u = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
  return `rgb(${Math.round(r0 + (r1 - r0) * u)},${Math.round(g0 + (g1 - g0) * u)},${Math.round(b0 + (b1 - b0) * u)})`;
}

function normalizeScores(scores: number[]): number[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of scores) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  if (!(span > 0) || !Number.isFinite(span)) {
    return scores.map(() => 0);
  }
  return scores.map((v) => (v - lo) / span);
}

export type OverlayMode = "none" | "heat" | "prune";

/**
 * Transparent patch overlay sized to the parent panel.
 * Does not redraw the video — only darkens pruned cells / paints heat / outlines kept.
 */
export function OverlayCanvas({
  metadata,
  mode = "none",
  heatScores,
  heatPalette = "magma",
  panelW,
  panelH,
}: {
  metadata: StepMetadata;
  mode?: OverlayMode;
  heatScores?: number[];
  heatPalette?: HeatPalette;
  panelW: number;
  panelH: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const [gridW, gridH] = metadata.grid;
    const nextW = Math.max(1, Math.round(panelW));
    const nextH = Math.max(1, Math.round(panelH));
    if (canvas.width !== nextW || canvas.height !== nextH) {
      canvas.width = nextW;
      canvas.height = nextH;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (mode === "none") return;

    const cellW = canvas.width / gridW;
    const cellH = canvas.height / gridH;
    const box = (idx: number): [number, number, number, number] => {
      const col = idx % gridW;
      const row = Math.floor(idx / gridW);
      return [col * cellW, row * cellH, cellW, cellH];
    };

    if (mode === "heat" && heatScores && heatScores.length === metadata.num_tokens) {
      const norm = normalizeScores(heatScores);
      ctx.fillStyle = "rgba(12,10,9,0.22)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (let idx = 0; idx < norm.length; idx++) {
        const [x, y, w, h] = box(idx);
        ctx.globalAlpha = 0.72;
        ctx.fillStyle =
          heatPalette === "magma"
            ? heatColorMagma(norm[idx])
            : heatColorJet(norm[idx]);
        ctx.fillRect(x + 0.5, y + 0.5, w - 1, h - 1);
      }
      ctx.globalAlpha = 1;
      return;
    }

    if (mode === "prune") {
      // Only darken pruned cells — do NOT outline every kept cell.
      // (Step 0 keeps everything; outlining all tokens looks like a static grid.)
      ctx.fillStyle = PRUNED_FILL;
      for (const idx of metadata.pruned) {
        const [x, y, w, h] = box(idx);
        ctx.fillRect(x, y, w, h);
      }
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      for (const idx of metadata.pruned) {
        const [x, y, w, h] = box(idx);
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      }
      // Outline kept cells only when sparse enough to read as motion hotspots
      const keepRatio =
        metadata.num_tokens > 0
          ? metadata.kept.length / metadata.num_tokens
          : 1;
      if (keepRatio < 0.55) {
        ctx.lineWidth = Math.max(1.5, Math.min(cellW, cellH) * 0.06);
        ctx.strokeStyle = KEEP;
        for (const idx of metadata.kept) {
          const [x, y, w, h] = box(idx);
          const inset = ctx.lineWidth;
          ctx.strokeRect(x + inset, y + inset, w - inset * 2, h - inset * 2);
        }
      }
    }
  }, [metadata, mode, heatScores, heatPalette, panelW, panelH]);

  useEffect(() => {
    paint();
  }, [paint]);

  return (
    <div className="absolute inset-0 pointer-events-none">
      <canvas
        ref={canvasRef}
        className="h-full w-full block pointer-events-none"
      />
    </div>
  );
}

export function OverlayLegend({ mode }: { mode: OverlayMode }) {
  if (mode === "none") return null;
  if (mode === "heat") {
    return (
      <div className="flex items-center gap-3 text-[11px] text-fg-muted">
        <span className="demo-label text-fg-muted">Dissimilarity</span>
        <span>low → high (magma)</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-4 text-[11px] text-fg-muted">
      <span className="inline-flex items-center gap-1.5">
        <span
          className="inline-block h-2.5 w-2.5 border-2"
          style={{ borderColor: KEEP }}
        />
        kept
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span
          className="inline-block h-2.5 w-2.5"
          style={{ background: "rgba(0,0,0,0.75)" }}
        />
        pruned
      </span>
    </div>
  );
}
