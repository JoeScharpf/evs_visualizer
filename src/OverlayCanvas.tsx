import { useCallback, useEffect, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
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
 * When interactive (paused), hover shows pairwise cosine similarity vs previous step.
 */
export function OverlayCanvas({
  metadata,
  mode = "none",
  heatScores,
  heatPalette = "magma",
  panelW,
  panelH,
  interactive = false,
  stepIdx = 0,
  dissimilarity,
}: {
  metadata: StepMetadata;
  mode?: OverlayMode;
  heatScores?: number[];
  heatPalette?: HeatPalette;
  panelW: number;
  panelH: number;
  /** Enable hit-testing + tooltip (use while paused). */
  interactive?: boolean;
  stepIdx?: number;
  /** Per-token dissimilarity for the current temporal step. */
  dissimilarity?: number[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{
    idx: number;
    xPct: number;
    yPct: number;
  } | null>(null);

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

  useEffect(() => {
    if (!interactive) setHover(null);
  }, [interactive]);

  const [gridW, gridH] = metadata.grid;

  const onMove = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    if (!interactive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const col = Math.min(gridW - 1, Math.max(0, Math.floor(x * gridW)));
    const row = Math.min(gridH - 1, Math.max(0, Math.floor(y * gridH)));
    setHover({ idx: row * gridW + col, xPct: x * 100, yPct: y * 100 });
  };

  const isKept = hover ? metadata.kept.includes(hover.idx) : false;
  const dissimVal =
    hover &&
    dissimilarity &&
    dissimilarity.length === metadata.num_tokens &&
    stepIdx > 0
      ? dissimilarity[hover.idx]
      : undefined;
  const cosVal =
    dissimVal != null && Number.isFinite(dissimVal)
      ? 1 - dissimVal
      : undefined;

  return (
    <div className="absolute inset-0 pointer-events-none">
      <canvas
        ref={canvasRef}
        className={`h-full w-full block ${
          interactive ? "pointer-events-auto cursor-crosshair" : "pointer-events-none"
        }`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      />
      {interactive && hover && mode !== "none" && (
        <div
          className="pointer-events-none absolute z-10 rounded bg-stone-900/90 px-2 py-1 text-[11px] text-white shadow"
          style={{
            left: `min(${hover.xPct}%, calc(100% - 160px))`,
            top: `min(${hover.yPct}%, calc(100% - 56px))`,
            transform: "translate(8px, 8px)",
          }}
        >
          <div className="font-mono">
            #{hover.idx} · {isKept ? "kept" : "pruned"}
          </div>
          {stepIdx === 0 ? (
            <div className="text-stone-300">cos sim n/a (first step)</div>
          ) : cosVal != null ? (
            <div className="text-stone-300">
              cos sim {cosVal.toFixed(3)}
              {dissimVal != null ? ` · dissim ${dissimVal.toFixed(3)}` : ""}
            </div>
          ) : (
            <div className="text-stone-300">cos sim n/a</div>
          )}
        </div>
      )}
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
