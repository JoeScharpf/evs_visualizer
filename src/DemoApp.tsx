/**
 * EVS multi-example demo (white page, video + prune overlay).
 *
 * Keys 1..N: first press cues that example (paused at start, no overlay);
 * press the same number again to play with prune overlay.
 * Space: pause / resume (keeps overlay + current time). Esc freezes back to cue.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { OverlayCanvas } from "./OverlayCanvas";
import {
  maskToMetadata,
  timeToStep,
  videoUrlForBase,
  type EvsPack,
} from "./lib/types";

const MANIFEST_URL = "/pack/examples.json";
const PANEL_MAX_H = 560;

type ExampleEntry = {
  key: number;
  id: string;
  pack: string;
  title?: string;
};

type Manifest = { examples: ExampleEntry[] };

/** cue = start frame, no overlay; playing = running; paused = freeze mid-clip with overlay */
type Phase = "cue" | "playing" | "paused";

export default function DemoApp() {
  const [examples, setExamples] = useState<ExampleEntry[]>([]);
  const [selectedKey, setSelectedKey] = useState<number | null>(null);
  const [pack, setPack] = useState<EvsPack | null>(null);
  const [packBase, setPackBase] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("cue");
  const [stepIdx, setStepIdx] = useState(0);
  const [viewportW, setViewportW] = useState(
    typeof window !== "undefined" ? window.innerWidth : 1200
  );
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const selectedKeyRef = useRef<number | null>(null);
  const phaseRef = useRef<Phase>("cue");

  useEffect(() => {
    selectedKeyRef.current = selectedKey;
  }, [selectedKey]);
  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    const onResize = () => setViewportW(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const armCue = useCallback(() => {
    const v = videoRef.current;
    if (v) {
      v.pause();
      v.currentTime = 0;
    }
    setPhase("cue");
    setStepIdx(0);
  }, []);

  const startPlayback = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = 0;
    setStepIdx(0);
    setPhase("playing");
    void v.play().catch(() => setPhase("cue"));
  }, []);

  const pausePlayback = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    setPhase("paused");
  }, []);

  const resumePlayback = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    setPhase("playing");
    void v.play().catch(() => setPhase("paused"));
  }, []);

  const loadExample = useCallback(
    async (entry: ExampleEntry, autoPlay: boolean) => {
      const packPath = `/pack/${entry.pack.replace(/^\//, "")}`;
      const base = packPath.replace(/\/pack\.json$/, "");
      const data = (await fetch(packPath).then((r) => {
        if (!r.ok) throw new Error(`pack ${r.status}`);
        return r.json();
      })) as EvsPack;
      setPack(data);
      setPackBase(base);
      setSelectedKey(entry.key);
      setStepIdx(0);
      setPhase("cue");
      requestAnimationFrame(() => {
        const v = videoRef.current;
        if (v) {
          v.pause();
          v.currentTime = 0;
          if (autoPlay) {
            setPhase("playing");
            void v.play().catch(() => setPhase("cue"));
          }
        }
      });
    },
    []
  );

  useEffect(() => {
    let cancelled = false;
    fetch(MANIFEST_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`manifest ${r.status}`);
        return r.json();
      })
      .then(async (data: Manifest) => {
        if (cancelled) return;
        const list = (data.examples ?? []).slice().sort((a, b) => a.key - b.key);
        setExamples(list);
        for (const entry of list) {
          const packPath = `/pack/${entry.pack.replace(/^\//, "")}`;
          const res = await fetch(packPath, { method: "HEAD" });
          if (res.ok) {
            if (!cancelled) await loadExample(entry, false);
            return;
          }
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [loadExample]);

  const duration = pack?.video_duration ?? 0;
  const vUrl = pack ? videoUrlForBase(pack, packBase) : null;

  const syncFromVideo = useCallback(() => {
    const v = videoRef.current;
    if (!v || !pack) return;
    if (phaseRef.current === "cue") return;
    const dur =
      duration > 0 ? duration : Number.isFinite(v.duration) ? v.duration : 1;
    setStepIdx(timeToStep(v.currentTime, dur, pack.num_frames));
  }, [pack, duration]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !vUrl) return;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    if (phase === "playing") {
      if (v.paused) void v.play().catch(() => setPhase("paused"));
    } else {
      v.pause();
      if (phase === "cue" && v.currentTime !== 0) v.currentTime = 0;
    }
  }, [vUrl, phase]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        const cur = phaseRef.current;
        if (cur === "playing") pausePlayback();
        else if (cur === "paused") resumePlayback();
        else startPlayback();
        return;
      }

      if (e.key === "Escape") {
        e.preventDefault();
        armCue();
        return;
      }

      const digit = /^[1-9]$/.test(e.key) ? Number(e.key) : null;
      if (digit == null) return;
      const entry = examples.find((ex) => ex.key === digit);
      if (!entry) return;
      e.preventDefault();

      const curKey = selectedKeyRef.current;
      const curPhase = phaseRef.current;
      if (curKey !== digit) {
        void loadExample(entry, false);
        return;
      }
      // Same example: cue → play; playing/paused → cue
      if (curPhase === "cue") startPlayback();
      else armCue();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    examples,
    loadExample,
    armCue,
    startPlayback,
    pausePlayback,
    resumePlayback,
  ]);

  const metadata = useMemo(
    () => (pack ? maskToMetadata(pack, stepIdx) : null),
    [pack, stepIdx]
  );

  const layout = useMemo(() => {
    if (!pack) return null;
    const vw = pack.video_width ?? 16;
    const vh = pack.video_height ?? 9;
    const panelH = PANEL_MAX_H;
    const panelW = Math.round(PANEL_MAX_H * (vw / vh));
    const scale = Math.min(1, (viewportW - 48) / panelW);
    return { panelW, panelH, scale };
  }, [pack, viewportW]);

  const showOverlay = phase !== "cue" && metadata != null;

  return (
    <div className="h-screen w-screen overflow-hidden flex items-center justify-center bg-white">
      {pack && vUrl && metadata && layout && (
        <div
          style={{
            width: layout.panelW * layout.scale,
            height: layout.panelH * layout.scale,
          }}
        >
          <div
            className="relative"
            style={{
              width: layout.panelW,
              height: layout.panelH,
              transform: `scale(${layout.scale})`,
              transformOrigin: "top left",
            }}
          >
            <div
              className="relative overflow-hidden"
              style={{ width: layout.panelW, height: layout.panelH }}
            >
              <video
                ref={videoRef}
                key={vUrl}
                src={vUrl}
                className="absolute inset-0 h-full w-full object-fill"
                muted
                loop
                playsInline
                preload="auto"
                onTimeUpdate={syncFromVideo}
                onSeeked={syncFromVideo}
                onLoadedMetadata={() => {
                  const v = videoRef.current;
                  if (!v) return;
                  if (phaseRef.current === "playing") {
                    void v.play().catch(() => {});
                  } else if (phaseRef.current === "cue") {
                    v.pause();
                    v.currentTime = 0;
                  } else {
                    v.pause();
                  }
                }}
              />
              {showOverlay && (
                <div className="absolute inset-0">
                  <OverlayCanvas
                    metadata={metadata}
                    mode="prune"
                    panelW={layout.panelW}
                    panelH={layout.panelH}
                    interactive={phase === "paused"}
                    stepIdx={stepIdx}
                    dissimilarity={pack.dissimilarity[stepIdx]}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
