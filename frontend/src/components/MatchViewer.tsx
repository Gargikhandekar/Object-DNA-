import { useEffect, useMemo, useState } from "react";
import type { OrbMatch } from "../types";

interface Side { src: string; label: string; size?: [number, number] }

function useNaturalSize(src: string, known?: [number, number]): [number, number] | undefined {
  const [size, setSize] = useState<[number, number] | undefined>(known);
  useEffect(() => {
    if (known) { setSize(known); return; }
    const img = new Image();
    img.onload = () => setSize([img.naturalWidth, img.naturalHeight]);
    img.src = src;
  }, [src, known]);
  return size;
}

/**
 * Side-by-side view of two processed images with ORB correspondences.
 * Coordinates from the API refer to the processed images, which are shown here.
 * Cyan = RANSAC inlier (geometrically consistent), rose = outlier.
 */
export function MatchViewer({ left, right, matches, maxLines = 250 }: { left: Side; right: Side; matches: OrbMatch[]; maxLines?: number }) {
  const ls = useNaturalSize(left.src, left.size);
  const rs = useNaturalSize(right.src, right.size);
  const [showIn, setShowIn] = useState(true);
  const [showOut, setShowOut] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const nIn = matches.filter((m) => m.inlier).length;

  const shown = useMemo(() => {
    const list = matches.map((m, i) => ({ ...m, i })).filter((m) => (m.inlier ? showIn : showOut));
    return list.slice(0, maxLines);
  }, [matches, showIn, showOut, maxLines]);

  if (!ls || !rs) return <div className="aspect-[2/1] w-full animate-pulse rounded-lg bg-graphite-800" />;
  const H = 480;
  const s1 = H / ls[1];
  const s2 = H / rs[1];
  const w1 = ls[0] * s1;
  const gap = 16;
  const W = w1 + gap + rs[0] * s2;

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-4 text-xs">
        <label className="flex items-center gap-1.5 text-slate-300">
          <input type="checkbox" checked={showIn} onChange={(e) => setShowIn(e.target.checked)} className="accent-cyan-400" />
          <span className="inline-block h-0.5 w-4 bg-cyan-400" /> Geometrically consistent ({nIn})
        </label>
        <label className="flex items-center gap-1.5 text-slate-300">
          <input type="checkbox" checked={showOut} onChange={(e) => setShowOut(e.target.checked)} className="accent-rose-400" />
          <span className="inline-block h-0.5 w-4 bg-rose-400" /> Outliers ({matches.length - nIn})
        </label>
        {shown.length >= maxLines && <span className="text-slate-500">showing first {maxLines}</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H + 22}`} className="w-full rounded-lg bg-graphite-950" role="img"
        aria-label={`${nIn} geometrically consistent keypoint matches between ${left.label} and ${right.label}`}>
        <image href={left.src} x={0} y={0} width={w1} height={H} />
        <image href={right.src} x={w1 + gap} y={0} width={rs[0] * s2} height={H} />
        {shown.map((m) => {
          const x1 = m.qx * s1, y1 = m.qy * s1, x2 = w1 + gap + m.rx * s2, y2 = m.ry * s2;
          const active = hover === m.i;
          const color = m.inlier ? "#22d3ee" : "#fb7185";
          return (
            <g key={m.i} onMouseEnter={() => setHover(m.i)} onMouseLeave={() => setHover(null)}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={active ? 2.5 : 1} strokeOpacity={hover === null ? 0.65 : active ? 1 : 0.15} />
              <circle cx={x1} cy={y1} r={active ? 4 : 2.2} fill={color} />
              <circle cx={x2} cy={y2} r={active ? 4 : 2.2} fill={color} />
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" strokeWidth={8} />
            </g>
          );
        })}
        <text x={6} y={H + 16} fill="#94a3b8" fontSize={13}>{left.label}</text>
        <text x={w1 + gap + 6} y={H + 16} fill="#94a3b8" fontSize={13}>{right.label}</text>
      </svg>
      {hover !== null && matches[hover] && (
        <p className="mt-1 font-mono text-[11px] text-slate-400">
          match #{hover}: ({matches[hover].qx.toFixed(0)}, {matches[hover].qy.toFixed(0)}) → ({matches[hover].rx.toFixed(0)}, {matches[hover].ry.toFixed(0)}) · {matches[hover].inlier ? "RANSAC inlier" : "outlier"}
        </p>
      )}
    </div>
  );
}
