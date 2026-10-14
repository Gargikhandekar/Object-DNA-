import { GitCompareArrows } from "lucide-react";
import { useState } from "react";
import { EvidenceBreakdown } from "../components/EvidenceBreakdown";
import { ImageDropzone } from "../components/ImageDropzone";
import { MatchViewer } from "../components/MatchViewer";
import { EmptyState, ErrorBox, PageHeader, Section, fmt } from "../components/ui";
import { api } from "../services/api";
import type { CompareResponse } from "../types";

const DECISION_STYLE = {
  SAME_INSTANCE: "border-emerald-400/50 bg-emerald-400/10 text-emerald-300",
  INCONCLUSIVE: "border-amber-400/50 bg-amber-400/10 text-amber-300",
  DIFFERENT_INSTANCE: "border-rose-400/50 bg-rose-400/10 text-rose-300",
};

export default function Compare() {
  const [a, setA] = useState<File[]>([]);
  const [b, setB] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [res, setRes] = useState<CompareResponse>();

  const run = async () => {
    if (!a[0] || !b[0]) return;
    setBusy(true);
    setError(undefined);
    try {
      setRes(await api.compare(a[0], b[0]));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Compare Images" subtitle="Same-instance verification: do these two photos show the same physical object? Uses the same features, fusion and thresholds as identification." />
      <div className="grid gap-6 md:grid-cols-2">
        {[{ f: a, set: setA, label: "Image A" }, { f: b, set: setB, label: "Image B" }].map(({ f, set, label }) => (
          <Section key={label} title={label}>
            <ImageDropzone files={f} onChange={(x) => { set(x); setRes(undefined); }} multiple={false} compact label={f[0] ? `Selected: ${f[0].name} (click to replace)` : undefined} />
          </Section>
        ))}
      </div>
      <div className="flex gap-3">
        <button className="btn-primary" disabled={!a[0] || !b[0] || busy} onClick={run}><GitCompareArrows className="h-4 w-4" />{busy ? "Comparing…" : "Compare"}</button>
        {error && <div className="flex-1"><ErrorBox message={error} /></div>}
      </div>

      {!res ? (
        !busy && <EmptyState title="Select two images to compare" icon={<GitCompareArrows className="h-8 w-8" />} />
      ) : (
        <>
          <div className="card card-pad">
            <div className="flex flex-wrap items-center gap-4">
              <span className={`rounded-full border px-4 py-1.5 text-sm font-semibold tracking-wide ${DECISION_STYLE[res.decision]}`}>{res.decision.replace("_", " ")}</span>
              <span className="font-mono text-sm text-slate-300">fused similarity {fmt(res.score)}</span>
              <span className="font-mono text-xs text-slate-500">{res.inference_ms.toFixed(0)} ms</span>
            </div>
            <p className="mt-2 text-sm text-slate-400">{res.reason}</p>
            <div className="mt-4 grid gap-3 text-sm sm:grid-cols-4">
              <Metric label="Deep cosine" value={res.evidence.raw.deep == null ? "unavailable" : fmt(res.evidence.raw.deep)} />
              <Metric label="ORB inliers / matches" value={res.evidence.orb ? `${res.evidence.orb.inliers} / ${res.evidence.orb.good_matches}` : "—"} />
              <Metric label="LBP texture" value={fmt(res.evidence.raw.texture)} />
              <Metric label="HSV colour" value={fmt(res.evidence.raw.color)} />
            </div>
            <div className="mt-3 rounded-lg border border-graphite-700 bg-graphite-850 p-3 text-xs text-slate-400">
              Raw pixel similarity (64×64 grayscale NCC): <span className="font-mono text-slate-200">{res.raw_pixel_similarity.toFixed(3)}</span>.
              Images of the same object can have low pixel similarity because of viewpoint, scale, background or lighting changes — which is why the decision relies on local keypoints, texture and embeddings instead.
            </div>
          </div>
          <div className="grid gap-6 xl:grid-cols-5">
            <div className="xl:col-span-3">
              <Section title="ORB keypoint correspondences" subtitle="Cyan lines passed RANSAC geometric verification; rose lines are outliers. Hover a line to inspect it.">
                <MatchViewer left={{ src: res.image_a, label: "Image A", size: res.size_a }} right={{ src: res.image_b, label: "Image B", size: res.size_b }} matches={res.matches} />
              </Section>
            </div>
            <div className="xl:col-span-2">
              <Section title="Evidence breakdown"><EvidenceBreakdown fused={res.fused} evidence={res.evidence} /></Section>
            </div>
          </div>
          <p className="text-[11px] text-slate-500">{res.note}</p>
        </>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-graphite-850 p-3">
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-slate-100">{value}</div>
    </div>
  );
}
