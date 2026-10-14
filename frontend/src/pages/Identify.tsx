import { History, ScanSearch } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { EvidenceBreakdown } from "../components/EvidenceBreakdown";
import { ImageDropzone } from "../components/ImageDropzone";
import { MatchViewer } from "../components/MatchViewer";
import { useToast } from "../components/Toast";
import { WebcamCapture } from "../components/WebcamCapture";
import { EmptyState, ErrorBox, PageHeader, ScoreBar, Section, StatusBadge, fmt } from "../components/ui";
import { useObjectUrls } from "../hooks/useApi";
import { api, url } from "../services/api";
import type { IdentifyResponse } from "../types";

const STAGES = ["Upload & validate", "Normalise image", "Extract features", "Match against registered fingerprints", "Aggregate & decide"];

export default function Identify() {
  const [file, setFile] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(-1);
  const [error, setError] = useState<string>();
  const [res, setRes] = useState<IdentifyResponse>();
  const [sel, setSel] = useState(0);
  const [recording, setRecording] = useState(false);
  const [recorded, setRecorded] = useState<string>();
  const toast = useToast();
  const [preview] = useObjectUrls(file);

  const run = async () => {
    if (!file[0]) return;
    setBusy(true);
    setError(undefined);
    setRes(undefined);
    setRecorded(undefined);
    setStage(0);
    const timer = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 450);
    try {
      const r = await api.identify(file[0], 5);
      setRes(r);
      setSel(0);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      clearInterval(timer);
      setStage(-1);
      setBusy(false);
    }
  };

  const record = async (force: boolean) => {
    const c = res?.candidates[sel];
    if (!c || !file[0]) return;
    setRecording(true);
    try {
      const r = await api.addObservation(c.object_public_id, file[0], undefined, force);
      if (r.accepted) {
        setRecorded(c.object_public_id);
        toast("success", `Observation added to ${c.name}'s history.`);
      } else toast("error", `Not recorded: ${r.reason}`);
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setRecording(false);
    }
  };

  const cand = res?.candidates[sel];
  return (
    <div className="space-y-6">
      <PageHeader title="Identify Object" subtitle="Is this a physical object that has been registered before? The system ranks registered instances and only reports a match when the evidence passes the acceptance criteria." />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="self-start"><Section title="Query image">
          {file[0] ? (
            <div className={`relative overflow-hidden rounded-lg ${busy ? "scanline" : ""}`}>
              <img src={preview} alt="Query" className="max-h-80 w-full rounded-lg object-contain bg-graphite-950" />
            </div>
          ) : (
            <ImageDropzone files={file} onChange={setFile} multiple={false} />
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            <button className="btn-primary" onClick={run} disabled={!file[0] || busy}><ScanSearch className="h-4 w-4" />{busy ? "Identifying…" : "Identify"}</button>
            {file[0] && !busy && <button className="btn-secondary" onClick={() => { setFile([]); setRes(undefined); }}>Change image</button>}
          </div>
          {!file[0] && <div className="mt-3"><WebcamCapture onCapture={(f) => setFile([f])} /></div>}
          {busy && (
            <ol className="mt-4 space-y-1 text-xs" aria-live="polite">
              {STAGES.map((s, i) => <li key={s} className={i < stage ? "text-emerald-400" : i === stage ? "text-accent" : "text-slate-600"}>{i < stage ? "✓" : i === stage ? "›" : "·"} {s}</li>)}
            </ol>
          )}
          {error && <div className="mt-4"><ErrorBox message={error} /></div>}
        </Section></div>

        <div className="space-y-6 lg:col-span-2">
          {!res ? (
            <EmptyState title="No result yet" icon={<ScanSearch className="h-8 w-8" />}>Upload a photo of an object and press Identify. Register objects first in <Link className="text-accent" to="/register">Register Object</Link>.</EmptyState>
          ) : (
            <>
              <div className={`card card-pad border-l-4 ${res.status === "MATCH_FOUND" ? "border-l-emerald-400" : res.status === "UNCERTAIN" ? "border-l-amber-400" : "border-l-rose-400"}`}>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <StatusBadge status={res.status} large />
                  <span className="font-mono text-xs text-slate-500">{res.inference_ms.toFixed(0)} ms · {res.references_compared} reference images compared</span>
                </div>
                {res.status === "MATCH_FOUND" && (
                  <div className="mt-3">
                    <div className="text-lg font-semibold text-slate-50">{res.predicted_name}</div>
                    <div className="font-mono text-sm text-accent">{res.predicted_object_id} · <span className="text-slate-400">{res.predicted_category}</span></div>
                  </div>
                )}
                {res.status === "UNCERTAIN" && <p className="mt-3 text-sm text-amber-200">Candidates exist, but the evidence is insufficient to confirm an identity. No identity is assigned.</p>}
                {res.status === "NO_RELIABLE_MATCH" && <p className="mt-3 text-sm text-rose-200">No registered object passes the acceptance criteria. This may be an unregistered object.</p>}
                <p className="mt-2 text-xs text-slate-400">{res.reason}</p>
                <p className="mt-1 font-mono text-[11px] text-slate-500">accept ≥ {res.thresholds.accept.toFixed(3)} · uncertain ≥ {res.thresholds.uncertain.toFixed(3)} · margin ≥ {res.thresholds.margin.toFixed(3)} · fusion: {res.fusion.source}</p>
                {!res.embedding.available && <p className="mt-1 text-[11px] text-amber-300/80">Deep embedding unavailable — decision used classical evidence only.</p>}
                {res.query_quality.warnings && res.query_quality.warnings.length > 0 && <p className="mt-1 text-[11px] text-amber-300/80">Query quality: {res.query_quality.warnings.join(" ")}</p>}
                {res.candidates.length > 0 && res.status !== "NO_RELIABLE_MATCH" && (
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    {recorded ? (
                      <Link to={`/history?object=${recorded}`} className="btn-secondary"><History className="h-4 w-4" /> View object history</Link>
                    ) : res.status === "MATCH_FOUND" && sel === 0 ? (
                      <button className="btn-secondary" onClick={() => record(false)} disabled={recording}><History className="h-4 w-4" />{recording ? "Recording…" : "Add to object history"}</button>
                    ) : res.status === "UNCERTAIN" && sel === 0 ? (
                      <button className="btn-secondary" onClick={() => record(true)} disabled={recording} title="Stores the observation flagged as user-confirmed">
                        <History className="h-4 w-4" />{recording ? "Recording…" : `I confirm this is ${res.candidates[0].name} — record`}
                      </button>
                    ) : null}
                  </div>
                )}
              </div>

              {res.candidates.length === 0 ? <EmptyState title="No registered objects to compare against" /> : (
                <Section title="Ranked candidates" subtitle="Object score = best fused score over that object's reference images. Click a candidate to inspect its evidence.">
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {res.candidates.map((c, i) => (
                      <button key={c.object_public_id} onClick={() => setSel(i)}
                        className={`flex gap-3 rounded-lg border p-2 text-left transition ${sel === i ? "border-accent bg-accent/5" : "border-graphite-700 hover:border-graphite-500"}`}>
                        <img src={url(c.best_reference_url)} alt="" className="h-16 w-16 rounded object-cover" loading="lazy" />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1 text-xs text-slate-500">#{c.rank} <span className="truncate font-mono">{c.object_public_id}</span></div>
                          <div className="truncate text-sm font-medium text-slate-100">{c.name}</div>
                          <ScoreBar value={c.score} color={i === 0 && res.status === "MATCH_FOUND" ? "bg-emerald-400" : "bg-accent"} label={`Similarity score ${c.score}`} />
                          <div className="text-[10px] text-slate-500">{c.supporting_references}/{c.references_compared} references above uncertain threshold</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </Section>
              )}

              {cand && (
                <Section title={`Evidence: query vs ${cand.name}`} subtitle={`Best-matching reference image #${cand.best_reference_image_id} · similarity score ${fmt(cand.score)} (not a probability)`}>
                  <div className="grid gap-6 2xl:grid-cols-5">
                    <div className="2xl:col-span-3">
                      {cand.matches.length > 0 ? (
                        <MatchViewer left={{ src: url(res.query_image_url), label: "Query", size: res.query_processed_size }} right={{ src: url(cand.best_reference_url), label: `Reference #${cand.best_reference_image_id}` }} matches={cand.matches} />
                      ) : (
                        <div className="grid grid-cols-2 gap-2">
                          <img src={url(res.query_image_url)} alt="Query" className="rounded" />
                          <img src={url(cand.best_reference_url)} alt="Reference" className="rounded" />
                          <p className="col-span-2 text-xs text-slate-500">No ORB correspondences to display for this candidate.</p>
                        </div>
                      )}
                    </div>
                    <div className="2xl:col-span-2"><EvidenceBreakdown fused={cand.fused} evidence={cand.evidence} /></div>
                  </div>
                  <details className="mt-4 text-xs text-slate-400">
                    <summary className="cursor-pointer">Per-reference scores for this object</summary>
                    <ul className="mt-2 grid gap-1 font-mono sm:grid-cols-3">{cand.reference_scores.map((r) => <li key={r.image_id}>ref #{r.image_id}: {r.score.toFixed(3)}</li>)}</ul>
                  </details>
                </Section>
              )}
              <p className="text-[11px] text-slate-500">{res.note}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
