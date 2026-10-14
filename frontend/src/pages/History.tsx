import { History as HistoryIcon, Upload } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ImageDropzone } from "../components/ImageDropzone";
import { useToast } from "../components/Toast";
import { EmptyState, ErrorBox, PageHeader, Section, Spinner, StatusBadge, dateTime, fmt } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, url } from "../services/api";
import type { Status } from "../types";

const tooltipStyle = { background: "#161a1f", border: "1px solid #384250", borderRadius: 8, fontSize: 12 };

export default function HistoryPage() {
  const [params, setParams] = useSearchParams();
  const objectId = params.get("object") ?? "";
  const objects = useApi(() => api.listObjects({ page_size: 200 }), []);
  return (
    <div className="space-y-6">
      <PageHeader title="Object History" subtitle="Registration and later observations of one physical object, plus the log of all identification attempts." />
      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="obj" className="text-sm text-slate-400">Object</label>
        <select id="obj" className="input w-80" value={objectId} onChange={(e) => setParams(e.target.value ? { object: e.target.value } : {})}>
          <option value="">— All identification attempts —</option>
          {objects.data?.items.map((o) => <option key={o.public_id} value={o.public_id}>{o.name} · {o.public_id}</option>)}
        </select>
      </div>
      {objectId ? <ObjectTimeline id={objectId} /> : <AttemptLog />}
    </div>
  );
}

function ObjectTimeline({ id }: { id: string }) {
  const { data, error, loading, reload } = useApi(() => api.objectHistory(id), [id]);
  const [file, setFile] = useState<File[]>([]);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [rejection, setRejection] = useState<string>();
  const toast = useToast();

  if (loading && !data) return <Spinner />;
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  if (!data) return null;

  const submit = async (force: boolean) => {
    if (!file[0]) return;
    setBusy(true);
    setRejection(undefined);
    try {
      const r = await api.addObservation(id, file[0], note, force);
      if (r.accepted) {
        toast("success", "Observation accepted and added to the history.");
        setFile([]);
        setNote("");
        reload();
      } else setRejection(`${r.status.replace(/_/g, " ")} (score ${fmt(r.score)}): ${r.reason}`);
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const series = data.observations.map((o) => ({ t: new Date(o.observed_at).toLocaleDateString() + " " + new Date(o.observed_at).toLocaleTimeString(), similarity: o.similarity }));
  const ch = (k: string, o: Record<string, number | null>) => o[k] == null ? "—" : Number(o[k]).toFixed(3);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-3">
        <Section title="Timeline" subtitle={<>First registered {dateTime(data.first_registered)} · <Link className="text-accent" to={`/library/${id}`}>{data.object.name}</Link></>}>
          <ol className="relative space-y-4 border-l border-graphite-700 pl-5">
            <li>
              <span className="absolute -left-1.5 mt-1 h-3 w-3 rounded-full bg-accent" />
              <div className="text-sm text-slate-200">Registered with {data.references.length} reference images</div>
              <div className="text-xs text-slate-500">{dateTime(data.first_registered)}</div>
              <div className="mt-2 flex gap-1">{data.references.slice(0, 6).map((r) => <img key={r.id} src={url(r.url)} alt="" className="h-10 w-10 rounded object-cover" />)}</div>
            </li>
            {data.observations.map((o) => (
              <li key={o.id}>
                <span className={`absolute -left-1.5 mt-1 h-3 w-3 rounded-full ${o.status === "ACCEPTED" ? "bg-emerald-400" : "bg-amber-400"}`} />
                <div className="text-sm text-slate-200">Observation · score {o.similarity.toFixed(3)}</div>
                <div className="text-xs text-slate-500">{dateTime(o.observed_at)} · {o.status.replace(/_/g, " ").toLowerCase()}</div>
                {o.note && <div className="text-xs italic text-slate-400">“{o.note}”</div>}
              </li>
            ))}
          </ol>
        </Section>
        <Section title="Similarity over time" subtitle="Fused similarity of each accepted observation to the registered references">
          {series.length === 0 ? <EmptyState title="No observations yet" /> : (
            <div className="h-56">
              <ResponsiveContainer>
                <LineChart data={series}>
                  <CartesianGrid stroke="#272e37" />
                  <XAxis dataKey="t" stroke="#5b6675" fontSize={10} />
                  <YAxis domain={[0, 1]} stroke="#5b6675" fontSize={11} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <ReferenceLine y={0.5} stroke="#384250" strokeDasharray="4 4" />
                  <Line dataKey="similarity" stroke="#22d3ee" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Section>
        <Section title="Submit a later image" subtitle="Accepted only if this object is the top-ranked match and passes the acceptance threshold.">
          <ImageDropzone files={file} onChange={setFile} multiple={false} compact label={file[0] ? file[0].name : undefined} />
          <input className="input mt-3" placeholder="Note (optional), e.g. after travel" value={note} onChange={(e) => setNote(e.target.value)} maxLength={2000} />
          <button className="btn-primary mt-3 w-full" disabled={!file[0] || busy} onClick={() => submit(false)}><Upload className="h-4 w-4" />{busy ? "Verifying…" : "Verify & add observation"}</button>
          {rejection && (
            <div className="mt-3 space-y-2">
              <ErrorBox message={`Not added. ${rejection}`} />
              {rejection.startsWith("UNCERTAIN") && <button className="btn-secondary w-full text-xs" disabled={busy} onClick={() => submit(true)}>I am certain this is the same object — record as user-confirmed</button>}
            </div>
          )}
        </Section>
      </div>
      {data.observations.length > 0 && (
        <Section title="Observations & measured surface characteristics" subtitle={data.note}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-left text-slate-500">
                <tr><th className="py-1">Image</th><th>Date</th><th>Score</th><th>Texture sim. (max)</th><th>Colour sim. (max)</th><th>Edge density Δ</th><th>Sharpness (obs / ref)</th><th>Brightness (obs / ref)</th><th>Keypoints (obs / ref)</th></tr>
              </thead>
              <tbody className="font-mono">
                {data.observations.map((o) => {
                  const s = o.surface_changes;
                  return (
                    <tr key={o.id} className="border-t border-graphite-800">
                      <td className="py-1.5"><a href={url(o.image_url)} target="_blank" rel="noreferrer"><img src={url(o.image_url)} alt="" className="h-12 w-12 rounded object-cover" /></a></td>
                      <td className="font-sans">{dateTime(o.observed_at)}</td>
                      <td>{o.similarity.toFixed(3)}</td>
                      <td>{ch("texture_similarity_max", s)}</td>
                      <td>{ch("color_similarity_max", s)}</td>
                      <td>{ch("edge_density_change", s)}</td>
                      <td>{s.sharpness ?? "—"} / {s.sharpness_reference_mean ?? "—"}</td>
                      <td>{s.brightness ?? "—"} / {s.brightness_reference_mean ?? "—"}</td>
                      <td>{s.keypoints ?? "—"} / {s.keypoints_reference_mean ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </div>
  );
}

function AttemptLog() {
  const [status, setStatus] = useState<Status | "">("");
  const [page, setPage] = useState(1);
  const { data, error, loading, reload } = useApi(() => api.history({ page, page_size: 25, status }), [page, status]);
  const pages = data ? Math.max(1, Math.ceil(data.total / 25)) : 1;
  return (
    <Section title="Identification attempts" subtitle="Every query submitted on the Identify page (unlabelled usage log)"
      actions={<select className="input w-56" value={status} onChange={(e) => { setStatus(e.target.value as Status | ""); setPage(1); }} aria-label="Filter by status">
        <option value="">All statuses</option><option value="MATCH_FOUND">Match found</option><option value="UNCERTAIN">Uncertain</option><option value="NO_RELIABLE_MATCH">No reliable match</option>
      </select>}>
      {error && <ErrorBox message={error} onRetry={reload} />}
      {loading && !data ? <Spinner /> : data && data.items.length === 0 ? <EmptyState title="No identification attempts" icon={<HistoryIcon className="h-8 w-8" />} /> : data && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-slate-500"><tr><th className="py-1">Query</th><th>When</th><th>Status</th><th>Selected object</th><th>Top candidates (score)</th><th>Time</th></tr></thead>
              <tbody>
                {data.items.map((h) => (
                  <tr key={h.id} className="border-t border-graphite-800">
                    <td className="py-2"><img src={url(h.query_image_url)} alt="" className="h-12 w-12 rounded object-cover" loading="lazy" /></td>
                    <td className="text-xs text-slate-400">{dateTime(h.created_at)}</td>
                    <td><StatusBadge status={h.status} /></td>
                    <td className="font-mono text-xs">{h.selected_object_public_id ? <Link className="text-accent" to={`/history?object=${h.selected_object_public_id}`}>{h.selected_object_public_id}</Link> : "—"}</td>
                    <td className="text-xs text-slate-400">{h.top_candidates.map((c) => `${c.name} (${c.score.toFixed(2)})`).join(", ") || "—"}</td>
                    <td className="font-mono text-xs">{h.inference_ms.toFixed(0)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {pages > 1 && (
            <div className="mt-4 flex items-center justify-center gap-3 text-sm">
              <button className="btn-secondary" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</button>
              <span className="text-slate-400">Page {page} / {pages}</span>
              <button className="btn-secondary" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>Next</button>
            </div>
          )}
        </>
      )}
    </Section>
  );
}
