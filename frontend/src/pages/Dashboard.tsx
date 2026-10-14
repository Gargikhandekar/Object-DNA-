import { Activity, Boxes, CheckCircle2, CircleHelp, Clock, Cpu, FlaskConical, Images, ScanSearch, XCircle } from "lucide-react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EmptyState, ErrorBox, PageHeader, Section, Spinner, Stat, StatusBadge, dateTime, fmt, pct } from "../components/ui";
import { useToast } from "../components/Toast";
import { useApi } from "../hooks/useApi";
import { api, url } from "../services/api";

const tooltipStyle = { background: "#161a1f", border: "1px solid #384250", borderRadius: 8, fontSize: 12 };

export default function Dashboard() {
  const { data, error, loading, reload } = useApi(api.dashboard, [], { pollMs: 20000 });
  const toast = useToast();

  if (loading && !data) return <Spinner label="Loading dashboard…" />;
  if (error && !data) return <ErrorBox message={error} onRetry={reload} />;
  if (!data) return null;
  const u = data.usage;
  const outcome = [
    { name: "Match found", value: u.accepted, color: "#34d399" },
    { name: "Uncertain", value: u.uncertain, color: "#fbbf24" },
    { name: "No reliable match", value: u.rejected, color: "#fb7185" },
  ];
  const latest = data.experiments[0];

  const backfill = async () => {
    try {
      const r = await api.backfillEmbeddings();
      toast("success", `Computed ${r.computed} missing embeddings.`);
    } catch (e) {
      toast("error", (e as Error).message);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Overview" subtitle="Application activity and model readiness. Usage counts are not accuracy — performance is measured only on labelled data in the Research Lab." />

      <div>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">Application usage (unlabelled)</h2>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Stat label="Registered objects" value={u.registered_objects} icon={<Boxes className="h-4 w-4" />} />
          <Stat label="Reference images" value={u.reference_images} icon={<Images className="h-4 w-4" />} />
          <Stat label="Query attempts" value={u.query_attempts} icon={<ScanSearch className="h-4 w-4" />} />
          <Stat label="Avg. inference" value={u.avg_inference_ms == null ? "—" : `${u.avg_inference_ms.toFixed(0)} ms`} icon={<Clock className="h-4 w-4" />} hint="feature extraction + matching" />
          <Stat label="Accepted" value={u.accepted} icon={<CheckCircle2 className="h-4 w-4" />} accent="text-emerald-400" />
          <Stat label="Uncertain" value={u.uncertain} icon={<CircleHelp className="h-4 w-4" />} accent="text-amber-400" />
          <Stat label="No reliable match" value={u.rejected} icon={<XCircle className="h-4 w-4" />} accent="text-rose-400" />
          <Stat label="Observations" value={u.observations} icon={<Activity className="h-4 w-4" />} hint="accepted into object histories" />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Section title="Recent identification attempts" subtitle="Latest queries submitted on the Identify page" actions={<Link to="/history" className="btn-ghost text-xs">View all</Link>}>
          {data.recent.length === 0 ? (
            <EmptyState title="No identification attempts yet"><Link className="text-accent" to="/identify">Identify an object</Link> to see activity here.</EmptyState>
          ) : (
            <ul className="divide-y divide-graphite-800">
              {data.recent.map((r) => (
                <li key={r.id} className="flex items-center gap-3 py-2">
                  <img src={url(r.query_image_url)} alt="" className="h-10 w-10 rounded object-cover" loading="lazy" />
                  <div className="min-w-0 flex-1">
                    <StatusBadge status={r.status} />
                    <div className="mt-0.5 truncate text-xs text-slate-500">{r.top_candidates[0] ? `${r.top_candidates[0].name} · ${fmt(r.best_score)}` : "no candidates"} · {dateTime(r.created_at)}</div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Section>

        <Section title="Decision outcomes" subtitle="Distribution of statuses across all queries">
          {u.query_attempts === 0 ? <EmptyState title="No data yet" /> : (
            <div className="h-56">
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={outcome} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={2}>
                    {outcome.map((o) => <Cell key={o.name} fill={o.color} stroke="none" />)}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-4 text-xs">{outcome.map((o) => <span key={o.name} className="flex items-center gap-1"><span className="h-2 w-2 rounded-full" style={{ background: o.color }} />{o.name}</span>)}</div>
            </div>
          )}
        </Section>

        <Section title="Model status" subtitle="Feature extractors and decision configuration">
          <dl className="space-y-3 text-sm">
            <div className="flex items-start gap-2">
              <Cpu className={`mt-0.5 h-4 w-4 ${data.model_status.available ? "text-violet-400" : "text-amber-400"}`} />
              <div>
                <dt className="text-slate-300">Deep embedding: {data.model_status.backbone}</dt>
                <dd className="text-xs text-slate-500">{data.model_status.available ? `${data.model_status.description ?? ""} ${data.model_status.loaded ? "· loaded" : "· lazy-loaded on first use"}` : data.model_status.message}</dd>
                {data.model_status.available && <button className="btn-ghost mt-1 px-2 py-1 text-xs" onClick={backfill}>Backfill missing embeddings</button>}
              </div>
            </div>
            <div className="flex items-start gap-2">
              <Cpu className="mt-0.5 h-4 w-4 text-accent" />
              <div><dt className="text-slate-300">Classical: ORB · LBP · HSV · Hu moments</dt><dd className="text-xs text-slate-500">always available (OpenCV)</dd></div>
            </div>
            <div className="rounded-lg bg-graphite-850 p-3 text-xs">
              <div className="mb-1 text-slate-400">Fusion: <span className="text-slate-200">{data.fusion.source}</span></div>
              <div className="font-mono text-slate-400">accept ≥ {data.fusion.accept_threshold.toFixed(3)} · uncertain ≥ {data.fusion.uncertain_threshold.toFixed(3)} · margin ≥ {data.fusion.margin_threshold.toFixed(3)}</div>
              <div className="mt-1 font-mono text-slate-500">{Object.entries(data.fusion.weights).filter(([, w]) => w > 0).map(([k, w]) => `${k} ${w.toFixed(2)}`).join(" · ")}</div>
            </div>
          </dl>
        </Section>
      </div>

      <Section title="Research experiment summaries" subtitle="Measured on labelled evaluation data (test split). Synthetic-data runs are sanity checks only." actions={<Link to="/research" className="btn-ghost text-xs"><FlaskConical className="h-4 w-4" />Research Lab</Link>}>
        {!latest ? (
          <EmptyState title="No completed experiments">Run an evaluation in the <Link to="/research" className="text-accent">Research Lab</Link>. Metrics appear here only after a real run.</EmptyState>
        ) : (
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <div className="mb-2 text-xs text-slate-400">Latest run <span className="font-mono text-slate-300">{latest.run_uid}</span> · {latest.dataset_id} {latest.is_synthetic && <span className="ml-1 rounded bg-amber-400/10 px-1.5 py-0.5 text-amber-300">synthetic</span>}</div>
              <div className="h-56">
                <ResponsiveContainer>
                  <BarChart data={Object.entries(latest.identification_top1).map(([m, v]) => ({ method: m, top1: v ?? 0 }))} layout="vertical" margin={{ left: 30 }}>
                    <CartesianGrid stroke="#272e37" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} stroke="#5b6675" fontSize={11} />
                    <YAxis type="category" dataKey="method" stroke="#5b6675" fontSize={11} width={110} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => pct(v)} />
                    <Bar dataKey="top1" name="Top-1 identification accuracy" fill="#22d3ee" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <table className="w-full self-start text-sm">
              <thead className="text-left text-xs text-slate-500"><tr><th className="py-1">Run</th><th>Dataset</th><th>Best top-1</th><th>Best ver. AUC</th></tr></thead>
              <tbody>
                {data.experiments.map((e) => {
                  const t1 = Math.max(...Object.values(e.identification_top1).map((v) => v ?? 0));
                  const auc = Math.max(...Object.values(e.verification_auc).map((v) => v ?? 0));
                  return (
                    <tr key={e.run_uid} className="border-t border-graphite-800">
                      <td className="py-1.5 font-mono text-xs">{e.run_uid}</td>
                      <td className="text-xs text-slate-400">{e.dataset_id.split("@")[0]}{e.is_synthetic ? " (synthetic)" : ""}</td>
                      <td className="font-mono text-xs">{Object.keys(e.identification_top1).length ? pct(t1) : "—"}</td>
                      <td className="font-mono text-xs">{Object.keys(e.verification_auc).length ? fmt(auc) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>
      {data.categories.length > 0 && (
        <p className="text-xs text-slate-500">Categories: {data.categories.map((c) => `${c.category} (${c.count})`).join(" · ")}</p>
      )}
    </div>
  );
}
