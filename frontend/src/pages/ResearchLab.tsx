import { AlertTriangle, Download, FlaskConical, Play, Sparkles, Trash2, Wand2 } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { useToast } from "../components/Toast";
import { EmptyState, ErrorBox, PageHeader, Section, Spinner, dateTime, fmt, pct } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";
import type { ExperimentMetrics, ExperimentRun } from "../types";

const COLORS: Record<string, string> = { orb: "#22d3ee", texture: "#34d399", color: "#fbbf24", deep: "#a78bfa", hybrid_default: "#94a3b8", hybrid_tuned: "#f472b6" };
const tooltipStyle = { background: "#161a1f", border: "1px solid #384250", borderRadius: 8, fontSize: 12 };

export default function ResearchLab() {
  const toast = useToast();
  const datasets = useApi(api.datasets, []);
  const runs = useApi(api.experiments, []);
  const [dataset, setDataset] = useState("");
  const [experiment, setExperiment] = useState("full");
  const [seed, setSeed] = useState(42);
  const [valFraction, setValFraction] = useState(0.4);
  const [perturbations, setPerturbations] = useState(true);
  const [starting, setStarting] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [selected, setSelected] = useState<string>();

  useEffect(() => {
    if (!dataset && datasets.data?.length) setDataset(datasets.data[0].name);
  }, [datasets.data, dataset]);
  const active = runs.data?.some((r) => r.status === "queued" || r.status === "running");
  useEffect(() => {
    if (!active) return;
    const id = setInterval(runs.reload, 2500);
    return () => clearInterval(id);
  }, [active, runs.reload]);
  useEffect(() => {
    if (!selected && runs.data?.length) setSelected(runs.data.find((r) => r.status === "completed")?.run_uid ?? runs.data[0].run_uid);
  }, [runs.data, selected]);

  const start = async () => {
    setStarting(true);
    try {
      const r = await api.runExperiment({ experiment, dataset, seed, val_fraction: valFraction, perturbations });
      toast("success", `Experiment ${r.run_uid} queued.`);
      setSelected(r.run_uid);
      runs.reload();
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setStarting(false);
    }
  };
  const generate = async () => {
    setGenerating(true);
    try {
      const d = await api.generateSynthetic();
      toast("success", `Generated ${d.images} synthetic images of ${d.objects} instances.`);
      datasets.reload();
      setDataset(d.name);
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Research Lab" subtitle="Classical vs. deep vs. hybrid Object DNA on labelled data. Weights and thresholds are fitted on the validation split only; all reported numbers come from held-out test identities." />
      <div className="grid gap-6 lg:grid-cols-3">
        <Section title="Run an evaluation" subtitle="Runs in a background worker; one at a time.">
          {datasets.error && <ErrorBox message={datasets.error} onRetry={datasets.reload} />}
          <div className="space-y-3">
            <div>
              <label className="label" htmlFor="ds">Dataset</label>
              {datasets.data && datasets.data.length === 0 ? (
                <p className="text-xs text-slate-400">No datasets found in <span className="font-mono">data/datasets/</span>. Import your own with <span className="font-mono">scripts/import_dataset.py</span> or generate the synthetic demo.</p>
              ) : (
                <select id="ds" className="input" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                  {datasets.data?.map((d) => <option key={d.name} value={d.name}>{d.name} — {d.objects} objects, {d.images} images{d.synthetic ? " (synthetic)" : ""}</option>)}
                </select>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label" htmlFor="exp">Experiment</label>
                <select id="exp" className="input" value={experiment} onChange={(e) => setExperiment(e.target.value)}>
                  <option value="full">Full suite</option><option value="verification">Verification</option><option value="identification">Identification</option><option value="robustness">Robustness</option>
                </select>
              </div>
              <div>
                <label className="label" htmlFor="seed">Seed</label>
                <input id="seed" type="number" className="input" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              </div>
              <div>
                <label className="label" htmlFor="vf">Validation fraction</label>
                <input id="vf" type="number" min={0.1} max={0.8} step={0.05} className="input" value={valFraction} onChange={(e) => setValFraction(Number(e.target.value))} />
              </div>
              <label className="mt-6 flex items-center gap-2 text-xs text-slate-300"><input type="checkbox" checked={perturbations} onChange={(e) => setPerturbations(e.target.checked)} className="accent-cyan-400" /> Synthetic perturbations</label>
            </div>
            <div className="flex flex-wrap gap-2">
              <button className="btn-primary" disabled={!dataset || starting} onClick={start}><Play className="h-4 w-4" />{starting ? "Queuing…" : "Run"}</button>
              <button className="btn-secondary" disabled={generating} onClick={generate} title="Procedural dataset for pipeline sanity checks — not real-world evidence"><Sparkles className="h-4 w-4" />{generating ? "Generating…" : "Generate synthetic demo"}</button>
            </div>
          </div>
        </Section>
        <div className="lg:col-span-2">
          <Section title="Experiment runs" subtitle="Select a run to inspect its results">
            {runs.error && <ErrorBox message={runs.error} onRetry={runs.reload} />}
            {runs.loading && !runs.data ? <Spinner /> : runs.data?.length === 0 ? <EmptyState title="No experiment runs yet" icon={<FlaskConical className="h-8 w-8" />}>Choose a dataset and press Run. No metrics are shown until a run completes.</EmptyState> : (
              <div className="max-h-72 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-graphite-900 text-left text-slate-500"><tr><th className="py-1">Run</th><th>Type</th><th>Dataset</th><th>Status</th><th>Created</th></tr></thead>
                  <tbody>
                    {runs.data?.map((r) => (
                      <tr key={r.run_uid} onClick={() => setSelected(r.run_uid)} className={`cursor-pointer border-t border-graphite-800 ${selected === r.run_uid ? "bg-accent/5" : "hover:bg-graphite-850"}`}>
                        <td className="py-1.5 font-mono text-slate-200">{r.run_uid}</td>
                        <td>{r.experiment}</td>
                        <td className="text-slate-400">{r.dataset_id.split("@")[0]}{r.is_synthetic && <span className="ml-1 text-amber-300">(synthetic)</span>}</td>
                        <td>{r.status === "running" || r.status === "queued" ? (
                          <div className="w-32"><div className="h-1.5 overflow-hidden rounded bg-graphite-700"><div className="h-full bg-accent transition-all" style={{ width: `${r.progress * 100}%` }} /></div><div className="mt-0.5 truncate text-[10px] text-slate-500">{r.message}</div></div>
                        ) : <span className={r.status === "completed" ? "text-emerald-400" : "text-rose-400"} title={r.message ?? ""}>{r.status}</span>}</td>
                        <td className="text-slate-500">{dateTime(r.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>
        </div>
      </div>
      {selected && <RunDetail uid={selected} onDeleted={() => { setSelected(undefined); runs.reload(); }} />}
    </div>
  );
}

function RunDetail({ uid, onDeleted }: { uid: string; onDeleted: () => void }) {
  const toast = useToast();
  const { data, error, loading, reload } = useApi(() => api.experiment(uid), [uid]);
  const [confirm, setConfirm] = useState(false);
  const running = data && (data.status === "running" || data.status === "queued");
  useEffect(() => {
    if (!running) return;
    const id = setInterval(reload, 3000);
    return () => clearInterval(id);
  }, [running, reload]);

  if (loading && !data) return <Spinner />;
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  if (!data) return null;
  if (running) return <div className="card card-pad"><Spinner label={`${data.message ?? "Running"} (${(data.progress * 100).toFixed(0)}%)`} /></div>;
  if (data.status === "failed") return <ErrorBox message={`Run ${uid} failed: ${data.message}`} />;
  const m = data.metrics;

  const apply = async () => {
    try {
      const f = await api.applyFusion(uid);
      toast("success", `Application now uses ${f.source} (accept ≥ ${f.accept_threshold}).`);
    } catch (e) {
      toast("error", (e as Error).message);
    }
  };
  const del = async () => {
    try {
      await api.deleteExperiment(uid);
      toast("success", "Run deleted.");
      onDeleted();
    } catch (e) {
      toast("error", (e as Error).message);
    }
    setConfirm(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-slate-100">Run <span className="font-mono text-accent">{uid}</span> <span className="text-sm font-normal text-slate-500">· {m.dataset_id} · {m.runtime_s}s</span></h2>
        <div className="flex flex-wrap gap-2">
          <a className="btn-secondary text-xs" href={api.exportCsvUrl(uid)}><Download className="h-3.5 w-3.5" /> Summary CSV</a>
          {Object.values(data.artifact_paths).map((f) => <a key={f} className="btn-ghost text-xs" href={api.artifactUrl(uid, f)} target="_blank" rel="noreferrer"><Download className="h-3.5 w-3.5" />{f}</a>)}
          <button className="btn-ghost text-xs text-rose-300" onClick={() => setConfirm(true)}><Trash2 className="h-3.5 w-3.5" /> Delete</button>
        </div>
      </div>
      {m.is_synthetic && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-200">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> Synthetic, procedurally generated dataset. These numbers only verify that the pipeline behaves sensibly; they are not evidence of real-world performance. Collect a real dataset (see README) for research conclusions.
        </div>
      )}
      {m.methods_unavailable && Object.keys(m.methods_unavailable).length > 0 && (
        <div className="text-xs text-slate-400">Not evaluated: {Object.entries(m.methods_unavailable).map(([k, v]) => <span key={k} className="mr-3"><span className="font-mono text-slate-300">{k}</span> — {v}</span>)}</div>
      )}
      <MethodTable m={m} />
      <div className="grid gap-6 xl:grid-cols-2">
        <RocChart m={m} />
        <ComparisonChart m={m} />
      </div>
      <Robustness m={m} />
      <div className="grid gap-6 xl:grid-cols-2">
        <Confusion m={m} />
        <div className="space-y-6">
          <Section title="Hybrid fusion fitted on validation" subtitle="Normalisation: lo = 5th pct of different-instance pairs, hi = 95th pct of same-instance pairs; weights: simplex grid search maximising validation ROC-AUC."
            actions={m.hybrid_tuning?.weights && <button className="btn-secondary text-xs" onClick={apply}><Wand2 className="h-3.5 w-3.5" /> Use in application</button>}>
            {m.hybrid_tuning?.weights ? (
              <table className="w-full text-xs">
                <thead className="text-left text-slate-500"><tr><th className="py-1">Component</th><th>Weight</th><th>lo</th><th>hi</th></tr></thead>
                <tbody className="font-mono">
                  {Object.entries(m.hybrid_tuning.weights).map(([c, w]) => (
                    <tr key={c} className="border-t border-graphite-800"><td className="py-1 font-sans">{c}</td><td>{w.toFixed(2)}</td><td>{fmt(m.hybrid_tuning?.ranges?.[c]?.lo)}</td><td>{fmt(m.hybrid_tuning?.ranges?.[c]?.hi)}</td></tr>
                  ))}
                </tbody>
              </table>
            ) : <p className="text-xs text-slate-500">{m.hybrid_tuning?.message ?? "Not available."}</p>}
            {m.hybrid_tuning?.validation_roc_auc != null && <p className="mt-2 text-xs text-slate-500">Validation ROC-AUC of the fitted fusion: <span className="font-mono">{fmt(m.hybrid_tuning.validation_roc_auc)}</span></p>}
          </Section>
          <SplitInfo m={m} />
        </div>
      </div>
      <ConfirmDialog open={confirm} title="Delete experiment run?" message={`Run ${uid} and its artifacts will be permanently deleted.`} onCancel={() => setConfirm(false)} onConfirm={del} />
    </div>
  );
}

const label = (m: ExperimentMetrics, k: string) => m.method_labels?.[k] ?? k;

function MethodTable({ m }: { m: ExperimentMetrics }) {
  const methods = m.methods_evaluated ?? [];
  const id = m.identification?.methods ?? {};
  const ver = m.verification?.methods ?? {};
  return (
    <Section title="Method comparison (test split)" subtitle={`Identification: ${m.identification?.gallery_images ?? "—"} gallery images, ${m.identification?.known_objects ?? "—"} enrolled objects, ${m.identification?.unknown_objects ?? 0} unregistered. Thresholds chosen on validation.`}>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-slate-500">
            <tr>
              <th className="py-1">Method</th>
              <th title="Closed-set rank-1 accuracy on enrolled queries">Top-1</th><th>Top-5</th>
              <th title="Correct accepted / all accepted">Acc. precision</th><th title="Correct accepted / enrolled queries">Acc. recall</th>
              <th>Unknown rejection</th><th>Open-set acc.</th>
              <th className="border-l border-graphite-700 pl-2">Ver. AUC</th><th>EER</th><th>Precision</th><th>Recall</th><th>F1</th><th>FMR</th><th>FNMR</th>
              <th className="border-l border-graphite-700 pl-2" title="Mean time to match one query against the gallery">Match ms</th><th>Extract ms</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {methods.map((k) => {
              const i = id[k];
              const v = ver[k]?.test;
              return (
                <tr key={k} className="border-t border-graphite-800">
                  <td className="py-1.5 font-sans text-slate-200"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: COLORS[k] }} />{label(m, k)}</td>
                  <td>{pct(i?.top1_accuracy)}</td><td>{pct(i?.top5_accuracy)}</td><td>{pct(i?.accepted_precision)}</td><td>{pct(i?.accepted_recall)}</td>
                  <td>{pct(i?.unknown_rejection_rate)}</td><td>{pct(i?.open_set_accuracy)}</td>
                  <td className="border-l border-graphite-700 pl-2">{fmt(v?.roc_auc)}</td><td>{fmt(v?.eer)}</td><td>{fmt(v?.precision)}</td><td>{fmt(v?.recall)}</td><td>{fmt(v?.f1)}</td><td>{fmt(v?.fmr)}</td><td>{fmt(v?.fnmr)}</td>
                  <td className="border-l border-graphite-700 pl-2">{fmt(i?.matching_ms, 1)}</td><td>{fmt(i?.extraction_ms, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {m.verification && <p className="mt-2 text-[11px] text-slate-500">Verification pairs — test: {String(m.verification.test_pairs.positives)} same-instance / {String(m.verification.test_pairs.negatives)} different-instance ({String(m.verification.test_pairs.hard_negatives_same_model)} same-model hard negatives). ORB scores are inlier counts, so their thresholds are counts, not [0,1] values. “—” = not available for this run.</p>}
    </Section>
  );
}

function RocChart({ m }: { m: ExperimentMetrics }) {
  const ver = m.verification?.methods ?? {};
  const series = Object.entries(ver).filter(([, v]) => v.available && v.test?.curve?.length);
  return (
    <Section title="ROC — same-instance verification (test)" subtitle="True match rate vs false match rate">
      {series.length === 0 ? <EmptyState title="Verification not run for this experiment" /> : (
        <div className="h-72">
          <ResponsiveContainer>
            <LineChart margin={{ bottom: 10 }}>
              <CartesianGrid stroke="#272e37" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} stroke="#5b6675" fontSize={11} label={{ value: "FMR", position: "insideBottom", offset: -4, fill: "#5b6675", fontSize: 11 }} />
              <YAxis type="number" domain={[0, 1]} stroke="#5b6675" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {series.map(([k, v]) => <Line key={k} data={v.test!.curve} dataKey="tpr" name={`${label(m, k)} (${fmt(v.test!.roc_auc)})`} stroke={COLORS[k]} dot={false} strokeWidth={2} type="linear" />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Section>
  );
}

function ComparisonChart({ m }: { m: ExperimentMetrics }) {
  const rows = (m.methods_evaluated ?? []).map((k) => ({
    method: k,
    "Top-1": m.identification?.methods[k]?.top1_accuracy ?? null,
    "Open-set acc.": m.identification?.methods[k]?.open_set_accuracy ?? null,
    "Ver. AUC": m.verification?.methods[k]?.test?.roc_auc ?? null,
  }));
  return (
    <Section title="Classical vs deep vs hybrid" subtitle="Test-split metrics per method">
      <div className="h-72">
        <ResponsiveContainer>
          <BarChart data={rows}>
            <CartesianGrid stroke="#272e37" vertical={false} />
            <XAxis dataKey="method" stroke="#5b6675" fontSize={10} />
            <YAxis domain={[0, 1]} stroke="#5b6675" fontSize={11} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => (v == null ? "n/a" : Number(v).toFixed(3))} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="Top-1" fill="#22d3ee" radius={[3, 3, 0, 0]} />
            <Bar dataKey="Open-set acc." fill="#a78bfa" radius={[3, 3, 0, 0]} />
            <Bar dataKey="Ver. AUC" fill="#34d399" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Section>
  );
}

function Robustness({ m }: { m: ExperimentMetrics }) {
  const conds = m.robustness?.conditions ?? [];
  const methods = m.methods_evaluated ?? [];
  const [metric, setMetric] = useState<"top1_accuracy" | "false_match_rate" | "open_set_accuracy">("top1_accuracy");
  const chart = useMemo(() => conds.filter((c) => c.status === "evaluated" && c.condition !== "unregistered").map((c) => ({
    condition: c.condition, ...Object.fromEntries(methods.map((k) => [k, c.methods[k]?.[metric] ?? null])),
  })), [conds, methods, metric]);
  if (!m.robustness) return <Section title="Robustness by condition"><EmptyState title="Robustness not evaluated in this run" /></Section>;
  return (
    <Section title="Robustness by acquisition condition (test split)" subtitle={m.robustness.source + ". Conditions absent from the dataset are marked pending; synthetic perturbations are labelled as such."}
      actions={<select className="input w-52 text-xs" value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)} aria-label="Robustness metric">
        <option value="top1_accuracy">Top-1 accuracy</option><option value="open_set_accuracy">Open-set accuracy</option><option value="false_match_rate">False match rate</option>
      </select>}>
      {chart.length > 0 && (
        <div className="mb-4 h-72">
          <ResponsiveContainer>
            <BarChart data={chart}>
              <CartesianGrid stroke="#272e37" vertical={false} />
              <XAxis dataKey="condition" stroke="#5b6675" fontSize={10} interval={0} angle={-20} textAnchor="end" height={60} />
              <YAxis domain={[0, 1]} stroke="#5b6675" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => (v == null ? "n/a" : Number(v).toFixed(3))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {methods.map((k) => <Bar key={k} dataKey={k} name={label(m, k)} fill={COLORS[k]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-slate-500"><tr><th className="py-1">Condition</th><th>Source</th><th>N</th>{methods.map((k) => <th key={k}>{k}<div className="font-normal text-[10px]">top-1 / FMR / ms</div></th>)}</tr></thead>
          <tbody>
            {conds.map((c) => (
              <tr key={c.condition} className="border-t border-graphite-800">
                <td className="py-1.5 text-slate-200">{c.label}</td>
                <td className="text-slate-500">{c.source}</td>
                <td className="font-mono">{c.n_queries}</td>
                {c.status === "pending" ? <td colSpan={methods.length} className="text-amber-300/80">pending — no {c.condition === "unregistered" ? "unregistered objects" : "images for this condition"} in the test split</td> :
                  methods.map((k) => {
                    const v = c.methods[k];
                    return <td key={k} className="font-mono text-slate-300">{v ? (c.condition === "unregistered" ? `rej ${pct(v.unknown_rejection_rate, 0)}` : `${pct(v.top1_accuracy, 0)} / ${pct(v.false_match_rate, 0)}`) : "—"}<span className="text-slate-600"> {v ? v.mean_matching_ms.toFixed(0) : ""}</span></td>;
                  })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-slate-500">FMR here = fraction of queries accepted as the wrong identity (or, for unregistered objects, accepted at all). Observations on small or synthetic test sets carry wide uncertainty.</p>
    </Section>
  );
}

function Confusion({ m }: { m: ExperimentMetrics }) {
  const methods = (m.methods_evaluated ?? []).filter((k) => m.identification?.methods[k]?.confusion);
  const [k, setK] = useState<string>("");
  const method = k || (methods.includes("hybrid_tuned") ? "hybrid_tuned" : methods[0]);
  const cm = method ? m.identification?.methods[method]?.confusion : undefined;
  if (!cm) return <Section title="Confusion matrix"><EmptyState title="Identification not run for this experiment" /></Section>;
  const max = Math.max(1, ...cm.matrix.flat());
  return (
    <Section title="Open-set confusion matrix (test)" subtitle="Rows: true identity (+ unknown). Columns: predicted identity (+ rejected)."
      actions={<select className="input w-48 text-xs" value={method} onChange={(e) => setK(e.target.value)} aria-label="Method">{methods.map((x) => <option key={x} value={x}>{label(m, x)}</option>)}</select>}>
      <div className="overflow-auto">
        <table className="text-[9px]">
          <thead><tr><th />{cm.cols.map((c) => <th key={c} className="h-16 align-bottom font-mono font-normal text-slate-500"><div className="w-4 -rotate-90 whitespace-nowrap">{c.replace("object_", "")}</div></th>)}</tr></thead>
          <tbody>
            {cm.rows.map((r, i) => (
              <tr key={r}>
                <th className="pr-1 text-right font-mono font-normal text-slate-500">{r.replace("object_", "")}</th>
                {cm.matrix[i].map((v, j) => {
                  const diag = cm.cols[j] === r || (r === "unknown" && cm.cols[j] === "rejected");
                  return <td key={j} title={`${r} → ${cm.cols[j]}: ${v}`} className="h-4 w-4 text-center font-mono" style={{ background: v ? (diag ? `rgba(34,211,238,${0.15 + 0.85 * v / max})` : `rgba(251,113,133,${0.2 + 0.8 * v / max})`) : "#161a1f", color: v ? "#0b0d10" : "transparent" }}>{v || ""}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function SplitInfo({ m }: { m: ExperimentMetrics }) {
  const s = m.split;
  if (!s) return null;
  return (
    <Section title="Dataset split" subtitle={s.strategy}>
      <dl className="grid grid-cols-2 gap-y-1 text-xs">
        <dt className="text-slate-500">Split ID</dt><dd className="font-mono">{s.split_id}</dd>
        <dt className="text-slate-500">Validation identities</dt><dd>{s.val_objects.length}</dd>
        <dt className="text-slate-500">Test identities</dt><dd>{s.test_objects.length}</dd>
        <dt className="text-slate-500">Unregistered identities</dt><dd>{s.unknown_objects.length}</dd>
        <dt className="text-slate-500">Gallery</dt><dd>{s.gallery.images} images · sessions {s.gallery.sessions.join(", ")}</dd>
        <dt className="text-slate-500">Validation queries</dt><dd>{s.val_queries.images} · sessions {s.val_queries.sessions.join(", ")}</dd>
        <dt className="text-slate-500">Test queries</dt><dd>{s.test_queries.images} · sessions {s.test_queries.sessions.join(", ")}</dd>
        {Object.entries(s.excluded).map(([k, v]) => <Fragment key={k}><dt className="text-slate-500">Excluded: {k.replace(/_/g, " ")}</dt><dd>{v}</dd></Fragment>)}
      </dl>
      {s.notes.length > 0 && <ul className="mt-2 text-xs text-amber-300">{s.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
    </Section>
  );
}

export type { ExperimentRun };
