import { Boxes, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { EmptyState, ErrorBox, PageHeader, Spinner } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, url } from "../services/api";

const PAGE = 24;

export default function Library() {
  const [q, setQ] = useState("");
  const [debounced, setDebounced] = useState("");
  const [category, setCategory] = useState("");
  const [page, setPage] = useState(1);
  useEffect(() => {
    const t = setTimeout(() => { setDebounced(q); setPage(1); }, 300);
    return () => clearTimeout(t);
  }, [q]);
  const { data, error, loading, reload } = useApi(() => api.listObjects({ q: debounced, category, page, page_size: PAGE }), [debounced, category, page]);
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1;

  return (
    <div className="space-y-6">
      <PageHeader title="Object Library" subtitle="Every card is one registered physical instance." actions={<Link to="/register" className="btn-primary">Register object</Link>} />
      <div className="flex flex-wrap gap-3">
        <div className="relative min-w-[240px] flex-1">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-500" aria-hidden />
          <input className="input pl-9" placeholder="Search by name, object ID, category or description" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search objects" />
        </div>
        <select className="input w-48" value={category} onChange={(e) => { setCategory(e.target.value); setPage(1); }} aria-label="Filter by category">
          <option value="">All categories</option>
          {data?.categories.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      {error && <ErrorBox message={error} onRetry={reload} />}
      {loading && !data ? <Spinner /> : data && data.items.length === 0 ? (
        <EmptyState title={debounced || category ? "No objects match your filters" : "No objects registered yet"} icon={<Boxes className="h-8 w-8" />}>
          {!debounced && !category && <Link to="/register" className="text-accent">Register your first physical object</Link>}
        </EmptyState>
      ) : data && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
            {data.items.map((o) => (
              <Link key={o.public_id} to={`/library/${o.public_id}`} className="card group overflow-hidden transition hover:border-accent/60">
                <div className="aspect-square overflow-hidden bg-graphite-850">
                  {o.thumbnail_url ? <img src={url(o.thumbnail_url)} alt={o.name} loading="lazy" className="h-full w-full object-cover transition group-hover:scale-105" /> : <div className="flex h-full items-center justify-center text-graphite-500"><Boxes /></div>}
                </div>
                <div className="space-y-1 p-3">
                  <div className="truncate text-sm font-medium text-slate-100">{o.name}</div>
                  <div className="font-mono text-[11px] text-accent">{o.public_id}</div>
                  <div className="flex flex-wrap items-center gap-1 text-[11px] text-slate-500">
                    <span className="rounded bg-graphite-800 px-1.5 py-0.5 text-slate-300">{o.category}</span>
                    <span>{o.image_count} img</span>
                    <span className={o.fingerprint_status === "complete" ? "text-emerald-400" : "text-amber-400"}>● {o.fingerprint_status}</span>
                  </div>
                  <div className="text-[10px] text-slate-600">Registered {new Date(o.created_at).toLocaleDateString()}</div>
                </div>
              </Link>
            ))}
          </div>
          {pages > 1 && (
            <div className="flex items-center justify-center gap-3 text-sm">
              <button className="btn-secondary" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</button>
              <span className="text-slate-400">Page {page} of {pages} · {data.total} objects</span>
              <button className="btn-secondary" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>Next</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
