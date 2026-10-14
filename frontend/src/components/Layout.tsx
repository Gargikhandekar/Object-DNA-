import { Activity, Boxes, Dna, FlaskConical, GitCompareArrows, History, LayoutDashboard, ScanSearch, SquarePlus } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const NAV = [
  { to: "/", label: "Overview", Icon: LayoutDashboard },
  { to: "/register", label: "Register Object", Icon: SquarePlus },
  { to: "/identify", label: "Identify Object", Icon: ScanSearch },
  { to: "/compare", label: "Compare Images", Icon: GitCompareArrows },
  { to: "/library", label: "Object Library", Icon: Boxes },
  { to: "/history", label: "Object History", Icon: History },
  { to: "/research", label: "Research Lab", Icon: FlaskConical },
];

function ApiStatus() {
  const { data, error, loading } = useApi(api.health, [], { pollMs: 15000 });
  const ok = !!data && !error;
  return (
    <div className="space-y-2 rounded-lg border border-graphite-700 bg-graphite-850 p-3 text-xs" aria-live="polite">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${loading && !data ? "bg-slate-500" : ok ? "bg-emerald-400" : "bg-rose-500"}`} />
        <span className="text-slate-300">{loading && !data ? "Connecting to API…" : ok ? `API online · v${data.version}` : "API unreachable"}</span>
      </div>
      {data && (
        <div className="flex items-start gap-2">
          <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${data.embedding.available ? (data.embedding.loaded ? "bg-violet-400" : "bg-violet-400/50") : "bg-amber-400"}`} />
          <span className="text-slate-400">
            {data.embedding.available
              ? `Deep encoder: ${data.embedding.backbone}${data.embedding.loaded ? " (loaded)" : " (loads on first use)"}`
              : `Deep encoder unavailable — classical features only`}
          </span>
        </div>
      )}
      {data && <div className="text-slate-500">Fusion: {data.fusion_source}</div>}
    </div>
  );
}

export function Layout() {
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-graphite-800 bg-graphite-900/60 p-4 lg:flex">
        <div className="mb-8 flex items-center gap-2 px-2">
          <Dna className="h-7 w-7 text-accent" aria-hidden />
          <div>
            <div className="font-mono text-sm font-bold tracking-[0.25em] text-slate-50">OBJECT DNA</div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500">instance-level recognition</div>
          </div>
        </div>
        <nav className="flex-1 space-y-1" aria-label="Main">
          {NAV.map(({ to, label, Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${isActive ? "bg-accent/10 text-accent" : "text-slate-400 hover:bg-graphite-800 hover:text-slate-200"}`}>
              <Icon className="h-4 w-4" aria-hidden /> {label}
            </NavLink>
          ))}
        </nav>
        <ApiStatus />
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 overflow-x-auto border-b border-graphite-800 bg-graphite-900/60 px-3 py-2 lg:hidden">
          <Activity className="h-5 w-5 shrink-0 text-accent" />
          {NAV.map(({ to, label }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `whitespace-nowrap rounded px-2 py-1 text-xs ${isActive ? "bg-accent/10 text-accent" : "text-slate-400"}`}>{label}</NavLink>
          ))}
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 p-4 sm:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
