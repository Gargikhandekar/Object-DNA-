import { AlertTriangle, CheckCircle2, CircleHelp, Inbox, Loader2, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import type { Status } from "../types";

export function Spinner({ label, className = "" }: { label?: string; className?: string }) {
  return (
    <span role="status" className={`inline-flex items-center gap-2 text-sm text-slate-400 ${className}`}>
      <Loader2 className="h-4 w-4 animate-spin text-accent" aria-hidden />
      {label ?? "Loading…"}
    </span>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-200">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="flex-1 break-words">{message}</div>
      {onRetry && (
        <button className="btn-ghost px-2 py-1 text-xs" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-graphite-600 px-6 py-12 text-center">
      <div className="mb-3 text-graphite-500">{icon ?? <Inbox className="h-8 w-8" aria-hidden />}</div>
      <p className="font-medium text-slate-300">{title}</p>
      {children && <div className="mt-2 max-w-md text-sm text-slate-500">{children}</div>}
    </div>
  );
}

const STATUS_STYLE: Record<Status, { cls: string; label: string; Icon: typeof CheckCircle2 }> = {
  MATCH_FOUND: { cls: "border-emerald-400/50 bg-emerald-400/10 text-emerald-300", label: "MATCH FOUND", Icon: CheckCircle2 },
  UNCERTAIN: { cls: "border-amber-400/50 bg-amber-400/10 text-amber-300", label: "UNCERTAIN", Icon: CircleHelp },
  NO_RELIABLE_MATCH: { cls: "border-rose-400/50 bg-rose-400/10 text-rose-300", label: "NO RELIABLE MATCH", Icon: XCircle },
};

export function StatusBadge({ status, large = false }: { status: Status; large?: boolean }) {
  const s = STATUS_STYLE[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-semibold tracking-wide ${s.cls} ${large ? "px-4 py-1.5 text-sm" : "px-2.5 py-0.5 text-[11px]"}`}>
      <s.Icon className={large ? "h-4 w-4" : "h-3 w-3"} aria-hidden />
      {s.label}
    </span>
  );
}

export function Stat({ label, value, hint, icon, accent = "text-accent" }: { label: string; value: ReactNode; hint?: string; icon?: ReactNode; accent?: string }) {
  return (
    <div className="card card-pad">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">{label}</span>
        {icon && <span className={accent}>{icon}</span>}
      </div>
      <div className="mt-2 font-mono text-2xl font-semibold text-slate-50">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

/** Horizontal bar for a [0,1] value. */
export function ScoreBar({ value, max = 1, color = "bg-accent", label }: { value: number | null | undefined; max?: number; color?: string; label?: string }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div className="flex items-center gap-2" aria-label={label}>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-graphite-700">
        <div className={`h-full rounded-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-12 text-right font-mono text-xs text-slate-300">{value == null ? "—" : value.toFixed(3)}</span>
    </div>
  );
}

export function Section({ title, subtitle, actions, children }: { title: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="card card-pad">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-slate-100">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="h-page">{title}</h1>
        {subtitle && <p className="sub mt-1 max-w-3xl">{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}

export function fmt(v: number | null | undefined, digits = 3): string {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;
}

export function dateTime(s: string): string {
  return new Date(s).toLocaleString();
}
