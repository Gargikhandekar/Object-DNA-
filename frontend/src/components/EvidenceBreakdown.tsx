import type { Evidence, Fused } from "../types";
import { fmt } from "./ui";

export const COMPONENT_INFO: Record<string, { label: string; raw: string; color: string }> = {
  deep: { label: "Deep embedding", raw: "cosine similarity", color: "#a78bfa" },
  geometric: { label: "ORB geometric", raw: "RANSAC inliers", color: "#22d3ee" },
  texture: { label: "LBP texture", raw: "histogram intersection", color: "#34d399" },
  color: { label: "HSV colour", raw: "1 − Bhattacharyya", color: "#fbbf24" },
  shape: { label: "Contour shape", raw: "Hu-moment similarity", color: "#f472b6" },
};

/** Additive breakdown of the fused score: contribution_i = weight_i × normalised_i. */
export function EvidenceBreakdown({ fused, evidence, compact = false }: { fused: Fused; evidence: Evidence; compact?: boolean }) {
  const comps = Object.keys(COMPONENT_INFO);
  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 flex justify-between text-xs text-slate-400">
          <span>Fused similarity score (sum of contributions)</span>
          <span className="font-mono text-slate-200">{fmt(fused.score)}</span>
        </div>
        <div className="flex h-3 overflow-hidden rounded-full bg-graphite-700" role="img" aria-label="Contribution of each component to the fused score">
          {comps.map((c) => fused.contributions[c] ? (
            <div key={c} title={`${COMPONENT_INFO[c].label}: ${fused.contributions[c].toFixed(3)}`} style={{ width: `${fused.contributions[c] * 100}%`, background: COMPONENT_INFO[c].color }} />
          ) : null)}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-slate-500">
            <tr>
              <th className="py-1 pr-2 font-medium">Component</th>
              {!compact && <th className="py-1 pr-2 font-medium">Raw value</th>}
              <th className="py-1 pr-2 font-medium">Normalised</th>
              <th className="py-1 pr-2 font-medium">Weight</th>
              <th className="py-1 font-medium">Contribution</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {comps.map((c) => {
              const raw = evidence.raw[c];
              const missing = fused.missing.includes(c) || raw == null;
              const w = fused.used_weights[c];
              if (missing && !fused.missing.includes(c) && !w) return null;
              return (
                <tr key={c} className="border-t border-graphite-800">
                  <td className="py-1.5 pr-2 font-sans text-slate-300">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: COMPONENT_INFO[c].color }} />
                    {COMPONENT_INFO[c].label}
                  </td>
                  {!compact && <td className="py-1.5 pr-2 text-slate-400">{raw == null ? "—" : c === "geometric" ? raw.toFixed(0) : raw.toFixed(3)} <span className="font-sans text-[10px] text-slate-600">{COMPONENT_INFO[c].raw}</span></td>}
                  <td className="py-1.5 pr-2">{missing ? <span className="font-sans text-slate-500">unavailable</span> : fmt(fused.normalized[c])}</td>
                  <td className="py-1.5 pr-2 text-slate-400">{w ? w.toFixed(2) : missing ? "excluded" : "0"}</td>
                  <td className="py-1.5 text-slate-100">{fused.contributions[c] != null ? fused.contributions[c].toFixed(3) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {evidence.orb && (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg bg-graphite-850 p-3 text-xs sm:grid-cols-3">
          <dt className="text-slate-500">Keypoints (query / ref)</dt><dd className="font-mono sm:col-span-2">{evidence.orb.query_keypoints} / {evidence.orb.reference_keypoints}</dd>
          <dt className="text-slate-500">Ratio-test + mutual matches</dt><dd className="font-mono sm:col-span-2">{evidence.orb.good_matches}</dd>
          <dt className="text-slate-500">RANSAC inliers</dt><dd className="font-mono sm:col-span-2">{evidence.orb.inliers} ({(evidence.orb.inlier_ratio * 100).toFixed(1)}%) · {evidence.orb.geometric_model ?? "no model"}</dd>
        </dl>
      )}
      <p className="text-[11px] leading-relaxed text-slate-500">
        Evidence summary, not a causal explanation. Each raw value is mapped to [0, 1] with a clipped linear ramp; the fused
        score is the weighted mean over available components. It is a similarity score, not a probability.
      </p>
    </div>
  );
}
