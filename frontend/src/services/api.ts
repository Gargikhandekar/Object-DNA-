import type {
  CompareResponse, Dashboard, DatasetInfo, ExperimentRun, FusionConfig, Health, HistoryList, IdentifyResponse,
  ObjectDetail, ObjectHistory, ObjectList, ObservationResult, RegistrationResult, Status,
} from "../types";

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  status: number;
  details: unknown;
  constructor(status: number, message: string, details?: unknown) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

/** Turn FastAPI error payloads into a readable message. */
export function describeError(detail: unknown): string {
  if (!detail) return "Request failed.";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: (string | number)[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".") || "request"}: ${d.msg}`)
      .join("; ");
  }
  if (typeof detail === "object") {
    const d = detail as { message?: string; files?: { filename: string; error: string }[]; conflicts?: { filename: string; object_id: string; object_name: string }[] };
    const parts = [d.message ?? "Request failed."];
    d.files?.forEach((f) => parts.push(`${f.filename}: ${f.error}`));
    d.conflicts?.forEach((c) => parts.push(`${c.filename} is already a reference of ${c.object_name} (${c.object_id})`));
    return parts.join(" ");
  }
  return String(detail);
}

export function url(path: string): string {
  return `${API_BASE}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url(path), init);
  } catch {
    throw new ApiError(0, "Cannot reach the Object DNA API. Is the backend running on port 8000?");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown = undefined;
  try {
    body = text ? JSON.parse(text) : undefined;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | undefined)?.detail ?? body;
    throw new ApiError(res.status, describeError(detail), detail);
  }
  return body as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  dashboard: () => request<Dashboard>("/api/dashboard"),

  listObjects: (params: { q?: string; category?: string; page?: number; page_size?: number } = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && qs.set(k, String(v)));
    return request<ObjectList>(`/api/objects?${qs}`);
  },
  getObject: (id: string) => request<ObjectDetail>(`/api/objects/${encodeURIComponent(id)}`),
  registerObject: (data: { name: string; category: string; description?: string; files: File[]; capture_metadata?: Record<string, string>[] }) => {
    const fd = new FormData();
    fd.append("name", data.name);
    fd.append("category", data.category);
    if (data.description) fd.append("description", data.description);
    if (data.capture_metadata) fd.append("capture_metadata", JSON.stringify(data.capture_metadata));
    data.files.forEach((f) => fd.append("files", f, f.name));
    return request<RegistrationResult>("/api/objects", { method: "POST", body: fd });
  },
  addImages: (id: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    return request<RegistrationResult>(`/api/objects/${encodeURIComponent(id)}/images`, { method: "POST", body: fd });
  },
  deleteImage: (id: string, imageId: number) =>
    request<void>(`/api/objects/${encodeURIComponent(id)}/images/${imageId}`, { method: "DELETE" }),
  deleteObject: (id: string) => request<void>(`/api/objects/${encodeURIComponent(id)}`, { method: "DELETE" }),
  objectHistory: (id: string) => request<ObjectHistory>(`/api/objects/${encodeURIComponent(id)}/history`),
  addObservation: (id: string, file: File, note?: string, force = false) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    if (note) fd.append("note", note);
    fd.append("force", String(force));
    return request<ObservationResult>(`/api/objects/${encodeURIComponent(id)}/observations`, { method: "POST", body: fd });
  },

  identify: (file: File, topK = 5) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("top_k", String(topK));
    return request<IdentifyResponse>("/api/identify", { method: "POST", body: fd });
  },
  compare: (a: File, b: File) => {
    const fd = new FormData();
    fd.append("file_a", a, a.name);
    fd.append("file_b", b, b.name);
    return request<CompareResponse>("/api/compare", { method: "POST", body: fd });
  },
  history: (params: { page?: number; page_size?: number; status?: Status | ""; object_id?: string } = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && qs.set(k, String(v)));
    return request<HistoryList>(`/api/history?${qs}`);
  },

  fusion: () => request<FusionConfig>("/api/settings/fusion"),
  setDecision: (accept_threshold: number, uncertain_threshold: number, margin_threshold: number) =>
    request<FusionConfig>("/api/settings/decision", json("PUT", { accept_threshold, uncertain_threshold, margin_threshold })),
  resetFusion: () => request<FusionConfig>("/api/settings/fusion/reset", { method: "POST" }),
  backfillEmbeddings: () => request<{ computed: number }>("/api/settings/backfill-embeddings", { method: "POST" }),

  experiments: () => request<ExperimentRun[]>("/api/experiments"),
  experiment: (uid: string) => request<ExperimentRun>(`/api/experiments/${encodeURIComponent(uid)}`),
  runExperiment: (body: { experiment: string; dataset: string; seed?: number; val_fraction?: number; max_pairs?: number; perturbations?: boolean; methods?: string[] | null }) =>
    request<ExperimentRun>("/api/experiments/run", json("POST", body)),
  deleteExperiment: (uid: string) => request<void>(`/api/experiments/${encodeURIComponent(uid)}`, { method: "DELETE" }),
  applyFusion: (uid: string) => request<FusionConfig>(`/api/experiments/${encodeURIComponent(uid)}/apply-fusion`, { method: "POST" }),
  datasets: () => request<DatasetInfo[]>("/api/experiments/datasets"),
  generateSynthetic: () => request<DatasetInfo>("/api/experiments/datasets/synthetic", json("POST", {})),
  artifactUrl: (uid: string, name: string) => url(`/api/experiments/${encodeURIComponent(uid)}/artifacts/${encodeURIComponent(name)}`),
  exportCsvUrl: (uid: string) => url(`/api/experiments/${encodeURIComponent(uid)}/export.csv`),
};
