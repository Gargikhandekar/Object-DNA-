import { CheckCircle2, Fingerprint, Lightbulb } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { ImageDropzone } from "../components/ImageDropzone";
import { useToast } from "../components/Toast";
import { WebcamCapture } from "../components/WebcamCapture";
import { ErrorBox, PageHeader, Section } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, url } from "../services/api";
import type { RegistrationResult } from "../types";

const STAGES = ["Validating images", "Normalising & segmenting", "Extracting ORB / LBP / colour / shape", "Computing deep embedding", "Saving fingerprints"];

export default function Register() {
  const [name, setName] = useState("");
  const [category, setCategory] = useState("");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<string>();
  const [result, setResult] = useState<RegistrationResult>();
  const toast = useToast();
  const cats = useApi(() => api.listObjects({ page_size: 1 }), []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !category.trim() || files.length === 0) return;
    setBusy(true);
    setError(undefined);
    setStage(0);
    // Progress stages are indicative: the server processes the whole request in one call.
    const timer = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), Math.max(400, 350 * files.length));
    try {
      const r = await api.registerObject({ name, category, description, files });
      setResult(r);
      toast("success", `Registered ${r.object.name} as ${r.object.public_id}`);
      setFiles([]);
      setName("");
      setDescription("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      clearInterval(timer);
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Register Object" subtitle="Enrol one specific physical object. Its identity is defined by its visual fingerprint — not by its name, category or file names. Two identical-looking products must be registered as two objects." />
      <div className="grid gap-6 lg:grid-cols-3">
        <form onSubmit={submit} className="space-y-6 lg:col-span-2">
          <Section title="Object metadata" subtitle="A unique object ID (ODNA-XXXXXXXX) is assigned automatically.">
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="label" htmlFor="name">Object name *</label>
                <input id="name" className="input" value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} placeholder="e.g. Blue steel bottle (dented base)" />
              </div>
              <div>
                <label className="label" htmlFor="category">Category *</label>
                <input id="category" className="input" list="cats" value={category} onChange={(e) => setCategory(e.target.value)} required maxLength={100} placeholder="e.g. bottle" />
                <datalist id="cats">{cats.data?.categories.map((c) => <option key={c} value={c} />)}</datalist>
              </div>
              <div className="sm:col-span-2">
                <label className="label" htmlFor="desc">Description (optional)</label>
                <textarea id="desc" className="input min-h-[80px]" value={description} onChange={(e) => setDescription(e.target.value)} maxLength={5000} placeholder="Distinguishing marks, e.g. scratch near the cap, sticker residue on the base" />
              </div>
            </div>
          </Section>

          <Section title="Reference images" subtitle={`${files.length} selected · capture the SAME physical object from several angles, distances and lighting conditions`}>
            <ImageDropzone files={files} onChange={setFiles} />
            <div className="mt-4"><WebcamCapture onCapture={(f) => setFiles((x) => [...x, f])} /></div>
          </Section>

          {error && <ErrorBox message={error} />}
          {busy && (
            <div className="card card-pad" aria-live="polite">
              <div className="mb-3 flex items-center gap-2 text-sm text-slate-200"><Fingerprint className="h-4 w-4 animate-pulse text-accent" /> Generating Object DNA…</div>
              <ol className="space-y-1 text-xs">
                {STAGES.map((s, i) => (
                  <li key={s} className={i < stage ? "text-emerald-400" : i === stage ? "text-accent" : "text-slate-600"}>{i < stage ? "✓" : i === stage ? "›" : "·"} {s}</li>
                ))}
              </ol>
            </div>
          )}
          <div className="flex gap-3">
            <button type="submit" className="btn-primary" disabled={busy || !name.trim() || !category.trim() || files.length === 0}>
              <Fingerprint className="h-4 w-4" /> {busy ? "Generating fingerprint…" : "Generate fingerprint & register"}
            </button>
            {files.length > 0 && !busy && <button type="button" className="btn-ghost" onClick={() => setFiles([])}>Clear images</button>}
          </div>
        </form>

        <aside className="space-y-6">
          <Section title="Capture guidance">
            <ul className="space-y-2 text-sm text-slate-400">
              {["Use 3–8 reference images of the exact same object.", "Vary the angle (front, sides, 30–45°) and the distance.", "Include close-ups of distinctive marks: scratches, dents, wear, stickers.", "Use at least two lighting conditions; avoid strong glare.", "Keep the object large in the frame and in focus.", "Plain backgrounds help the foreground mask; varied ones test robustness."].map((t) => (
                <li key={t} className="flex gap-2"><Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />{t}</li>
              ))}
            </ul>
          </Section>
          {result && (
            <Section title="Registered" subtitle={result.object.public_id}>
              <div className="flex items-center gap-2 text-sm text-emerald-300"><CheckCircle2 className="h-4 w-4" /> {result.object.name} · {result.object.image_count} images fingerprinted</div>
              <div className="mt-3 grid grid-cols-4 gap-2">{result.object.images.map((i) => <img key={i.id} src={url(i.url)} alt="" className="aspect-square rounded object-cover" />)}</div>
              {result.warnings.length > 0 && (
                <ul className="mt-3 space-y-1 text-xs text-amber-300">{result.warnings.map((w, i) => <li key={i}>⚠ {w.filename ? `${w.filename}: ` : ""}{w.message}</li>)}</ul>
              )}
              <Link to={`/library/${result.object.public_id}`} className="btn-secondary mt-4 w-full">Open object details</Link>
            </Section>
          )}
        </aside>
      </div>
    </div>
  );
}
