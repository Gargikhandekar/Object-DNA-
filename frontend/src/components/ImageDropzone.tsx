import { ImagePlus, X } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { useObjectUrls } from "../hooks/useApi";

const ACCEPT = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_MB = 15;

/** Drag-and-drop / click image picker with client-side pre-validation and previews. */
export function ImageDropzone({ files, onChange, multiple = true, label, compact = false }: {
  files: File[]; onChange: (f: File[]) => void; multiple?: boolean; label?: string; compact?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [rejected, setRejected] = useState<string[]>([]);
  const urls = useObjectUrls(files);

  const add = (list: FileList | null) => {
    if (!list) return;
    const ok: File[] = [];
    const bad: string[] = [];
    Array.from(list).forEach((f) => {
      if (!ACCEPT.includes(f.type)) bad.push(`${f.name}: unsupported type`);
      else if (f.size > MAX_MB * 1024 * 1024) bad.push(`${f.name}: larger than ${MAX_MB} MB`);
      else ok.push(f);
    });
    setRejected(bad);
    onChange(multiple ? [...files, ...ok] : ok.slice(0, 1));
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    add(e.dataTransfer.files);
  };

  return (
    <div>
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        className={`flex w-full flex-col items-center justify-center rounded-xl border-2 border-dashed transition ${compact ? "py-6" : "py-10"} ${drag ? "border-accent bg-accent/5" : "border-graphite-600 hover:border-graphite-500 hover:bg-graphite-850"}`}
      >
        <ImagePlus className="mb-2 h-7 w-7 text-accent" aria-hidden />
        <span className="text-sm font-medium text-slate-200">{label ?? (multiple ? "Drop images here or click to browse" : "Drop an image or click to browse")}</span>
        <span className="mt-1 text-xs text-slate-500">JPEG, PNG, WebP or BMP · up to {MAX_MB} MB each</span>
      </button>
      <input ref={input} type="file" accept={ACCEPT.join(",")} multiple={multiple} className="hidden" onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
      {rejected.length > 0 && <ul className="mt-2 text-xs text-rose-300">{rejected.map((r) => <li key={r}>{r}</li>)}</ul>}
      {files.length > 0 && multiple && (
        <div className="mt-4 grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-5">
          {files.map((f, i) => (
            <figure key={`${f.name}-${i}`} className="group relative overflow-hidden rounded-lg border border-graphite-700 bg-graphite-850">
              <img src={urls[i]} alt={f.name} className="aspect-square w-full object-cover" />
              <figcaption className="truncate px-2 py-1 text-[10px] text-slate-400">{f.name}</figcaption>
              <button type="button" aria-label={`Remove ${f.name}`} onClick={() => onChange(files.filter((_, j) => j !== i))}
                className="absolute right-1 top-1 rounded-full bg-black/70 p-1 text-slate-200 opacity-80 hover:opacity-100">
                <X className="h-3 w-3" />
              </button>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
