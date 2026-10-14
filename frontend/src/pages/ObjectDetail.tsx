import { ArrowLeft, History, ImagePlus, Trash2 } from "lucide-react";
import { Fragment, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { ImageDropzone } from "../components/ImageDropzone";
import { useToast } from "../components/Toast";
import { ErrorBox, PageHeader, Section, Spinner, dateTime } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, url } from "../services/api";
import type { ReferenceImage } from "../types";

export default function ObjectDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const toast = useToast();
  const { data, error, loading, reload, setData } = useApi(() => api.getObject(id), [id]);
  const [files, setFiles] = useState<File[]>([]);
  const [adding, setAdding] = useState(false);
  const [confirmObj, setConfirmObj] = useState(false);
  const [confirmImg, setConfirmImg] = useState<ReferenceImage | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);

  if (loading && !data) return <Spinner />;
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  if (!data) return null;
  const img = data.images.find((i) => i.id === selected) ?? data.images[0];

  const addImages = async () => {
    setAdding(true);
    try {
      const r = await api.addImages(id, files);
      setData(r.object);
      setFiles([]);
      toast("success", `Added ${files.length} reference image(s).`);
      r.warnings.forEach((w) => toast("info", `${w.filename ?? ""} ${w.message}`));
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setAdding(false);
    }
  };
  const deleteObject = async () => {
    setBusy(true);
    try {
      await api.deleteObject(id);
      toast("success", `Deleted ${data.name} and all its images and fingerprints.`);
      nav("/library");
    } catch (e) {
      toast("error", (e as Error).message);
      setBusy(false);
    }
  };
  const deleteImage = async () => {
    if (!confirmImg) return;
    setBusy(true);
    try {
      await api.deleteImage(id, confirmImg.id);
      toast("success", "Reference image removed.");
      setConfirmImg(null);
      setSelected(null);
      reload();
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <Link to="/library" className="btn-ghost -ml-3 text-xs"><ArrowLeft className="h-4 w-4" /> Library</Link>
      <PageHeader
        title={data.name}
        subtitle={<><span className="font-mono text-accent">{data.public_id}</span> · {data.category} · registered {dateTime(data.created_at)} · fingerprint {data.fingerprint_status}{data.description && <><br />{data.description}</>}</>}
        actions={<div className="flex gap-2">
          <Link to={`/history?object=${data.public_id}`} className="btn-secondary"><History className="h-4 w-4" /> History ({data.observation_count})</Link>
          <button className="btn-danger" onClick={() => setConfirmObj(true)}><Trash2 className="h-4 w-4" /> Delete object</button>
        </div>}
      />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Section title={`Reference images (${data.images.length})`} subtitle="Normalised images used for fingerprinting. Click to inspect metadata.">
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5">
              {data.images.map((i) => (
                <button key={i.id} onClick={() => setSelected(i.id)} className={`overflow-hidden rounded-lg border-2 ${img?.id === i.id ? "border-accent" : "border-transparent"}`}>
                  <img src={url(i.url)} alt={i.original_filename ?? `reference ${i.id}`} className="aspect-square w-full object-cover" loading="lazy" />
                </button>
              ))}
            </div>
          </Section>
          <Section title="Add reference images" subtitle="More views (angles, distances, lighting) usually make identification more reliable.">
            <ImageDropzone files={files} onChange={setFiles} compact />
            <button className="btn-primary mt-4" disabled={!files.length || adding} onClick={addImages}><ImagePlus className="h-4 w-4" />{adding ? "Fingerprinting…" : `Add ${files.length || ""} image(s)`}</button>
          </Section>
        </div>
        {img && (
          <Section title={`Image #${img.id}`} subtitle={img.original_filename ?? undefined} actions={data.images.length > 1 && <button className="btn-ghost px-2 py-1 text-xs text-rose-300" onClick={() => setConfirmImg(img)}><Trash2 className="h-3.5 w-3.5" /> Remove</button>}>
            <img src={url(img.url)} alt="" className="mb-3 w-full rounded-lg" />
            <dl className="grid grid-cols-2 gap-y-1 text-xs">
              <dt className="text-slate-500">Original size</dt><dd className="font-mono">{img.width} × {img.height}</dd>
              <dt className="text-slate-500">Uploaded</dt><dd>{dateTime(img.uploaded_at)}</dd>
              <dt className="text-slate-500">Sharpness (Lap. var)</dt><dd className="font-mono">{img.quality.sharpness_laplacian_var ?? "—"}</dd>
              <dt className="text-slate-500">Brightness / contrast</dt><dd className="font-mono">{img.quality.brightness_mean ?? "—"} / {img.quality.contrast_std ?? "—"}</dd>
              <dt className="text-slate-500">Foreground mask</dt><dd>{img.quality.foreground_mask_reliable ? `reliable (${((img.quality.foreground_fraction ?? 0) * 100).toFixed(0)}% of frame)` : "not reliable — full image used"}</dd>
              {Object.entries(img.capture_metadata).map(([k, v]) => <Fragment key={k}><dt className="text-slate-500">{k}</dt><dd>{v}</dd></Fragment>)}
            </dl>
            {img.quality.warnings && img.quality.warnings.length > 0 && <ul className="mt-2 text-xs text-amber-300">{img.quality.warnings.map((w) => <li key={w}>⚠ {w}</li>)}</ul>}
            {img.quality.embedding_message && <p className="mt-2 text-xs text-amber-300/80">Deep embedding: {img.quality.embedding_message}</p>}
            <h3 className="mb-2 mt-4 text-xs font-semibold uppercase tracking-wider text-slate-400">Fingerprint records</h3>
            <table className="w-full text-[11px]">
              <thead className="text-left text-slate-500"><tr><th>Extractor</th><th>Version</th><th>Dim</th></tr></thead>
              <tbody className="font-mono">
                {img.fingerprints.map((f) => (
                  <tr key={f.id} className="border-t border-graphite-800" title={JSON.stringify(f.summary)}>
                    <td className="py-1 text-slate-200">{f.extractor}</td><td className="text-slate-400">{f.feature_version}</td><td>{f.dimension ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {img.fingerprints[0] && <p className="mt-2 font-mono text-[10px] text-slate-500">preprocessing: {Object.entries(img.fingerprints[0].preprocessing).filter(([, v]) => typeof v !== "object").map(([k, v]) => `${k}=${String(v)}`).join(", ")}</p>}
          </Section>
        )}
      </div>
      <ConfirmDialog open={confirmObj} busy={busy} title={`Delete ${data.name}?`} message={`This permanently deletes ${data.public_id}, its ${data.images.length} reference images, all fingerprints and ${data.observation_count} history observations. Identification history entries keep the object ID for auditing.`} onCancel={() => setConfirmObj(false)} onConfirm={deleteObject} />
      <ConfirmDialog open={!!confirmImg} busy={busy} title="Remove reference image?" message="The image and its fingerprint files will be deleted." confirmLabel="Remove" onCancel={() => setConfirmImg(null)} onConfirm={deleteImage} />
    </div>
  );
}
