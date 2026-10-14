import { Camera, CameraOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";

/** Optional webcam capture; produces a JPEG File. Hidden gracefully if unsupported. */
export function WebcamCapture({ onCapture }: { onCapture: (f: File) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string>();
  const supported = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);
  useEffect(() => {
    if (video.current && stream) video.current.srcObject = stream;
  }, [stream]);

  if (!supported) return <p className="text-xs text-slate-500">Webcam capture is not supported in this browser.</p>;

  const start = async () => {
    try {
      setError(undefined);
      setStream(await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, facingMode: "environment" } }));
    } catch (e) {
      setError(`Camera unavailable: ${(e as Error).message}`);
    }
  };
  const stop = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
  };
  const capture = () => {
    const v = video.current;
    if (!v) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")?.drawImage(v, 0, 0);
    c.toBlob((b) => {
      if (b) onCapture(new File([b], `webcam-${Date.now()}.jpg`, { type: "image/jpeg" }));
      stop();
    }, "image/jpeg", 0.92);
  };

  return (
    <div className="space-y-2">
      {stream ? (
        <>
          <video ref={video} autoPlay playsInline muted className="w-full rounded-lg border border-graphite-700" />
          <div className="flex gap-2">
            <button type="button" className="btn-primary" onClick={capture}><Camera className="h-4 w-4" /> Capture</button>
            <button type="button" className="btn-secondary" onClick={stop}><CameraOff className="h-4 w-4" /> Close</button>
          </div>
        </>
      ) : (
        <button type="button" className="btn-secondary" onClick={start}><Camera className="h-4 w-4" /> Use webcam</button>
      )}
      {error && <p className="text-xs text-rose-300">{error}</p>}
    </div>
  );
}
