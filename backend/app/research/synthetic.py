"""Procedural synthetic dataset for pipeline sanity checks.

THIS IS NOT A SUBSTITUTE FOR REAL DATA. It exists so the full pipeline,
tests and experiments can run without a camera. Results on it only show
that the code behaves sensibly; they say nothing about real-world accuracy.

Design: several *models* (shared printed design, e.g. the same product
label with a repeated logo). Each model has several physical *instances*
that share the design but carry instance-specific marks (scratches,
speckles, smudges, slight print offset). Each instance is rendered under
controlled acquisition conditions (viewpoint, lighting, background,
distance, occlusion, blur) in separate capture sessions.
"""

from __future__ import annotations

import csv
import zlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CONDITIONS = (
    "same",
    "viewpoint_moderate",
    "viewpoint_large",
    "lighting",
    "background",
    "distance",
    "occlusion",
    "blur_lowres",
)

OBJ_W, OBJ_H = 300, 420
CANVAS = 480


def _model_design(rng: np.random.Generator) -> np.ndarray:
    """Printed design shared by every instance of a model."""
    base = np.array(rng.integers(60, 230, 3), np.uint8)
    img = np.full((OBJ_H, OBJ_W, 3), base, np.uint8)
    # Stripes / bands.
    for _ in range(rng.integers(2, 5)):
        y = int(rng.integers(0, OBJ_H - 40))
        h = int(rng.integers(10, 50))
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (0, y), (OBJ_W, y + h), col, -1)
    # Repeated logo: a random polygon stamped on a grid (hard for local features).
    pts = (rng.random((rng.integers(3, 7), 2)) * 36).astype(np.int32)
    logo_col = tuple(int(c) for c in rng.integers(0, 255, 3))
    step = int(rng.integers(55, 80))
    for yy in range(20, OBJ_H - 30, step):
        for xx in range(15, OBJ_W - 30, step):
            cv2.fillPoly(img, [pts + np.array([xx, yy])], logo_col)
    # Text-like glyph rows.
    for _ in range(rng.integers(1, 3)):
        y = int(rng.integers(40, OBJ_H - 20))
        cv2.putText(img, "".join(chr(int(c)) for c in rng.integers(65, 90, 6)), (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _instance_marks(design: np.ndarray, rng: np.random.Generator, wear: float = 1.0) -> np.ndarray:
    """Instance-specific irregularities on top of the shared design."""
    img = design.astype(np.float32)
    # Slight print misregistration.
    M = np.float32([[1, 0, rng.uniform(-2, 2)], [0, 1, rng.uniform(-2, 2)]])
    img = cv2.warpAffine(img, M, (OBJ_W, OBJ_H), borderMode=cv2.BORDER_REFLECT)
    overlay = img.copy()
    for _ in range(int(rng.integers(4, 9) * wear)):  # scratches
        p0 = rng.integers([0, 0], [OBJ_W, OBJ_H])
        length = rng.uniform(25, 110)
        ang = rng.uniform(0, np.pi)
        p1 = (p0 + length * np.array([np.cos(ang), np.sin(ang)])).astype(int)
        shade = float(rng.choice([25.0, 235.0]))
        cv2.line(overlay, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (shade,) * 3,
                 int(rng.integers(1, 3)), cv2.LINE_AA)
    for _ in range(int(rng.integers(25, 60) * wear)):  # speckles / dents
        c = tuple(int(v) for v in rng.integers([0, 0], [OBJ_W, OBJ_H]))
        shade = float(rng.uniform(0, 255))
        cv2.circle(overlay, c, int(rng.integers(1, 4)), (shade,) * 3, -1, cv2.LINE_AA)
    for _ in range(int(rng.integers(1, 3))):  # smudges / wear patches
        c = tuple(int(v) for v in rng.integers([20, 20], [OBJ_W - 20, OBJ_H - 20]))
        blob = np.zeros((OBJ_H, OBJ_W), np.float32)
        cv2.ellipse(blob, c, (int(rng.integers(10, 30)), int(rng.integers(8, 25))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 6)[..., None]
        overlay = overlay * (1 - 0.35 * blob) + 0.35 * blob * float(rng.uniform(80, 220))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _background(kind: int, rng: np.random.Generator) -> np.ndarray:
    if kind == 0:  # desk-like plain gradient
        g = np.linspace(150, 200, CANVAS, dtype=np.float32)
        bg = np.stack([g[:, None].repeat(CANVAS, 1)] * 3, -1) * np.array([1.0, 0.97, 0.9])
    elif kind == 1:  # wood-like stripes
        x = np.arange(CANVAS)[None, :].repeat(CANVAS, 0).astype(np.float32)
        y = np.arange(CANVAS)[:, None].repeat(CANVAS, 1).astype(np.float32)
        v = 120 + 40 * np.sin(x / 9.0 + 3 * np.sin(y / 60.0))
        bg = np.stack([v * 1.1, v * 0.8, v * 0.55], -1)
    elif kind == 2:  # checker / tablecloth
        y, x = np.mgrid[0:CANVAS, 0:CANVAS]
        v = (((x // 40) + (y // 40)) % 2) * 70 + 120
        bg = np.stack([v, v * 0.9, v * 1.05], -1).astype(np.float32)
    else:  # cluttered noise
        bg = cv2.GaussianBlur(rng.uniform(40, 220, (CANVAS, CANVAS, 3)).astype(np.float32), (0, 0), 3)
    bg = bg + rng.normal(0, 3, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


@dataclass
class RenderParams:
    rot: float = 0.0  # degrees
    tilt_x: float = 0.0  # perspective strength
    tilt_y: float = 0.0
    scale: float = 1.0
    shift: tuple[float, float] = (0.0, 0.0)
    gain: float = 1.0
    gamma: float = 1.0
    light_grad: float = 0.0
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    background: int = 0
    occlusion: float = 0.0
    blur: float = 0.0
    lowres: float = 1.0


def render(texture: np.ndarray, p: RenderParams, rng: np.random.Generator) -> np.ndarray:
    bg = _background(p.background, rng).astype(np.float32)
    h, w = texture.shape[:2]
    base_scale = 0.78 * CANVAS / max(h, w) * p.scale
    cw, ch = w * base_scale, h * base_scale
    cx, cy = CANVAS / 2 + p.shift[0], CANVAS / 2 + p.shift[1]
    corners = np.float32([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
    # Perspective: shrink the far edges.
    corners[:2, 0] *= (1 - p.tilt_y)
    corners[2:, 0] *= (1 + p.tilt_y)
    corners[[0, 3], 1] *= (1 - p.tilt_x)
    corners[[1, 2], 1] *= (1 + p.tilt_x)
    a = np.deg2rad(p.rot)
    R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = corners @ R.T + np.float32([cx, cy])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(texture, H, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (CANVAS, CANVAS))
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    # Soft shadow.
    shadow = cv2.GaussianBlur(np.roll(mask[..., 0], (8, 8), (0, 1)), (0, 0), 6)[..., None]
    out = bg * (1 - 0.35 * shadow)
    out = out * (1 - mask) + warped.astype(np.float32) * mask

    # Illumination: gain, gradient, colour temperature, gamma.
    y = np.linspace(-1, 1, CANVAS, dtype=np.float32)[:, None, None]
    out = out * p.gain * (1 + p.light_grad * y) * np.float32(p.tint)
    out = 255.0 * np.power(np.clip(out / 255.0, 0, 1), p.gamma)

    if p.occlusion > 0:
        ow = int(CANVAS * np.sqrt(p.occlusion) * 0.9)
        x0 = int(rng.integers(int(cx - cw / 2), max(int(cx - cw / 2) + 1, int(cx + cw / 2 - ow))))
        y0 = int(rng.integers(int(cy - ch / 2), max(int(cy - ch / 2) + 1, int(cy + ch / 2 - ow))))
        col = tuple(float(c) for c in rng.uniform(30, 220, 3))
        cv2.rectangle(out, (x0, y0), (x0 + ow, y0 + int(ow * 0.8)), col, -1)
    out = np.clip(out + rng.normal(0, 2.5, out.shape), 0, 255).astype(np.uint8)
    if p.blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), p.blur)
    if p.lowres < 1:
        small = cv2.resize(out, None, fx=p.lowres, fy=p.lowres, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (CANVAS, CANVAS), interpolation=cv2.INTER_LINEAR)
    return out


def condition_params(cond: str, rng: np.random.Generator, ref_background: int) -> RenderParams:
    p = RenderParams(rot=float(rng.uniform(-5, 5)), shift=tuple(rng.uniform(-10, 10, 2)),
                     gain=float(rng.uniform(0.95, 1.05)), background=ref_background)
    if cond == "viewpoint_moderate":
        p.rot = float(rng.choice([-1, 1]) * rng.uniform(12, 25))
        p.tilt_x, p.tilt_y = (float(v) for v in rng.uniform(-0.12, 0.12, 2))
    elif cond == "viewpoint_large":
        p.rot = float(rng.choice([-1, 1]) * rng.uniform(35, 70))
        p.tilt_x, p.tilt_y = (float(v) * rng.choice([-1, 1]) for v in rng.uniform(0.15, 0.28, 2))
    elif cond == "lighting":
        p.gain = float(rng.uniform(0.55, 1.35))
        p.gamma = float(rng.uniform(0.7, 1.4))
        p.light_grad = float(rng.uniform(-0.35, 0.35))
        p.tint = tuple(float(v) for v in rng.uniform(0.85, 1.15, 3))
    elif cond == "background":
        p.background = int((ref_background + rng.integers(1, 4)) % 4)
    elif cond == "distance":
        p.scale = float(rng.choice([rng.uniform(0.5, 0.65), rng.uniform(1.15, 1.3)]))
    elif cond == "occlusion":
        p.occlusion = float(rng.uniform(0.08, 0.16))
    elif cond == "blur_lowres":
        p.blur = float(rng.uniform(1.2, 2.2))
        p.lowres = float(rng.uniform(0.35, 0.55))
    return p


def generate_dataset(
    out_dir: Path,
    n_models: int = 5,
    instances_per_model: int = 4,
    unregistered_per_model: int = 1,
    n_references: int = 3,
    query_sessions: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("s2", CONDITIONS),
        ("s3", ("same", "viewpoint_moderate", "lighting", "background")),
    ),
    seed: int = 7,
) -> Path:
    """Write images + ``manifest.csv`` and return the manifest path."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    categories = ["bottle", "notebook", "phone_case", "mug", "wallet", "box", "card", "label"]
    obj_counter = 0
    for m in range(n_models):
        design = _model_design(np.random.default_rng(seed * 1000 + m))
        category = categories[m % len(categories)]
        model_id = f"model_{m + 1:02d}"
        for i in range(instances_per_model + unregistered_per_model):
            obj_counter += 1
            registered = i < instances_per_model
            obj_id = f"object_{obj_counter:03d}"
            inst_rng = np.random.default_rng(seed * 100_000 + obj_counter)
            texture = _instance_marks(design, inst_rng)
            odir = out_dir / "objects" / obj_id
            (odir / "reference").mkdir(parents=True, exist_ok=True)
            (odir / "query").mkdir(parents=True, exist_ok=True)
            meta = {"name": f"{category.replace('_', ' ').title()} #{i + 1} ({model_id})",
                    "category": category, "model": model_id, "registered": registered,
                    "synthetic": True}
            (odir / "object.json").write_text(json.dumps(meta, indent=2))
            ref_bg = 0
            if registered:
                for k in range(n_references):
                    p = RenderParams(rot=float(inst_rng.uniform(-8, 8)) + (k - 1) * 10,
                                     tilt_x=float(inst_rng.uniform(-0.06, 0.06)),
                                     tilt_y=float(inst_rng.uniform(-0.06, 0.06)),
                                     scale=float(inst_rng.uniform(0.9, 1.05)),
                                     background=ref_bg)
                    img = render(texture, p, inst_rng)
                    name = f"s1__reference__{k + 1:02d}.jpg"
                    cv2.imwrite(str(odir / "reference" / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    rows.append(_row(obj_id, f"objects/{obj_id}/reference/{name}", "reference", "s1",
                                     "reference", category, model_id, registered))
            for session, conds in query_sessions:
                for cond in conds:
                    p = condition_params(cond, inst_rng, ref_bg)
                    img = render(texture, p, inst_rng)
                    name = f"{session}__{cond}__01.jpg"
                    cv2.imwrite(str(odir / "query" / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    rows.append(_row(obj_id, f"objects/{obj_id}/query/{name}", "query", session, cond,
                                     category, model_id, registered))
    manifest = out_dir / "manifest.csv"
    write_manifest(manifest, rows)
    (out_dir / "DATASET_INFO.json").write_text(json.dumps({
        "name": out_dir.name, "synthetic": True, "seed": seed, "models": n_models,
        "instances_per_model": instances_per_model, "unregistered_per_model": unregistered_per_model,
        "note": "Procedurally generated. Use only for pipeline sanity checks, not as evidence of real-world "
                "performance.",
    }, indent=2))
    return manifest


MANIFEST_FIELDS = ["object_id", "image_path", "role", "session", "condition", "category", "model", "registered"]


def _row(obj_id, path, role, session, cond, category, model, registered) -> dict:
    return {"object_id": obj_id, "image_path": path, "role": role, "session": session, "condition": cond,
            "category": category, "model": model, "registered": str(bool(registered)).lower()}


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def synthetic_object_image(seed: int = 0, condition: str = "same", model_seed: int | None = None) -> np.ndarray:
    """Single rendered image (for tests). Same ``seed`` => same physical instance."""
    design = _model_design(np.random.default_rng(1000 + (seed if model_seed is None else model_seed)))
    texture = _instance_marks(design, np.random.default_rng(50_000 + seed))
    rng = np.random.default_rng(zlib.crc32(f"{seed}:{condition}".encode()))
    return render(texture, condition_params(condition, rng, 0), rng)
