"""ORB local features, Hamming matching and RANSAC geometric verification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ORB_VERSION = "orb-v1"


@dataclass
class OrbFeatures:
    keypoints: np.ndarray  # (N, 6) float32: x, y, size, angle, response, octave
    descriptors: np.ndarray  # (N, 32) uint8

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])

    @staticmethod
    def empty() -> OrbFeatures:
        return OrbFeatures(np.zeros((0, 6), np.float32), np.zeros((0, 32), np.uint8))


@dataclass
class OrbMatchResult:
    query_keypoints: int
    reference_keypoints: int
    raw_matches: int  # knn candidates considered
    good_matches: int  # after Lowe ratio + mutual cross-check
    inliers: int  # geometrically consistent (RANSAC)
    inlier_ratio: float  # inliers / good_matches
    model: str | None  # "homography", "affine" or None
    matches: list[tuple[float, float, float, float, bool]]  # qx,qy,rx,ry,inlier

    def summary(self) -> dict:
        return {
            "query_keypoints": self.query_keypoints,
            "reference_keypoints": self.reference_keypoints,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "geometric_model": self.model,
        }


def extract_orb(gray: np.ndarray, mask: np.ndarray | None = None, nfeatures: int = 1500) -> OrbFeatures:
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
                         patchSize=31, fastThreshold=10)
    kps, desc = orb.detectAndCompute(gray, mask)
    if desc is None or not kps:
        return OrbFeatures.empty()
    arr = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave] for k in kps], np.float32)
    return OrbFeatures(arr, desc.astype(np.uint8))


_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _ratio_matches(d1: np.ndarray, d2: np.ndarray, ratio: float) -> dict[int, tuple[int, float]]:
    out: dict[int, tuple[int, float]] = {}
    if len(d1) < 2 or len(d2) < 2:
        return out
    for pair in _BF.knnMatch(d1, d2, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            out[m.queryIdx] = (m.trainIdx, m.distance)
    return out


def match_orb(
    q: OrbFeatures,
    r: OrbFeatures,
    ratio: float = 0.75,
    reproj_threshold: float = 6.0,
    keep_matches: bool = False,
) -> OrbMatchResult:
    """Ratio-test + mutual (symmetric) matching followed by RANSAC.

    A homography is tried first (appropriate for planar or distant surfaces);
    if it fails, a 4-DoF partial affine model is used, which is more tolerant
    for small numbers of matches.
    """
    if q.count < 2 or r.count < 2:
        return OrbMatchResult(q.count, r.count, 0, 0, 0, 0.0, None, [])

    fwd = _ratio_matches(q.descriptors, r.descriptors, ratio)
    bwd = _ratio_matches(r.descriptors, q.descriptors, ratio)
    good = [(qi, ri) for qi, (ri, _) in fwd.items() if bwd.get(ri, (-1, 0))[0] == qi]

    n_good = len(good)
    inlier_mask = np.zeros(n_good, bool)
    model = None
    if n_good >= 4:
        src = np.float32([q.keypoints[i, :2] for i, _ in good]).reshape(-1, 1, 2)
        dst = np.float32([r.keypoints[j, :2] for _, j in good]).reshape(-1, 1, 2)
        H = None
        if n_good >= 8:
            H, m = cv2.findHomography(src, dst, cv2.RANSAC, reproj_threshold, maxIters=2000,
                                      confidence=0.995)
            if H is not None and m is not None and _homography_plausible(H):
                inlier_mask = m.ravel().astype(bool)
                model = "homography"
            else:
                H = None
        if H is None:
            A, m = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                               ransacReprojThreshold=reproj_threshold,
                                               maxIters=2000, confidence=0.995)
            if A is not None and m is not None:
                inlier_mask = m.ravel().astype(bool)
                model = "affine"
    # A model supported by fewer than 4 correspondences is not evidence.
    n_inliers = int(inlier_mask.sum()) if inlier_mask.sum() >= 4 else 0
    if n_inliers == 0:
        inlier_mask[:] = False
        model = None

    matches: list[tuple[float, float, float, float, bool]] = []
    if keep_matches:
        for (qi, ri), inl in zip(good, inlier_mask):
            matches.append((float(q.keypoints[qi, 0]), float(q.keypoints[qi, 1]),
                            float(r.keypoints[ri, 0]), float(r.keypoints[ri, 1]), bool(inl)))
    return OrbMatchResult(
        query_keypoints=q.count,
        reference_keypoints=r.count,
        raw_matches=len(fwd),
        good_matches=n_good,
        inliers=n_inliers,
        inlier_ratio=(n_inliers / n_good) if n_good else 0.0,
        model=model,
        matches=matches,
    )


def _homography_plausible(H: np.ndarray) -> bool:
    """Reject degenerate homographies (extreme scale, reflections, projective blow-up)."""
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-9:
        return False
    H = H / H[2, 2]
    det = np.linalg.det(H[:2, :2])
    if det <= 0.05 or det > 20:
        return False
    if np.abs(H[2, :2]).max() > 0.004:
        return False
    return True
