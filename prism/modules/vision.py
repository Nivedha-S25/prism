"""Module 3 — Stress Management (Computer Vision).

Analyses periodic webcam frames to estimate candidate stress / engagement.

Pipeline:

1. :class:`FramePreprocessor` resizes frames to 128x128 and drops frames whose
   face-detection confidence is below ``face_confidence_threshold`` (default
   0.85).
2. :class:`FaceFeatureExtractor` uses MediaPipe FaceMesh + OpenCV to derive:
   gaze-direction stability, blink rate, and head-pose variance (pitch, yaw,
   roll Euler angles).
3. :class:`StressAnalyzer` aggregates these engagement indicators into a single
   normalised Stress Score (``StressS``), where a higher value means calmer /
   more engaged.

MediaPipe / OpenCV are imported lazily. When unavailable the extractor produces
deterministic pseudo-features from frame pixel statistics so the pipeline still
returns a score.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from prism.config import Settings, get_settings
from prism.schemas import StressResult

logger = logging.getLogger(__name__)

FRAME_SIZE = 128

# Aggregation weights for the three engagement indicators.
_STRESS_WEIGHTS = {
    "gaze_stability": 0.40,
    "blink_rate_score": 0.30,
    "head_pose_stability": 0.30,
}

# Healthy blink rate band (blinks per minute). Distance from this band lowers
# the blink sub-score (both very low staring and very high fluttering hurt).
_BLINK_OPTIMAL = (10.0, 22.0)


@dataclass
class FrameFeatures:
    """Per-frame extracted features."""

    face_confidence: float
    gaze_offset: float            # 0 = centred, larger = looking away
    eye_aspect_ratio: float       # low values indicate a blink
    pitch: float                  # Euler angles (degrees)
    yaw: float
    roll: float


class FramePreprocessor:
    """Resize frames and filter out low-confidence (non-face) frames."""

    def __init__(self, frame_size: int = FRAME_SIZE, confidence_threshold: float = 0.85) -> None:
        self.frame_size = frame_size
        self.confidence_threshold = confidence_threshold

    def resize(self, frame: np.ndarray) -> np.ndarray:
        try:
            import cv2

            return cv2.resize(frame, (self.frame_size, self.frame_size))
        except Exception:
            # Pure-numpy nearest-neighbour fallback.
            return _resize_numpy(frame, self.frame_size)

    def keep(self, face_confidence: float) -> bool:
        return face_confidence >= self.confidence_threshold


class FaceFeatureExtractor:
    """Extracts gaze, blink and head-pose features from a frame.

    Uses MediaPipe FaceMesh when available; otherwise derives deterministic
    pseudo-features from pixel statistics.
    """

    def __init__(self) -> None:
        self._mesh = None

    @property
    def available(self) -> bool:
        try:
            import mediapipe  # noqa: F401
        except Exception:  # pragma: no cover - depends on env
            return False
        return True

    def _ensure_mesh(self):
        if self._mesh is None:
            import mediapipe as mp

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._mesh

    def extract(self, frame: np.ndarray) -> FrameFeatures:
        if self.available:
            try:
                return self._extract_mediapipe(frame)
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.warning("MediaPipe extraction failed (%s); using heuristic.", exc)
        return self._extract_heuristic(frame)

    def _extract_mediapipe(self, frame: np.ndarray) -> FrameFeatures:
        import cv2

        mesh = self._ensure_mesh()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return FrameFeatures(0.0, 1.0, 0.3, 0.0, 0.0, 0.0)

        lm = result.multi_face_landmarks[0].landmark
        pts = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float64)

        # Gaze offset: iris centre vs eye-corner midpoint (refined landmarks).
        left_iris = pts[468:473].mean(axis=0)
        right_iris = pts[473:478].mean(axis=0)
        eye_center = (pts[33] + pts[263]) / 2.0
        gaze_offset = float(np.linalg.norm((left_iris + right_iris) / 2.0 - eye_center))

        ear = float((_eye_aspect_ratio(pts, "left") + _eye_aspect_ratio(pts, "right")) / 2.0)
        pitch, yaw, roll = _head_pose_from_landmarks(pts)
        confidence = 1.0  # FaceMesh returns a face only above its threshold.
        return FrameFeatures(confidence, gaze_offset, ear, pitch, yaw, roll)

    @staticmethod
    def _extract_heuristic(frame: np.ndarray) -> FrameFeatures:
        """Deterministic pseudo-features from frame statistics (no MediaPipe)."""
        arr = np.asarray(frame, dtype=np.float64)
        if arr.size == 0:
            return FrameFeatures(0.0, 1.0, 0.3, 0.0, 0.0, 0.0)
        mean = float(arr.mean()) / 255.0
        std = float(arr.std()) / 255.0
        # Map statistics into plausible, bounded feature values.
        confidence = float(min(0.80 + std, 1.0))
        gaze_offset = float(min(abs(mean - 0.5), 0.5))
        ear = float(0.15 + 0.2 * std)
        pitch = float((mean - 0.5) * 20.0)
        yaw = float((std - 0.25) * 20.0)
        roll = float((mean - std - 0.25) * 10.0)
        return FrameFeatures(confidence, gaze_offset, ear, pitch, yaw, roll)


@dataclass
class StressAnalyzer:
    """Module 3 orchestrator producing the Stress Score (StressS)."""

    settings: Settings | None = None
    blink_ear_threshold: float = 0.20
    fps: float = 1.0  # frames are typically periodic screenshots (1 per second)
    preprocessor: FramePreprocessor = field(init=False)
    extractor: FaceFeatureExtractor = field(init=False)

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        self.preprocessor = FramePreprocessor(
            self.settings.frame_size, self.settings.face_confidence_threshold
        )
        self.extractor = FaceFeatureExtractor()

    def analyze(self, frames: Sequence[np.ndarray]) -> StressResult:
        """Analyse a sequence of webcam frames (numpy BGR arrays)."""
        kept: list[FrameFeatures] = []
        dropped = 0
        for frame in frames:
            resized = self.preprocessor.resize(np.asarray(frame))
            feats = self.extractor.extract(resized)
            if self.preprocessor.keep(feats.face_confidence):
                kept.append(feats)
            else:
                dropped += 1

        used_fallback = not self.extractor.available
        if not kept:
            return StressResult(0.0, 0.0, 0.0, 0, dropped, 0.0, True,
                                {"reason": "no_valid_frames"})

        return self._aggregate(kept, dropped, used_fallback)

    def _aggregate(
        self, feats: list[FrameFeatures], dropped: int, used_fallback: bool
    ) -> StressResult:
        gaze_offsets = np.array([f.gaze_offset for f in feats])
        # Gaze stability: low mean offset and low variance => stable.
        gaze_stability = float(
            np.clip(1.0 - (gaze_offsets.mean() * 1.5 + gaze_offsets.std()), 0.0, 1.0)
        )

        # Blink rate from EAR threshold crossings, normalised against optimal band.
        blinks = int(np.sum(np.array([f.eye_aspect_ratio for f in feats]) < self.blink_ear_threshold))
        duration_min = max(len(feats) / (self.fps * 60.0), 1e-6)
        blink_rate = blinks / duration_min
        blink_rate_score = _blink_band_score(blink_rate)

        # Head-pose stability: low variance of pitch/yaw/roll => stable.
        pose = np.array([[f.pitch, f.yaw, f.roll] for f in feats])
        pose_var = float(pose.var(axis=0).mean())
        head_pose_stability = float(np.clip(1.0 - pose_var / 200.0, 0.0, 1.0))

        components = {
            "gaze_stability": gaze_stability,
            "blink_rate_score": blink_rate_score,
            "head_pose_stability": head_pose_stability,
        }
        stress = 100.0 * sum(components[k] * w for k, w in _STRESS_WEIGHTS.items())

        return StressResult(
            gaze_stability=round(gaze_stability, 4),
            blink_rate_score=round(blink_rate_score, 4),
            head_pose_stability=round(head_pose_stability, 4),
            frames_used=len(feats),
            frames_dropped=dropped,
            stress_score=float(np.clip(stress, 0.0, 100.0)),
            used_fallback=used_fallback,
            raw={"blink_rate_per_min": round(blink_rate, 3), "component_weights": _STRESS_WEIGHTS},
        )


# ----------------------------- helpers -------------------------------------

def _resize_numpy(frame: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
    ys = (np.linspace(0, h - 1, size)).astype(int)
    xs = (np.linspace(0, w - 1, size)).astype(int)
    return arr[np.ix_(ys, xs)]


def _eye_aspect_ratio(pts: np.ndarray, side: str) -> float:
    """Eye Aspect Ratio (EAR) using MediaPipe FaceMesh indices."""
    idx = (
        (33, 160, 158, 133, 153, 144) if side == "left" else (362, 385, 387, 263, 373, 380)
    )
    p = pts[list(idx)][:, :2]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = 2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6
    return float(vertical / horizontal)


def _head_pose_from_landmarks(pts: np.ndarray) -> tuple[float, float, float]:
    """Approximate pitch/yaw/roll (degrees) from a few stable landmarks."""
    nose = pts[1]
    left_eye, right_eye = pts[33], pts[263]
    yaw = float(np.degrees(np.arctan2(nose[0] - (left_eye[0] + right_eye[0]) / 2.0, 0.5)))
    pitch = float(np.degrees(np.arctan2(nose[1] - (left_eye[1] + right_eye[1]) / 2.0, 0.5)))
    roll = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0] + 1e-6)))
    return pitch, yaw, roll


def _blink_band_score(blink_rate: float) -> float:
    low, high = _BLINK_OPTIMAL
    if low <= blink_rate <= high:
        return 1.0
    if blink_rate < low:
        return float(np.clip(blink_rate / low, 0.0, 1.0))
    return float(np.clip(1.0 - (blink_rate - high) / high, 0.0, 1.0))
