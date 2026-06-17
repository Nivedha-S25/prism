"""FastAPI application exposing the PRISM modules via a secure REST API.

Endpoints (all data endpoints require a bearer API key):

* ``GET  /health``               — liveness probe (open)
* ``POST /v1/behavioral``        — transcript -> BhvS
* ``POST /v1/personality``       — text -> OCEAN traits + PAS
* ``POST /v1/vision``            — uploaded frames -> StressS
* ``POST /v1/score``             — four sub-scores (+ optional role) -> ICS
* ``POST /v1/evaluate``          — pipeline over a single candidate payload -> ICS
"""

from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from prism.api.security import require_api_key
from prism.config import get_settings
from prism.modules.behavioral import BehavioralAnalyzer
from prism.modules.personality import PASWeights, PersonalityAnalyzer
from prism.modules.scoring import PRISMScoringEngine, ScoringWeights
from prism.modules.vision import StressAnalyzer

logger = logging.getLogger(__name__)


# --------------------------- request / response models ---------------------

class BehavioralRequest(BaseModel):
    transcript: str = Field(..., description="Transcribed candidate answer.")


class PersonalityRequest(BaseModel):
    text: str = Field(..., description="Candidate's typed response.")
    weights: dict[str, float] | None = Field(
        default=None, description="Optional PAS trait-weight overrides."
    )


class ScoreRequest(BaseModel):
    cog_score: float = Field(..., ge=0, le=100)
    bhv_score: float = Field(..., ge=0, le=100)
    stress_score: float = Field(..., ge=0, le=100)
    pas: float = Field(..., ge=0, le=100)
    role: str | None = Field(default=None, description="Optional role profile.")
    weights: dict[str, float] | None = Field(
        default=None, description="Explicit alpha/beta/gamma/delta overrides."
    )


class EvaluateRequest(BaseModel):
    transcript: str
    text: str
    cog_score: float = Field(..., ge=0, le=100)
    role: str | None = None


# --------------------------------- app -------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()
    settings.configure_logging()

    app = FastAPI(
        title="PRISM API",
        version="0.1.0",
        description="Psychometric Recruitment Intelligence Scoring Model backend.",
    )

    # Analyzers are stateless w.r.t. requests; instantiate once.
    behavioral = BehavioralAnalyzer(settings)
    personality = PersonalityAnalyzer(settings)
    stress = StressAnalyzer(settings)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": app.version}

    @app.post("/v1/behavioral")
    def score_behavioral(
        req: BehavioralRequest, _: str = Depends(require_api_key)
    ) -> dict:
        return behavioral.analyze(transcript=req.transcript).to_dict()

    @app.post("/v1/personality")
    def score_personality(
        req: PersonalityRequest, _: str = Depends(require_api_key)
    ) -> dict:
        analyzer = personality
        if req.weights:
            analyzer = PersonalityAnalyzer(settings, weights=PASWeights(**req.weights))
        return analyzer.analyze(req.text).to_dict()

    @app.post("/v1/vision")
    async def score_vision(
        frames: list[UploadFile] = File(...), _: str = Depends(require_api_key)
    ) -> dict:
        decoded: list[np.ndarray] = [
            img for f in frames if (img := _decode_image(await f.read())) is not None
        ]
        if not decoded:
            raise HTTPException(status_code=422, detail="No decodable image frames.")
        return stress.analyze(decoded).to_dict()

    @app.post("/v1/score")
    def score_ics(req: ScoreRequest, _: str = Depends(require_api_key)) -> dict:
        engine = _build_engine(req.role, req.weights)
        return engine.integrate_scores(
            cog_score=req.cog_score,
            bhv_score=req.bhv_score,
            stress_score=req.stress_score,
            pas=req.pas,
        ).to_dict()

    @app.post("/v1/evaluate")
    def evaluate(req: EvaluateRequest, _: str = Depends(require_api_key)) -> dict:
        bhv = behavioral.analyze(transcript=req.transcript)
        per = personality.analyze(req.text)
        # Vision requires frames; this endpoint scores the text/audio path and
        # assumes a neutral stress baseline when no frames are supplied.
        engine = _build_engine(req.role, None)
        ics = engine.integrate_scores(
            cog_score=req.cog_score,
            bhv_score=bhv.bhv_score,
            stress_score=50.0,
            pas=per.pas,
        )
        return {
            "behavioral": bhv.to_dict(),
            "personality": per.to_dict(),
            "integrated": ics.to_dict(),
        }

    return app


def _build_engine(role: str | None, weights: dict[str, float] | None) -> PRISMScoringEngine:
    if weights:
        return PRISMScoringEngine(ScoringWeights(**weights))
    if role:
        return PRISMScoringEngine.for_role(role)
    return PRISMScoringEngine()


def _decode_image(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes to a numpy array using cv2 or Pillow if present."""
    try:
        import cv2

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        pass
    try:
        from PIL import Image

        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        logger.warning("Could not decode an uploaded frame (no cv2/Pillow).")
        return None


# Module-level app for `uvicorn prism.api.app:app`.
app = create_app()
