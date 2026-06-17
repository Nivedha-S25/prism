"""Module 1 — Behavioral Analysis (Audio + LLM).

Pipeline:

1. :class:`WhisperTranscriber` converts a candidate audio recording to text
   using OpenAI Whisper.
2. :class:`LLMEvaluator` scores the transcript for lexical quality, coherence
   and situational-judgment alignment on a 0-100 scale using a prompt-engineered
   Google Gemini call.
3. :class:`BehavioralAnalyzer` orchestrates the two and produces the Behavioral
   Score (``BhvS``).

Every external dependency (Whisper, Gemini) is imported lazily and wrapped in a
deterministic heuristic fallback, so the module is importable and runnable even
when those libraries / API keys are not available.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass

from prism.config import Settings, get_settings
from prism.schemas import BehavioralResult

logger = logging.getLogger(__name__)

# Sub-score weights used to combine the LLM dimensions into BhvS.
_BHV_WEIGHTS = {
    "lexical_quality": 0.30,
    "coherence": 0.30,
    "situational_judgment": 0.40,
}


class WhisperTranscriber:
    """Thin wrapper around OpenAI Whisper for speech-to-text."""

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model = None  # lazily loaded

    @property
    def available(self) -> bool:
        try:
            import whisper  # noqa: F401
        except Exception:  # pragma: no cover - depends on env
            return False
        return True

    def _ensure_model(self) -> None:
        if self._model is None:
            import whisper  # local import keeps the dep optional

            logger.info("Loading Whisper model '%s'", self.model_name)
            self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text.

        Raises:
            RuntimeError: if Whisper is not installed. Callers that want a
                graceful degradation should catch this.
        """
        if not self.available:
            raise RuntimeError("Whisper is not installed; cannot transcribe audio.")
        self._ensure_model()
        assert self._model is not None
        result = self._model.transcribe(audio_path)
        return str(result.get("text", "")).strip()


class LLMEvaluator:
    """Prompt-engineered LLM wrapper that scores transcribed answers.

    Uses Google Gemini when an API key is configured; otherwise falls back to a
    deterministic linguistic heuristic so the pipeline always returns a score.
    """

    _PROMPT_TEMPLATE = (
        "You are an expert recruitment assessor. Evaluate the following spoken "
        "interview answer (already transcribed). Score it on three dimensions, "
        "each an integer from 0 to 100:\n"
        "  - lexical_quality: richness, precision and professionalism of vocabulary\n"
        "  - coherence: logical flow and structural clarity of the response\n"
        "  - situational_judgment: soundness of the judgment / decision described\n\n"
        "Respond with ONLY a compact JSON object of the form: "
        '{{"lexical_quality": int, "coherence": int, "situational_judgment": int}}.\n\n'
        "Candidate answer:\n\"\"\"\n{transcript}\n\"\"\""
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def available(self) -> bool:
        if not self.settings.gemini_api_key:
            return False
        try:
            import google.generativeai  # noqa: F401
        except Exception:  # pragma: no cover - depends on env
            return False
        return True

    def _ensure_client(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            self._client = genai.GenerativeModel(self.settings.gemini_model)
        return self._client

    def score(self, transcript: str) -> tuple[dict[str, float], bool]:
        """Return ``(dimension_scores, used_fallback)``."""
        if self.available:
            try:
                return self._score_with_gemini(transcript), False
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Gemini scoring failed (%s); using heuristic.", exc)
        return self._score_heuristic(transcript), True

    def _score_with_gemini(self, transcript: str) -> dict[str, float]:
        client = self._ensure_client()
        prompt = self._PROMPT_TEMPLATE.format(transcript=transcript)
        response = client.generate_content(prompt)
        payload = _extract_json(response.text)
        return {
            "lexical_quality": _clip(float(payload["lexical_quality"])),
            "coherence": _clip(float(payload["coherence"])),
            "situational_judgment": _clip(float(payload["situational_judgment"])),
        }

    @staticmethod
    def _score_heuristic(transcript: str) -> dict[str, float]:
        """Deterministic linguistic heuristic used when no LLM is available.

        Approximates the three dimensions from simple, explainable text features
        so the pipeline degrades gracefully rather than failing.
        """
        words = re.findall(r"[A-Za-z']+", transcript.lower())
        n_words = len(words)
        if n_words == 0:
            return {"lexical_quality": 0.0, "coherence": 0.0, "situational_judgment": 0.0}

        unique_ratio = len(set(words)) / n_words
        sentences = [s for s in re.split(r"[.!?]+", transcript) if s.strip()]
        avg_sentence_len = n_words / max(len(sentences), 1)

        # Lexical quality: vocabulary diversity + adequate length.
        lexical = 100.0 * (0.6 * unique_ratio + 0.4 * min(n_words / 80.0, 1.0))

        # Coherence: rewards moderate sentence length (8-22 words) & connectors.
        connectors = {"because", "therefore", "however", "so", "then", "thus", "since"}
        connector_hits = sum(1 for w in words if w in connectors)
        len_penalty = math.exp(-((avg_sentence_len - 15.0) ** 2) / (2 * 8.0 ** 2))
        coherence = 100.0 * (0.7 * len_penalty + 0.3 * min(connector_hits / 4.0, 1.0))

        # Situational judgment: presence of action/decision vocabulary.
        sj_terms = {
            "decided", "prioritized", "resolved", "communicated", "team",
            "stakeholder", "deadline", "risk", "plan", "outcome", "responsibility",
        }
        sj_hits = sum(1 for w in words if w in sj_terms)
        situational = 100.0 * (0.5 * min(sj_hits / 5.0, 1.0) + 0.5 * unique_ratio)

        return {
            "lexical_quality": _clip(lexical),
            "coherence": _clip(coherence),
            "situational_judgment": _clip(situational),
        }


@dataclass
class BehavioralAnalyzer:
    """Module 1 orchestrator producing the Behavioral Score (BhvS)."""

    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        self.transcriber = WhisperTranscriber(self.settings.whisper_model)
        self.evaluator = LLMEvaluator(self.settings)

    def analyze(
        self,
        *,
        audio_path: str | None = None,
        transcript: str | None = None,
    ) -> BehavioralResult:
        """Run the full behavioral pipeline.

        Either ``audio_path`` or a pre-computed ``transcript`` must be provided.
        """
        if transcript is None and audio_path is None:
            raise ValueError("Provide either audio_path or transcript.")

        used_fallback = False
        if transcript is None:
            try:
                transcript = self.transcriber.transcribe(audio_path)  # type: ignore[arg-type]
            except RuntimeError as exc:
                logger.warning("Transcription unavailable (%s); expected a transcript.", exc)
                transcript = ""
                used_fallback = True

        dims, llm_fallback = self.evaluator.score(transcript)
        bhv = sum(dims[k] * w for k, w in _BHV_WEIGHTS.items())

        return BehavioralResult(
            transcript=transcript,
            lexical_quality=dims["lexical_quality"],
            coherence=dims["coherence"],
            situational_judgment=dims["situational_judgment"],
            bhv_score=_clip(bhv),
            used_fallback=used_fallback or llm_fallback,
            raw={"dimension_weights": _BHV_WEIGHTS},
        )


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))


def _extract_json(text: str) -> dict[str, float]:
    """Extract the first JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return json.loads(match.group(0))
