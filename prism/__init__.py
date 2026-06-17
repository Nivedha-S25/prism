"""PRISM — Psychometric Recruitment Intelligence Scoring Model.

A modular, multi-modal candidate-evaluation backend composed of four
decoupled modules:

* ``behavioral``  — Audio (Whisper) + LLM situational-judgment scoring -> BhvS
* ``personality`` — BERT-based Big Five (OCEAN) profiling -> PAS
* ``vision``      — MediaPipe stress / engagement analysis -> StressS
* ``scoring``     — Weighted aggregation into the Integrated Candidate Score (ICS)
"""

from prism.schemas import (
    BehavioralResult,
    IntegratedScore,
    PersonalityResult,
    StressResult,
    Trait,
)

__version__ = "0.1.0"

__all__ = [
    "BehavioralResult",
    "PersonalityResult",
    "StressResult",
    "IntegratedScore",
    "Trait",
    "__version__",
]
