"""Shared data structures passed between PRISM modules.

These are plain ``dataclasses`` (stdlib only) so the scoring pipeline has no
hard dependency on pydantic/torch. The FastAPI layer defines its own pydantic
request/response models and converts to/from these.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Trait(str, Enum):
    """Big Five (OCEAN) personality traits."""

    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


@dataclass
class BehavioralResult:
    """Output of Module 1 (Behavioral Analysis)."""

    transcript: str
    lexical_quality: float       # 0-100
    coherence: float             # 0-100
    situational_judgment: float  # 0-100
    bhv_score: float             # 0-100 (BhvS)
    used_fallback: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonalityResult:
    """Output of Module 2 (Personality Analysis)."""

    traits: dict[str, float]     # each trait normalised 0-1
    pas: float                   # 0-100 (Personality Alignment Score)
    used_fallback: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StressResult:
    """Output of Module 3 (Stress Management / Vision)."""

    gaze_stability: float        # 0-1
    blink_rate_score: float      # 0-1
    head_pose_stability: float   # 0-1
    frames_used: int
    frames_dropped: int
    stress_score: float          # 0-100 (StressS, higher = calmer/more engaged)
    used_fallback: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntegratedScore:
    """Output of Module 4 (PRISM Scoring Engine)."""

    cog_score: float             # CogS (0-100)
    bhv_score: float             # BhvS (0-100)
    stress_score: float          # StressS (0-100)
    pas: float                   # PAS (0-100)
    ics: float                   # Integrated Candidate Score (0-100)
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
