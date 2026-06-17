"""Module 4 — PRISM Scoring Engine (Integration).

Combines the per-module outputs into a single Integrated Candidate Score (ICS):

    ICS = alpha * CogS + beta * BhvS + gamma * StressS + delta * PAS

Default role-agnostic weights sum to 1.0:
    alpha = 0.35  (Cognitive Score)
    beta  = 0.25  (Behavioral Score)
    gamma = 0.20  (Stress Score)
    delta = 0.20  (Personality Alignment Score)

HR administrators can override the weights per job role.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from prism.schemas import BehavioralResult, IntegratedScore, PersonalityResult, StressResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the ICS linear combination.

    ``CogS`` (cognitive) is sourced externally (e.g. an aptitude test); the
    other three come from PRISM modules 1-3 / 2.
    """

    alpha: float = 0.35  # Cognitive Score (CogS)
    beta: float = 0.25   # Behavioral Score (BhvS)
    gamma: float = 0.20  # Stress Score (StressS)
    delta: float = 0.20  # Personality Alignment Score (PAS)

    def normalized(self) -> ScoringWeights:
        """Return weights rescaled to sum to 1.0 (no-op if already normalised)."""
        total = self.alpha + self.beta + self.gamma + self.delta
        if total <= 0:
            raise ValueError("Scoring weights must sum to a positive value.")
        return ScoringWeights(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
            delta=self.delta / total,
        )

    def as_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta}


# Example HR-configurable, role-specific weight profiles. These intentionally
# keep a sum of 1.0 so they can be used as drop-in overrides for the defaults.
ROLE_WEIGHT_PROFILES: dict[str, ScoringWeights] = {
    "default": ScoringWeights(),
    "software_engineer": ScoringWeights(alpha=0.45, beta=0.20, gamma=0.15, delta=0.20),
    "sales_representative": ScoringWeights(alpha=0.20, beta=0.40, gamma=0.15, delta=0.25),
    "customer_support": ScoringWeights(alpha=0.20, beta=0.30, gamma=0.25, delta=0.25),
    "people_manager": ScoringWeights(alpha=0.25, beta=0.30, gamma=0.15, delta=0.30),
}


class PRISMScoringEngine:
    """Aggregates module outputs into the Integrated Candidate Score (ICS)."""

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = (weights or ScoringWeights()).normalized()

    @classmethod
    def for_role(cls, role: str) -> PRISMScoringEngine:
        """Create an engine using a named role profile (falls back to default)."""
        profile = ROLE_WEIGHT_PROFILES.get(role.lower())
        if profile is None:
            logger.warning("Unknown role '%s'; using default weights.", role)
            profile = ROLE_WEIGHT_PROFILES["default"]
        return cls(profile)

    def integrate(
        self,
        *,
        cog_score: float,
        behavioral: BehavioralResult,
        stress: StressResult,
        personality: PersonalityResult,
    ) -> IntegratedScore:
        """Compute the ICS from a cognitive score plus the three module results."""
        return self.integrate_scores(
            cog_score=cog_score,
            bhv_score=behavioral.bhv_score,
            stress_score=stress.stress_score,
            pas=personality.pas,
        )

    def integrate_scores(
        self,
        *,
        cog_score: float,
        bhv_score: float,
        stress_score: float,
        pas: float,
    ) -> IntegratedScore:
        """Compute the ICS directly from the four scalar sub-scores."""
        w = self.weights
        cog = _clip(cog_score)
        bhv = _clip(bhv_score)
        stress = _clip(stress_score)
        pas_c = _clip(pas)
        ics = w.alpha * cog + w.beta * bhv + w.gamma * stress + w.delta * pas_c
        return IntegratedScore(
            cog_score=cog,
            bhv_score=bhv,
            stress_score=stress,
            pas=pas_c,
            ics=round(_clip(ics), 4),
            weights=w.as_dict(),
        )


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))
