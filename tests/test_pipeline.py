"""Smoke + unit tests for the PRISM pipeline (fallback paths)."""

from __future__ import annotations

import numpy as np

from prism.evaluation.metrics import evaluate_classification, evaluate_regression
from prism.modules.behavioral import BehavioralAnalyzer
from prism.modules.personality import PASWeights, PersonalityAnalyzer
from prism.modules.scoring import PRISMScoringEngine, ScoringWeights
from prism.modules.vision import StressAnalyzer


def test_behavioral_score_range() -> None:
    result = BehavioralAnalyzer().analyze(transcript="I prioritized the team and resolved the risk.")
    assert 0.0 <= result.bhv_score <= 100.0
    assert result.transcript


def test_personality_traits_and_pas() -> None:
    result = PersonalityAnalyzer().analyze("I plan carefully, help my teammates and explore new ideas.")
    assert set(result.traits) == {
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
    }
    assert all(0.0 <= v <= 1.0 for v in result.traits.values())
    assert 0.0 <= result.pas <= 100.0


def test_pas_weights_configurable() -> None:
    analyzer = PersonalityAnalyzer(weights=PASWeights(w_neuroticism=3.0))
    result = analyzer.analyze("I feel anxious and worried and stressed and overwhelmed.")
    assert 0.0 <= result.pas <= 100.0


def test_pas_spans_full_range_and_rewards_low_neuroticism() -> None:
    analyzer = PersonalityAnalyzer()
    ideal = {"openness": 1.0, "conscientiousness": 1.0, "extraversion": 1.0,
             "agreeableness": 1.0, "neuroticism": 0.0}
    worst = {"openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0,
             "agreeableness": 0.0, "neuroticism": 1.0}
    # Default weights all 1 -> ideal OCEAN reaches the full 100, worst reaches 0.
    assert analyzer.compute_pas(ideal) == 100.0
    assert analyzer.compute_pas(worst) == 0.0
    # High neuroticism must lower PAS relative to low neuroticism (all else equal).
    base = {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
            "agreeableness": 0.5, "neuroticism": 0.0}
    high_n = {**base, "neuroticism": 1.0}
    assert analyzer.compute_pas(high_n) < analyzer.compute_pas(base)


def test_vision_score_range() -> None:
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(15)]
    result = StressAnalyzer().analyze(frames)
    assert 0.0 <= result.stress_score <= 100.0
    assert result.frames_used + result.frames_dropped == 15


def test_scoring_weights_normalised() -> None:
    engine = PRISMScoringEngine(ScoringWeights(alpha=1, beta=1, gamma=1, delta=1))
    ics = engine.integrate_scores(cog_score=100, bhv_score=100, stress_score=100, pas=100)
    assert abs(ics.ics - 100.0) < 1e-6
    assert abs(sum(ics.weights.values()) - 1.0) < 1e-6


def test_default_weights_sum_to_one() -> None:
    w = ScoringWeights()
    assert abs((w.alpha + w.beta + w.gamma + w.delta) - 1.0) < 1e-9


def test_metrics_fallback() -> None:
    cls = evaluate_classification([1, 0, 1, 1], [1, 0, 0, 1])
    assert 0.0 <= cls.f1 <= 1.0
    reg = evaluate_regression([0.5, 0.2], [0.4, 0.25])
    assert reg.mae >= 0.0
