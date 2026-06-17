"""PRISM end-to-end demonstration.

Runs the three analysis modules **concurrently** on a single candidate's data,
then feeds their outputs into the scoring engine to produce the Integrated
Candidate Score (ICS).

Run:
    python main.py
    python main.py --role software_engineer --cog-score 82

The demo uses bundled sample data and degrades gracefully: if the heavy ML
dependencies (Whisper, Gemini, BERT, MediaPipe) are not installed, each module
falls back to a deterministic heuristic so the full pipeline still executes.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from prism.config import get_settings
from prism.modules.behavioral import BehavioralAnalyzer
from prism.modules.personality import PersonalityAnalyzer
from prism.modules.scoring import PRISMScoringEngine
from prism.modules.vision import StressAnalyzer

logger = logging.getLogger("prism.demo")

SAMPLE_TRANSCRIPT = (
    "When the project deadline was suddenly moved up, I prioritized the most "
    "critical features and communicated the trade-offs to my stakeholders. "
    "Because the team was under pressure, I organized a short daily check-in so "
    "we could resolve blockers quickly, and we delivered the core scope on time."
)

SAMPLE_TEXT = (
    "I really enjoy exploring new ideas and learning unfamiliar tools. I like to "
    "plan my work carefully and keep things organized so I hit my goals. I get "
    "energy from collaborating with people, and I try to support and listen to my "
    "teammates whenever they need help."
)


def make_sample_frames(n: int = 30, seed: int = 7) -> list[np.ndarray]:
    """Generate deterministic synthetic webcam frames for the demo."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8) for _ in range(n)]


def run_pipeline(cog_score: float, role: str | None) -> dict:
    settings = get_settings()
    settings.configure_logging()

    behavioral = BehavioralAnalyzer(settings)
    personality = PersonalityAnalyzer(settings)
    stress = StressAnalyzer(settings)
    frames = make_sample_frames()

    # --- run the three modules concurrently ---
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_bhv = pool.submit(behavioral.analyze, transcript=SAMPLE_TRANSCRIPT)
        f_per = pool.submit(personality.analyze, SAMPLE_TEXT)
        f_str = pool.submit(stress.analyze, frames)
        bhv_result = f_bhv.result()
        per_result = f_per.result()
        str_result = f_str.result()

    engine = PRISMScoringEngine.for_role(role) if role else PRISMScoringEngine()
    ics = engine.integrate(
        cog_score=cog_score,
        behavioral=bhv_result,
        stress=str_result,
        personality=per_result,
    )

    return {
        "behavioral": bhv_result.to_dict(),
        "personality": per_result.to_dict(),
        "stress": str_result.to_dict(),
        "integrated": ics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM end-to-end demo")
    parser.add_argument("--cog-score", type=float, default=78.0,
                        help="External cognitive score (0-100).")
    parser.add_argument("--role", type=str, default=None,
                        help="Optional role profile, e.g. software_engineer.")
    args = parser.parse_args()

    result = run_pipeline(cog_score=args.cog_score, role=args.role)

    print("\n=== PRISM Candidate Evaluation ===")
    print(json.dumps(result, indent=2))
    integrated = result["integrated"]
    print(f"\nIntegrated Candidate Score (ICS): {integrated['ics']:.2f} / 100")
    print(f"Weights: {integrated['weights']}")


if __name__ == "__main__":
    main()
