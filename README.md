# PRISM — Psychometric Recruitment Intelligence Scoring Model

A modular, multi-modal AI recruitment backend. PRISM evaluates a candidate across
three concurrent modules and fuses their outputs (plus an external cognitive
score) into a single **Integrated Candidate Score (ICS)**.

| Module | Signal | Tech | Output |
| --- | --- | --- | --- |
| 1. Behavioral | Spoken answers | OpenAI Whisper + LLM (Google Gemini) | `BhvS` |
| 2. Personality | Typed answers | `bert-base-uncased` (Big Five / OCEAN) | `PAS` |
| 3. Stress | Webcam frames | MediaPipe + OpenCV | `StressS` |
| 4. Scoring | Aggregation | Weighted linear model | `ICS` |

```
ICS = α·CogS + β·BhvS + γ·StressS + δ·PAS
defaults: α=0.35, β=0.25, γ=0.20, δ=0.20   (sum = 1.0)
```

## Project layout

```
prism/
├── main.py                     # concurrent end-to-end demo
├── requirements.txt
├── .env.example
├── prism/
│   ├── config.py               # dependency-free settings + .env loader
│   ├── schemas.py              # shared dataclasses (module I/O)
│   ├── modules/
│   │   ├── behavioral.py       # Module 1: Whisper STT + LLM scoring  -> BhvS
│   │   ├── personality.py      # Module 2: BERT OCEAN + training loop -> PAS
│   │   ├── vision.py           # Module 3: MediaPipe stress analysis  -> StressS
│   │   └── scoring.py          # Module 4: PRISM aggregator           -> ICS
│   ├── api/
│   │   ├── app.py              # FastAPI app (secure REST)
│   │   └── security.py         # bearer-token auth
│   └── evaluation/
│       └── metrics.py          # precision / recall / F1 / accuracy / regression
└── tests/
```

## Graceful degradation

Every external dependency (Whisper, Gemini, PyTorch/Transformers, MediaPipe) is
imported **lazily** and wrapped in a deterministic heuristic fallback. This means
you can import the package, run `main.py`, and exercise the REST API even before
the heavy ML stack is installed — each module sets `used_fallback=True` when it
takes the heuristic path. Install the full `requirements.txt` to use the real models.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full stack
# ...or a lightweight subset to just run the demo:
pip install numpy fastapi "uvicorn[standard]" pydantic scikit-learn

cp .env.example .env                      # add GEMINI_API_KEY etc.
python main.py --role software_engineer --cog-score 82
```

## REST API

```bash
uvicorn prism.api.app:app --reload
```

All `/v1/*` endpoints require a bearer token from `PRISM_API_KEYS` (default
`dev-local-key`). Interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/v1/personality \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "I plan carefully and enjoy helping my team."}'
```

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | liveness |
| POST | `/v1/behavioral` | `{transcript}` | BhvS + sub-scores |
| POST | `/v1/personality` | `{text, weights?}` | OCEAN traits + PAS |
| POST | `/v1/vision` | multipart image frames | StressS |
| POST | `/v1/score` | `{cog_score, bhv_score, stress_score, pas, role?, weights?}` | ICS |
| POST | `/v1/evaluate` | `{transcript, text, cog_score, role?}` | full breakdown |

## Personality model training

`prism.modules.personality.PersonalityTrainer` fine-tunes `bert-base-uncased`
with a 5-unit regression head: 5 epochs, AdamW, `lr=2e-5`, linear warm-up over
the first 10% of steps. Point `PERSONALITY_CHECKPOINT` at the saved weights to
enable real inference.

## Evaluation

```python
from prism.evaluation import evaluate_classification, evaluate_regression
report = evaluate_classification(y_true, y_pred)   # accuracy / precision / recall / f1
```

## Tests

```bash
pytest -q
```
