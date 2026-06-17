"""Module 2 — Personality Analysis (NLP / BERT).

Profiles a candidate against the Big Five (OCEAN) traits from their typed text
responses.

Components:

* :class:`TextPreprocessor` — WordPiece tokenization (BERT tokenizer) plus
  stop-word filtering.
* :class:`BigFiveRegressor` — ``bert-base-uncased`` with a 5-unit regression
  head (multi-label, sigmoid) mapping text -> OCEAN traits in ``[0, 1]``.
* :class:`PersonalityTrainer` — fine-tuning loop (5 epochs, AdamW, lr=2e-5,
  linear warm-up over the first 10% of steps).
* :class:`PersonalityAnalyzer` — inference + Personality Alignment Score (PAS)
  with configurable trait weights.

torch / transformers are imported lazily so the module degrades to a
deterministic lexical heuristic when they are unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from prism.config import Settings, get_settings
from prism.schemas import PersonalityResult, Trait

logger = logging.getLogger(__name__)

TRAIT_ORDER: tuple[Trait, ...] = (
    Trait.OPENNESS,
    Trait.CONSCIENTIOUSNESS,
    Trait.EXTRAVERSION,
    Trait.AGREEABLENESS,
    Trait.NEUROTICISM,
)

# A small, dependency-free English stop-word list for preprocessing.
STOP_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else of to in on at for with without from by
    is are was were be been being am do does did doing have has had having
    i you he she it we they me him her us them my your his its our their this
    that these those as so than too very can will just dont don't not no nor
    """.split()
)


@dataclass
class PASWeights:
    """Configurable weights for the Personality Alignment Score formula."""

    w_openness: float = 1.0
    w_conscientiousness: float = 1.0
    w_extraversion: float = 1.0
    w_agreeableness: float = 1.0
    w_neuroticism: float = 1.0

    @property
    def total(self) -> float:
        return (
            self.w_openness
            + self.w_conscientiousness
            + self.w_extraversion
            + self.w_agreeableness
            + self.w_neuroticism
        )


class TextPreprocessor:
    """WordPiece tokenization + stop-word filtering.

    When the HuggingFace tokenizer is unavailable, ``encode`` returns ``None``
    and only the lightweight cleaning utilities are used (heuristic path).
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 256) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
        except Exception:  # pragma: no cover - depends on env
            return False
        return True

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            logger.info("Loading tokenizer '%s'", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @staticmethod
    def clean(text: str) -> str:
        """Lower-case, strip non-alpha noise and remove stop-words."""
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        kept = [t for t in tokens if t not in STOP_WORDS]
        return " ".join(kept)

    def encode(self, text: str):
        """Return a HuggingFace ``BatchEncoding`` of WordPiece ids, or ``None``."""
        if not self.available:
            return None
        tokenizer = self._ensure_tokenizer()
        return tokenizer(
            self.clean(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )


def build_regressor(model_name: str = "bert-base-uncased", n_traits: int = 5):
    """Factory for the BERT regression model (returns ``None`` if torch missing).

    Defined as a factory so importing this module never requires torch. The
    actual ``nn.Module`` subclass is created at call time.
    """
    try:
        import torch
        from torch import nn
        from transformers import AutoModel
    except Exception:  # pragma: no cover - depends on env
        logger.warning("torch/transformers unavailable; regressor cannot be built.")
        return None

    class BigFiveRegressor(nn.Module):
        """``bert-base-uncased`` encoder + linear regression head (sigmoid)."""

        def __init__(self, base_model: str, num_traits: int) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_model)
            hidden = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.head = nn.Linear(hidden, num_traits)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled = outputs.last_hidden_state[:, 0]  # [CLS] representation
            return torch.sigmoid(self.head(self.dropout(pooled)))

    return BigFiveRegressor(model_name, n_traits)


@dataclass
class PersonalityTrainer:
    """Fine-tuning loop for :func:`build_regressor`.

    Hyper-parameters follow the spec: 5 epochs, AdamW, lr=2e-5, linear warm-up
    over the first 10% of total training steps.
    """

    model_name: str = "bert-base-uncased"
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    device: str = "cpu"

    def train(self, dataloader) -> object:
        """Run the fine-tuning loop and return the trained model.

        ``dataloader`` must yield batches with keys ``input_ids``,
        ``attention_mask`` (optionally ``token_type_ids``) and ``labels``
        (a float tensor of shape ``[batch, 5]`` in ``[0, 1]``).
        """
        import torch
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        model = build_regressor(self.model_name)
        if model is None:
            raise RuntimeError("Cannot train: torch/transformers not installed.")
        model.to(self.device)
        model.train()

        total_steps = max(len(dataloader) * self.epochs, 1)
        warmup_steps = int(total_steps * self.warmup_ratio)
        optimizer = AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        loss_fn = torch.nn.MSELoss()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                preds = model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    token_type_ids=(
                        batch["token_type_ids"].to(self.device)
                        if "token_type_ids" in batch
                        else None
                    ),
                )
                loss = loss_fn(preds, batch["labels"].to(self.device).float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())
            logger.info(
                "Epoch %d/%d - mean MSE: %.4f",
                epoch + 1,
                self.epochs,
                epoch_loss / max(len(dataloader), 1),
            )
        model.eval()
        return model


@dataclass
class PersonalityAnalyzer:
    """Module 2 orchestrator producing OCEAN traits and the PAS."""

    settings: Settings | None = None
    weights: PASWeights = field(default_factory=PASWeights)

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        self.preprocessor = TextPreprocessor(self.settings.personality_model_name)
        self._model = None

    def _load_model(self):
        assert self.settings is not None
        if self._model is None and self.preprocessor.available:
            self._model = build_regressor(self.settings.personality_model_name)
            if self._model is not None and self.settings.personality_checkpoint:
                import torch

                state = torch.load(self.settings.personality_checkpoint, map_location="cpu")
                self._model.load_state_dict(state)
                self._model.eval()
        return self._model

    def predict_traits(self, text: str) -> tuple[dict[str, float], bool]:
        """Return ``(trait_scores_0_1, used_fallback)``."""
        assert self.settings is not None
        model = self._load_model()
        encoding = self.preprocessor.encode(text)
        if model is not None and encoding is not None and self.settings.personality_checkpoint:
            try:
                import torch

                with torch.no_grad():
                    preds = model(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        token_type_ids=encoding.get("token_type_ids", None),
                    )[0]
                traits = {t.value: float(preds[i]) for i, t in enumerate(TRAIT_ORDER)}
                return traits, False
            except Exception as exc:  # pragma: no cover - model dependent
                logger.warning("BERT inference failed (%s); using heuristic.", exc)
        return self._heuristic_traits(text), True

    @staticmethod
    def _heuristic_traits(text: str) -> dict[str, float]:
        """Lexical lexicon heuristic for OCEAN when no fine-tuned model exists."""
        words = re.findall(r"[A-Za-z']+", text.lower())
        n = max(len(words), 1)
        lexicons = {
            Trait.OPENNESS: {"creative", "idea", "curious", "imagine", "novel", "explore", "learn"},
            Trait.CONSCIENTIOUSNESS: {"plan", "organized", "deadline", "detail", "responsible", "goal"},
            Trait.EXTRAVERSION: {"team", "people", "social", "lead", "energetic", "talk", "collaborate"},
            Trait.AGREEABLENESS: {"help", "support", "kind", "empathy", "cooperate", "trust", "listen"},
            Trait.NEUROTICISM: {"stress", "anxious", "worried", "nervous", "overwhelmed", "fear", "panic"},
        }
        traits: dict[str, float] = {}
        for trait, lex in lexicons.items():
            hits = sum(1 for w in words if w in lex)
            # Smooth, bounded mapping from hit density to a [0,1] score.
            traits[trait.value] = round(min(0.1 + hits / (n * 0.05 + 1.0), 1.0), 4)
        return traits

    def compute_pas(self, traits: dict[str, float]) -> float:
        """Weighted Personality Alignment Score on a 0-100 scale.

        Neuroticism is a negative indicator, so its *inverse* ``(1 - N)`` is
        rewarded. Keeping ``(1 - N)`` (rather than subtracting ``N``) means the
        neuroticism weight stays meaningful in ``w.total`` and the score can
        span the full 0-100 range: all traits ideal (O=C=E=A=1, N=0) -> 100.
        """
        w = self.weights
        numerator = (
            w.w_openness * traits[Trait.OPENNESS.value]
            + w.w_conscientiousness * traits[Trait.CONSCIENTIOUSNESS.value]
            + w.w_extraversion * traits[Trait.EXTRAVERSION.value]
            + w.w_agreeableness * traits[Trait.AGREEABLENESS.value]
            + w.w_neuroticism * (1.0 - traits[Trait.NEUROTICISM.value])
        )
        pas = (numerator / w.total) * 100.0 if w.total else 0.0
        return float(max(0.0, min(100.0, pas)))

    def analyze(self, text: str) -> PersonalityResult:
        traits, used_fallback = self.predict_traits(text)
        pas = self.compute_pas(traits)
        return PersonalityResult(
            traits=traits,
            pas=pas,
            used_fallback=used_fallback,
            raw={"weights": self.weights.__dict__},
        )
