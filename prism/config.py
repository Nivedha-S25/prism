"""Centralised configuration for the PRISM backend.

Configuration is intentionally dependency-free (stdlib only) so that the
package can be imported and the demo executed even before the heavy ML
dependencies are installed. Values are read from the process environment,
optionally seeded from a local ``.env`` file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (no third-party dependency).

    Existing environment variables always take precedence so that real
    deployment secrets are never overridden by a checked-in file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return value if value != "" else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable, process-wide settings."""

    # --- Module 1: behavioral ---
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-flash"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base"))

    # --- Module 2: personality ---
    personality_model_name: str = field(
        default_factory=lambda: _env("PERSONALITY_MODEL_NAME", "bert-base-uncased")
    )
    personality_checkpoint: str = field(
        default_factory=lambda: _env("PERSONALITY_CHECKPOINT", "")
    )

    # --- Module 3: vision ---
    face_confidence_threshold: float = field(
        default_factory=lambda: _env_float("FACE_CONFIDENCE_THRESHOLD", 0.85)
    )
    frame_size: int = 128

    # --- API security ---
    api_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            k.strip() for k in _env("PRISM_API_KEYS", "dev-local-key").split(",") if k.strip()
        )
    )

    # --- Misc ---
    device: str = field(default_factory=lambda: _env("PRISM_DEVICE", "cpu"))
    log_level: str = field(default_factory=lambda: _env("PRISM_LOG_LEVEL", "INFO"))

    def configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the cached, process-wide :class:`Settings` instance."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS
