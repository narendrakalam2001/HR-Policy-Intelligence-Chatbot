"""
config.py

Centralized configuration for the HR Policy Intelligence Chatbot.

All tunables (paths, chunking params, model names, API keys) live here so
that no other module hardcodes a magic string. Values are loaded from
environment variables (via a `.env` file in local dev, or real environment
variables in Hugging Face Spaces / Docker) with sane defaults.

Usage:
    from config import settings
    settings.GEMINI_API_KEY
    settings.CHROMA_PERSIST_DIR
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load .env file if present (no-op in environments where env vars are
# injected directly, e.g. HF Spaces secrets or Docker --env-file).
load_dotenv()

# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
POLICIES_DIR: Final[Path] = DATA_DIR / "policies"
CHROMA_DIR: Final[Path] = BASE_DIR / "chroma_store"
LOG_DIR: Final[Path] = BASE_DIR / "logs"


def _get_bool(env_var: str, default: bool) -> bool:
    """Parse a boolean-ish environment variable safely."""
    val = os.getenv(env_var)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(env_var: str, default: int) -> int:
    """Parse an integer environment variable safely, falling back on error."""
    val = os.getenv(env_var)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid int for %s=%r, using default %s", env_var, val, default
        )
        return default


def _get_float(env_var: str, default: float) -> float:
    """Parse a float environment variable safely, falling back on error."""
    val = os.getenv(env_var)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid float for %s=%r, using default %s", env_var, val, default
        )
        return default


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings, populated from environment variables.

    Frozen so that once loaded at import time, no module can accidentally
    mutate shared config mid-run (a common source of hard-to-debug state
    bugs in notebook-turned-production code).
    """

    # --- LLM (Google Gemini) -------------------------------------------- #
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    )
    GEMINI_TEMPERATURE: float = field(
        default_factory=lambda: _get_float("GEMINI_TEMPERATURE", 0.2)
    )
    GEMINI_MAX_OUTPUT_TOKENS: int = field(
        default_factory=lambda: _get_int("GEMINI_MAX_OUTPUT_TOKENS", 1024)
    )
    GEMINI_REQUEST_TIMEOUT_SECONDS: int = field(
        default_factory=lambda: _get_int("GEMINI_REQUEST_TIMEOUT_SECONDS", 20)
    )

    # --- Embeddings -------------------------------------------------------- #
    EMBEDDING_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    EMBEDDING_DEVICE: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))

    # --- Chunking ------------------------------------------------------- #
    CHUNK_SIZE: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))

    # --- Vector store (ChromaDB) ----------------------------------------- #
    CHROMA_PERSIST_DIR: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
    )
    CHROMA_COLLECTION_NAME: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "hr_policy_docs")
    )

    # --- Retrieval -------------------------------------------------------- #
    RETRIEVAL_TOP_K: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 4))
    RETRIEVAL_SCORE_THRESHOLD: float = field(
        default_factory=lambda: _get_float("RETRIEVAL_SCORE_THRESHOLD", 0.35)
    )

    # --- Conversation ------------------------------------------------------ #
    CONVERSATION_MEMORY_TURNS: int = field(
        default_factory=lambda: _get_int("CONVERSATION_MEMORY_TURNS", 5)
    )

    # --- Paths ------------------------------------------------------------ #
    POLICIES_DIR: str = field(default_factory=lambda: os.getenv("POLICIES_DIR", str(POLICIES_DIR)))
    LOG_DIR: str = field(default_factory=lambda: os.getenv("LOG_DIR", str(LOG_DIR)))

    # --- Misc --------------------------------------------------------------- #
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    DEBUG: bool = field(default_factory=lambda: _get_bool("DEBUG", False))

    def validate(self) -> None:
        """
        Validate required settings and raise a clear error if misconfigured.

        Called explicitly (not at import time) so that unit tests / linting
        can import this module without a live API key present.
        """
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
                "Google Gemini API key, or set it as an environment variable / "
                "Hugging Face Space secret."
            )
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be smaller than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})."
            )


def configure_logging(settings_obj: "Settings") -> None:
    """Configure root logging once, writing to console + a rotating file."""
    os.makedirs(settings_obj.LOG_DIR, exist_ok=True)
    log_file = Path(settings_obj.LOG_DIR) / "app.log"

    logging.basicConfig(
        level=getattr(logging, settings_obj.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# Singleton settings instance imported everywhere else.
settings = Settings()
configure_logging(settings)

# Ensure required directories exist at import time (cheap, idempotent).
os.makedirs(settings.POLICIES_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)