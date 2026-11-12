from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def env_value(name: str) -> str:
    """Read a Docker or python-dotenv value while removing optional outer quotes."""
    return os.getenv(name, "").strip().strip("'").strip('"')


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    embedding_model: str
    offline: bool


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        api_key=env_value("OPENAI_API_KEY"),
        base_url=env_value("OPENAI_BASE_URL") or env_value("OPENAI_BASE_UR"),
        model=env_value("OPENAI_MODEL"),
        embedding_model=env_value("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
        offline=env_value("PHARMA_OFFLINE").lower() in {"1", "true", "yes"},
    )

