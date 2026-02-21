from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    openai_api_key: str | None
    openai_base_url: str
    anthropic_api_key: str | None
    local_llm_base_url: str
    local_llm_api_key: str | None


def _load_dotenv_file(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_llm_config() -> LLMConfig:
    _load_dotenv_file()
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        local_llm_base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
        local_llm_api_key=os.getenv("LOCAL_LLM_API_KEY"),
    )
