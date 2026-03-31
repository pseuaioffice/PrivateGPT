"""
Shared runtime configuration helpers for MyAIAssistant packaging.

This module keeps the packaged backend .env deterministic:
- build.py writes a clean default .env and .env.example
- installer.iss patches selected keys during setup
- launcher.py reads the same keys at runtime
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping


DEFAULT_ENV_VALUES = OrderedDict(
    [
        ("HUGGINGFACE_TOKEN", ""),
        ("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/hf-inference/v1"),
        ("OPENAI_API_KEY", ""),
        ("MODEL_PROVIDER", "ollama"),
        ("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        ("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        ("CHAT_MODEL_LOCAL", "qwen2.5:0.5b"),
        ("EMBEDDING_MODEL_LOCAL", "nomic-embed-text"),
        ("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ("BACKEND_HOST", "127.0.0.1"),
        ("BACKEND_PORT", "8000"),
        ("FRONTEND_HOST", "127.0.0.1"),
        ("FRONTEND_PORT", "5000"),
        ("AUTO_OPEN_BROWSER", "true"),
        ("START_OLLAMA_WITH_LAUNCHER", "true"),
        ("DEBUG", "false"),
        ("CHROMA_PERSIST_DIR", "./chroma_db"),
        ("CHROMA_COLLECTION_NAME", "rag_documents"),
        ("DOCUMENTS_DIR", "./documents"),
        ("CHUNK_SIZE", "1000"),
        ("CHUNK_OVERLAP", "200"),
        ("DATABASE_URL", ""),
    ]
)


GROUPED_COMMENTS = OrderedDict(
    [
        (
            "Cloud providers",
            [
                "HUGGINGFACE_TOKEN",
                "HUGGINGFACE_BASE_URL",
                "OPENAI_API_KEY",
                "MODEL_PROVIDER",
                "CHAT_MODEL",
                "EMBEDDING_MODEL",
            ],
        ),
        (
            "Local Ollama defaults",
            [
                "CHAT_MODEL_LOCAL",
                "EMBEDDING_MODEL_LOCAL",
                "OLLAMA_BASE_URL",
                "START_OLLAMA_WITH_LAUNCHER",
            ],
        ),
        (
            "Launcher networking",
            [
                "BACKEND_HOST",
                "BACKEND_PORT",
                "FRONTEND_HOST",
                "FRONTEND_PORT",
                "AUTO_OPEN_BROWSER",
            ],
        ),
        (
            "Backend runtime",
            [
                "DEBUG",
                "CHROMA_PERSIST_DIR",
                "CHROMA_COLLECTION_NAME",
                "DOCUMENTS_DIR",
                "CHUNK_SIZE",
                "CHUNK_OVERLAP",
                "DATABASE_URL",
            ],
        ),
    ]
)


HEADER = """# MyAIAssistant runtime configuration
# This file is safe to share as a template. Fill in secrets only on the target machine.
# The Windows installer can preconfigure many of these values during setup.
"""


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _strip_quotes(value)
    return values


def read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    return parse_env_text(env_path.read_text(encoding="utf-8", errors="replace"))


def merged_env(overrides: Mapping[str, str] | None = None) -> OrderedDict[str, str]:
    merged = OrderedDict(DEFAULT_ENV_VALUES)
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            merged[key] = str(value)
    return merged


def render_env_text(overrides: Mapping[str, str] | None = None) -> str:
    values = merged_env(overrides)
    sections: list[str] = [HEADER.rstrip(), ""]

    for group_name, keys in GROUPED_COMMENTS.items():
        sections.append(f"# {group_name}")
        for key in keys:
            sections.append(f"{key}={values[key]}")
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def write_env_file(path: str | Path, overrides: Mapping[str, str] | None = None) -> Path:
    env_path = Path(path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(render_env_text(overrides), encoding="utf-8")
    return env_path


def ensure_env_file(
    env_path: str | Path,
    example_path: str | Path | None = None,
    overrides: Mapping[str, str] | None = None,
) -> Path:
    env_path = Path(env_path)
    if env_path.exists():
        return env_path

    example = Path(example_path) if example_path else None
    if example and example.exists():
        env_path.write_text(example.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        return env_path

    return write_env_file(env_path, overrides=overrides)


def load_runtime_config(env_path: str | Path) -> OrderedDict[str, str]:
    return merged_env(read_env_file(env_path))
