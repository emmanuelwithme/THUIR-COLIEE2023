from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ENV_YEAR_KEY = "COLIEE_TASK1_YEAR"
ENV_ROOT_KEY = "COLIEE_TASK1_ROOT"
ENV_DIR_KEY = "COLIEE_TASK1_DIR"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


@lru_cache(maxsize=1)
def load_dotenv_if_present() -> None:
    dotenv_path = _repo_root() / ".env"
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _strip_quotes(value))


@lru_cache(maxsize=1)
def get_task1_year() -> str:
    load_dotenv_if_present()
    return os.getenv(ENV_YEAR_KEY, "2025").strip()


@lru_cache(maxsize=1)
def get_task1_root() -> str:
    load_dotenv_if_present()
    root = os.getenv(ENV_ROOT_KEY, "./coliee_dataset/task1").strip()
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = _repo_root() / root_path
    return str(root_path.resolve())


@lru_cache(maxsize=1)
def get_task1_dir() -> str:
    load_dotenv_if_present()
    explicit = os.getenv(ENV_DIR_KEY, "").strip()
    if explicit:
        explicit_path = Path(explicit)
        if not explicit_path.is_absolute():
            explicit_path = _repo_root() / explicit_path
        return str(explicit_path.resolve())
    return str((Path(get_task1_root()) / get_task1_year()).resolve())


def task1_join(*parts: str) -> str:
    return str(Path(get_task1_dir()).joinpath(*parts))
