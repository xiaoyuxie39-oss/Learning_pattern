#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def parse_scalar(text: str) -> Any:
    s = text.strip()
    if not s:
        return ""
    if s in {"null", "NULL", "None", "none", "~"}:
        return None
    if s in {"true", "True"}:
        return True
    if s in {"false", "False"}:
        return False
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    try:
        if any(ch in s for ch in [".", "e", "E"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_simple_yaml(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            if indent % 2 != 0:
                raise ValueError(f"Invalid indent at line {lineno}: {line}")
            if ":" not in stripped:
                raise ValueError(f"Invalid mapping at line {lineno}: {line}")

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            if not stack:
                raise ValueError(f"Bad indentation at line {lineno}: {line}")

            parent = stack[-1][1]
            if not value:
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = parse_scalar(value)

    return root


def repo_root_from_file(file_path: str) -> Path:
    # scripts/stage3/*.py -> repo root is three levels up.
    return Path(file_path).resolve().parents[2]


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_log(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def as_str_dict(d: Dict[str, Any]) -> Dict[str, str]:
    return {k: "" if v is None else str(v) for k, v in d.items()}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
