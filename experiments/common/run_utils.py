from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm


def tools_environment_name(env: Any) -> str:
    """Name Terrarium uses to look up an environment's toolset.

    Environment subclasses set `tools_environment_name` to reuse a built-in toolset
    (e.g. `experiments/collusion/choice/`); otherwise the class name is used.
    """
    return str(getattr(env, "tools_environment_name", None) or env.__class__.__name__)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_ENV_REF = re.compile(r"\$\{(\w+)\}")


def _expand_env_refs(value: Any) -> Any:
    """Replace `${NAME}` in strings with the environment variable NAME."""
    if isinstance(value, str):

        def _sub(match: re.Match) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"Config references ${{{name}}} but {name} is not set."
                )
            return os.environ[name]

        return _ENV_REF.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_refs(v) for v in value]
    return value


def build_vllm_runtime(llm_cfg: dict[str, Any]) -> Any:
    """Terrarium's vLLM runtime, extended for a remote server that needs an API key.

    `${NAME}` in the `vllm` block (e.g. the Modal tunnel URL and key) is filled in
    here rather than at config load, so the key never lands in run_config.json.
    Terrarium checks that a server is up with unauthenticated GETs, which a server
    started with --api-key rejects. When auto_start_server is off there is nothing
    to launch anyway, so skip the check; a bad URL or key fails on the first request.
    """
    from terrarium.utils import build_vllm_runtime as _build_vllm_runtime

    llm_cfg = dict(llm_cfg)
    llm_cfg["vllm"] = _expand_env_refs(llm_cfg.get("vllm") or {})
    runtime = _build_vllm_runtime(llm_cfg)
    if not runtime.global_config.auto_start_server:
        runtime.server_manager.ensure_server = lambda spec: None
    return runtime


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, data)
    tmp.replace(path)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def configure_experiment_logging(
    logger: logging.Logger, root: Path, *, verbose: bool = True
) -> None:
    """Configure a dedicated experiment logger (file + tqdm-safe console)."""
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reset handlers so repeated calls (or notebooks) re-point to the new output root.
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)

    log_path = root / "experiment.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    if verbose:
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(console_handler)


def write_progress(root: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat()
    atomic_write_json(root / "progress.json", payload)


def normalize_seeds(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        # Allow comma-separated values as a convenience.
        parts = [p.strip() for p in raw.split(",")]
        seeds: list[int] = []
        for part in parts:
            if not part:
                continue
            try:
                seeds.append(int(part))
            except Exception:
                continue
        return seeds
    if isinstance(raw, list):
        seeds = []
        for item in raw:
            if item is None:
                continue
            try:
                seeds.append(int(item))
            except Exception:
                continue
        return seeds
    return []
