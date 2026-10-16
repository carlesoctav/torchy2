from __future__ import annotations

import importlib
from typing import Any


def make_env(env_name: str, env_config: dict[str, Any]) -> dict[str, Any]:
    env_module = importlib.import_module(f"env.{env_name}")
    return env_module.make(**env_config)
