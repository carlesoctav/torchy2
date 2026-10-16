from __future__ import annotations

import importlib
from typing import Any


def make_env(env_name: str, env_config: dict[str, Any]) -> dict[str, Any]:
    return importlib.import_module(f"env.{env_name}.client").make(**env_config)
