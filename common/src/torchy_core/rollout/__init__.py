from __future__ import annotations

import importlib
from typing import Any

from torchy_core.rollout.thinking_budget_rollout import (
    ThinkingBudgetRolloutLoop,
    ThinkingBudgetRolloutWorker,
)


def make_rollout_worker(
    rollout_name: str,
    rollout_worker_config: dict[str, Any],
    model_config: dict[str, Any],
    env_config: dict[str, Any],
    args: Any = None,
) -> Any:
    return importlib.import_module(f"torchy_core.rollout.{rollout_name}").make(
        rollout_worker_config, model_config, env_config, args=args
    )


__all__ = [
    "ThinkingBudgetRolloutLoop",
    "ThinkingBudgetRolloutWorker",
    "make_rollout_worker",
]
