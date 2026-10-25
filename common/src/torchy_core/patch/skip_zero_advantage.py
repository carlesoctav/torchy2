from __future__ import annotations

import logging
import math
from typing import Any

from torchy_core.rollout.thinking_budget_rollout import (
    _AsyncRolloutLoop,
    ThinkingBudgetRolloutWorker,
)
from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    RolloutGroup,
    RolloutSample,
    _AsyncRolloutLoop,
)

logger = logging.getLogger("torchy_core.patch.skip_zero_advantage")

_orig_async_score_group = _AsyncRolloutLoop._score_group
_orig_thinking_score_group = _AsyncRolloutLoop._score_group


def _is_zero_advantage_group(samples: list[RolloutSample]) -> bool:
    if not samples:
        return True
    reward_std = samples[0].metrics.get("reward_std", 0.0)
    if math.isnan(reward_std) or math.isclose(reward_std, 0.0, abs_tol=1e-7):
        return True
    return all(s.advantage == 0.0 for s in samples)


class SkipZeroAdvantageRolloutLoop(_AsyncRolloutLoop):
    async def _score_group(self, group: RolloutGroup) -> list[RolloutSample]:
        samples = await _orig_async_score_group(self, group)
        if _is_zero_advantage_group(samples):
            reward_std = samples[0].metrics.get("reward_std", 0.0) if samples else 0.0
            rew = samples[0].metrics.get("reward", 0.0) if samples else 0.0
            logger.info(
                f"[score] Skipping group {group.group_id} with zero advantage difference "
                f"(reward={rew:.4f}, reward_std={reward_std:.4f})"
            )
            return []
        return samples


class SkipZeroAdvantageThinkingBudgetRolloutLoop(_AsyncRolloutLoop):
    async def _score_group(self, group: RolloutGroup) -> list[RolloutSample]:
        samples = await _orig_thinking_score_group(self, group)
        if _is_zero_advantage_group(samples):
            reward_std = samples[0].metrics.get("reward_std", 0.0) if samples else 0.0
            rew = samples[0].metrics.get("reward", 0.0) if samples else 0.0
            logger.info(
                f"[score] Skipping group {group.group_id} with zero advantage difference "
                f"(reward={rew:.4f}, reward_std={reward_std:.4f})"
            )
            return []
        return samples


_PATCHED = False


def patch_skip_zero_advantage() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    logger.info("[patch_skip_zero_advantage] Patching rollout workers to skip groups with zero advantage difference")
    AsyncRolloutWorker._loop_cls = SkipZeroAdvantageRolloutLoop
    ThinkingBudgetRolloutWorker._loop_cls = SkipZeroAdvantageThinkingBudgetRolloutLoop
    _AsyncRolloutLoop._score_group = SkipZeroAdvantageRolloutLoop._score_group
    _AsyncRolloutLoop._score_group = SkipZeroAdvantageThinkingBudgetRolloutLoop._score_group
