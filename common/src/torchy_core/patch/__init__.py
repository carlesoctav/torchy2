from __future__ import annotations

from torchy_core.patch.fp8 import patch_fp8_training, unpatch_fp8_training
from torchy_core.patch.log_completions import patch_rollout_dataset
from torchy_core.patch.patch_save_tokenizer import (
    patch_save_tokenizer,
    unpatch_convert_added_tokens,
)
from torchy_core.patch.skip_zero_advantage import (
    SkipZeroAdvantageRolloutLoop,
    SkipZeroAdvantageThinkingBudgetRolloutLoop,
    patch_skip_zero_advantage,
)
from torchy_core.patch.trackio_step import patch_trackio_step
from torchy_core.patch.wait_lora import wait_lora_patch

__all__ = [
    "SkipZeroAdvantageRolloutLoop",
    "SkipZeroAdvantageThinkingBudgetRolloutLoop",
    "patch_fp8_training",
    "patch_rollout_dataset",
    "patch_save_tokenizer",
    "patch_skip_zero_advantage",
    "patch_trackio_step",
    "unpatch_convert_added_tokens",
    "unpatch_fp8_training",
    "wait_lora_patch",
]
