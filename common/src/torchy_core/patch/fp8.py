from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable

import torch
import torch.nn as nn
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

logger = logging.getLogger(__name__)


def default_fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
    if "lm_head" in fqn or "output_embeddings" in fqn:
        return False
    return isinstance(mod, nn.Linear)


def patch_fp8_training(
    recipe_name: str = "tensorwise",
    module_filter_fn: Callable[[nn.Module, str], bool] = default_fp8_module_filter,
    emulate: bool = False,
) -> None:
    """Patch create_model_from_path to convert linear layers to Float8Linear for FP8 training."""
    import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer

    if getattr(async_grpo_trainer, "_fp8_training_patched", False):
        return

    original_create_model = async_grpo_trainer.create_model_from_path
    async_grpo_trainer._original_create_model_from_path = original_create_model

    def fp8_create_model_from_path(model_id: str, **kwargs):
        model = original_create_model(model_id, **kwargs)
        config = Float8LinearConfig.from_recipe_name(recipe_name)
        if emulate:
            config = replace(config, emulate=True)
        convert_to_float8_training(model, module_filter_fn=module_filter_fn, config=config)
        logger.info(f"[fp8_patch] Converted base model '{model_id}' to Float8Linear (recipe: {recipe_name}).")
        return model

    async_grpo_trainer.create_model_from_path = fp8_create_model_from_path
    async_grpo_trainer._fp8_training_patched = True
    logger.info("[fp8_patch] Applied FP8 training patch to async_grpo_trainer.create_model_from_path.")


def unpatch_fp8_training() -> None:
    """Restore original create_model_from_path in async_grpo_trainer."""
    import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer

    if hasattr(async_grpo_trainer, "_original_create_model_from_path"):
        async_grpo_trainer.create_model_from_path = async_grpo_trainer._original_create_model_from_path
        delattr(async_grpo_trainer, "_original_create_model_from_path")
        async_grpo_trainer._fp8_training_patched = False
        logger.info("[fp8_patch] Reverted FP8 training patch on async_grpo_trainer.create_model_from_path.")
