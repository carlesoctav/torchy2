#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

import sws

from env import make_env
from torchy_core.patch import wait_lora_patch
from torchy_core.sws_utils import run as sws_run

logger = logging.getLogger(__name__)


def main(config: sws.FinalConfig):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    trl_logger = logging.getLogger("trl")
    if not trl_logger.handlers:
        trl_logger.addHandler(handler)
    trl_logger.setLevel(logging.INFO)
    core_logger = logging.getLogger("torchy_core")
    if not core_logger.handlers:
        core_logger.addHandler(handler)
    core_logger.setLevel(logging.INFO)

    wait_lora_patch(max_wait=300, retry_interval=2.0, initial_delay=0.0)

    import torch
    import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer
    from peft import LoraConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

    _create_model_from_path = async_grpo_trainer.create_model_from_path

    def _create_model_untied(model_id, **kwargs):
        model = _create_model_from_path(model_id, **kwargs)
        if hasattr(model, "lm_head") and hasattr(model, "config"):
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
            model.config.tie_word_embeddings = False
        return model

    async_grpo_trainer.create_model_from_path = _create_model_untied

    env_dict = make_env(config.env_name, config.env.to_dict() if hasattr(config.env, "to_dict") else dict(config.env))
    model_dict = config.model.to_dict() if hasattr(config.model, "to_dict") else dict(config.model)
    if getattr(config.model, "peft_config", None) is not None:
        peft_cfg = config.model.peft_config
        peft_dict = peft_cfg.to_dict() if hasattr(peft_cfg, "to_dict") else dict(peft_cfg)
        model_dict["peft_config"] = LoraConfig(**peft_dict)

    grpo_dict = config.grpo.to_dict() if hasattr(config.grpo, "to_dict") else dict(config.grpo)
    args = AsyncGRPOConfig(**grpo_dict)

    config_json = Path(f"{config.output_dir}/config.json")
    config_json.parent.mkdir(parents=True, exist_ok=True)
    with config_json.open("w") as f:
        f.write(config.to_json())

    trainer = AsyncGRPOTrainer(
        **model_dict,
        args=args,
        **env_dict,
    )

    last_checkpoint = get_last_checkpoint(config.output_dir) if os.path.isdir(config.output_dir) else None
    if last_checkpoint:
        logger.info(f"[resume] Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info(f"[done] Finished training. global_step={trainer.state.global_step}")

    final_dir = os.path.join(config.output_dir, "final-adapter")
    try:
        async_grpo_trainer.save_lora_adapter(
            trainer.accelerator.unwrap_model(trainer.model),
            trainer.accelerator,
            trainer._adapter_name,
            final_dir,
        )
        logger.info(f"[final] Saved final adapter to {final_dir}")
    except Exception as e:
        logger.warning(f"Could not save final adapter to {final_dir}: {e}")


def main_cli():
    sws_run(main)


if __name__ == "__main__":
    main_cli()

