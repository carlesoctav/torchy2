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
from torchy_core import make_rollout_worker
from torchy_core.patch import wait_lora_patch

from env import make_env

logger = logging.getLogger(__name__)


def main(config: sws.FinalConfig, unused_argv: list[str] | None = None):
    model_dict = config.train.model.to_dict()
    grpo_dict = config.train.grpo.to_dict()
    output_dir = config.output_dir
    env_name = config.env.name
    env_client_dict = config.env.client.to_dict()
    rollout_worker_name = config.rollout_worker_name
    rollout_worker_dict = config.rollout_worker.to_dict()
    resume_checkpoint = config.resume_checkpoint

    last_checkpoint = None

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

    # import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer
    from functools import cache

    from kernels import get_kernel
    from peft import LoraConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer, packing
    from trl.experimental.async_grpo.async_grpo_trainer import save_lora_adapter

    packing._flash_attn = cache(lambda : get_kernel("kernels-community/flash-attn2", version = 3))

    # _create_model_from_path = async_grpo_trainer.create_model_from_path

    # def _create_model_untied(model_id, **kwargs):
    #     model = _create_model_from_path(model_id, **kwargs)
    #     if hasattr(model, "lm_head") and hasattr(model, "config"):
    #         model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    #         model.config.tie_word_embeddings = False
    #     return model
    #
    # async_grpo_trainer.create_model_from_path = _create_model_untied

    env_client_dict = make_env(env_name, env_client_dict)
    if "peft_config" in model_dict:
        model_dict["peft_config"] = LoraConfig(**model_dict["peft_config"])

    args = AsyncGRPOConfig(**grpo_dict)

    rollout_worker = None
    if rollout_worker_name is not None:
        rollout_worker = make_rollout_worker(
            rollout_worker_name, rollout_worker_dict, model_dict, env_client_dict, args=args
        )

    config_json = Path(f"{output_dir}/config.json")
    config_json.parent.mkdir(parents=True, exist_ok=True)
    with config_json.open("w") as f:
        f.write(config.to_json())

    trainer = AsyncGRPOTrainer(
        **model_dict,
        args=args,
        rollout_worker=rollout_worker,
        **env_client_dict,
    )

    if resume_checkpoint:
        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        logger.info(f"[resume] Resuming from checkpoint: {last_checkpoint}")

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info(f"[done] Finished training. global_step={trainer.state.global_step}")
    except KeyboardInterrupt:
        final_dir = os.path.join(output_dir, f"checkpoint-{trainer.state.global_step}")
        logger.info(f"save final-checkpoint training. global_step={trainer.state.global_step}")
        save_lora_adapter(
            trainer.model,
            trainer.accelerator,
            trainer._adapter_name,
            final_dir,
        )
        logger.info(f"[final] Saved final adapter to {final_dir}")




if __name__ == "__main__":
    sws.run(main, forward_extras=True)
