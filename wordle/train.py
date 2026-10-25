import logging
import os
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

import sws
from torchy_core import make_rollout_worker
from torchy_core.patch import (
    patch_fp8_training,
    patch_rollout_dataset,
    patch_skip_zero_advantage,
    patch_trackio_step,
    wait_lora_patch,
)

from env import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main(config: sws.FinalConfig, unused_argv: list[str] | None = None):
    model_dict = config.train.model.to_dict()
    grpo_dict = config.train.grpo.to_dict()
    output_dir = config.output_dir
    env_name = config.env.name
    env_client_dict = config.env.client.to_dict()

    rollout_worker_name = config.get("rollout_worker_name", None)
    resume_checkpoint = config.get("resume_checkpoint", False)


    env_client_dict = make_env(env_name, env_client_dict)
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
    if config.get("fp8_train", False):
        patch_fp8_training()
    if config.get("log_rollout", False):
        patch_rollout_dataset(config.get("log_every", 1))
    if config.train.grpo.get("report_to") == "trackio" or config.get("trackio", False):
        patch_trackio_step()
    if config.get("skip_zero_advantage", False):
        patch_skip_zero_advantage()

    from functools import cache

    from kernels import get_kernel
    from peft import LoraConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer, packing


    packing._flash_attn = cache(lambda : get_kernel("kernels-community/flash-attn2", version = 3))

    if "peft_config" in model_dict:
        model_dict["peft_config"] = LoraConfig(**model_dict["peft_config"])

    args = AsyncGRPOConfig(**grpo_dict)

    rollout_worker = None
    if rollout_worker_name is not None:
        rollout_worker_dict = config.rollout_worker.to_dict()
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
    finally:
        logger.info(f"save final-checkpoint training. global_step={trainer.state.global_step}")
        trainer._save_checkpoint(trainer.model, trial=None)




if __name__ == "__main__":
    sws.run(main, forward_extras=True)
