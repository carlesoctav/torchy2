from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("torchy_core.patch.trackio_step")

_PATCHED = False


def patch_trackio_step() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    logger.info("[patch_trackio_step] Synchronizing Trackio step with trainer.state.global_step")

    from transformers.integrations.integration_utils import TrackioCallback, rewrite_logs
    import trl.experimental.async_grpo.async_grpo_trainer as agt

    def on_log(self, args: Any, state: Any, control: Any, model: Any = None, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero and logs is not None:
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._trackio.log({**non_scalar_logs, "train/global_step": state.global_step}, step=state.global_step)

    TrackioCallback.on_log = on_log

    orig_get_train_dataloader = agt.AsyncGRPOTrainer.get_train_dataloader

    def get_train_dataloader(self: Any) -> Any:
        dataloader = orig_get_train_dataloader(self)
        if self._rollout_dataset is not None:
            self._rollout_dataset._get_global_step = lambda: self.state.global_step
        return dataloader

    agt.AsyncGRPOTrainer.get_train_dataloader = get_train_dataloader
