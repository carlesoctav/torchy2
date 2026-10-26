from __future__ import annotations

import contextvars
import logging
import queue
import threading
import time
from typing import Any

import pandas as pd
import trackio
import trl.experimental.async_grpo.async_grpo_trainer as agt

logger = logging.getLogger("torchy_core.patch.log_completions")
agt_logger = logging.getLogger("trl.experimental.async_grpo.async_grpo_trainer")


def _format_args(args: Any) -> str:
    if isinstance(args, dict):
        return ", ".join(f"{k}={v!r}" for k, v in args.items())
    return str(args)


def format_entry(entry: Any) -> str:
    if isinstance(entry, list) and all(isinstance(m, dict) for m in entry):
        parts = []
        for msg in entry:
            role = msg.get("role", "")
            lines = []
            if role:
                lines.append(role.upper())
            reasoning = msg.get("reasoning_content") or msg.get("thinking")
            if reasoning:
                lines.append(str(reasoning))
            content = msg.get("content")
            if content:
                lines.append(str(content))
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "tool")
                    lines.append(f"{fn_name}({_format_args(fn.get('arguments', ''))})")
            elif "name" in msg and "args" in msg:
                lines.append(f"{msg['name']}({_format_args(msg['args'])})")
            elif not content and not reasoning:
                lines.append(str(msg))
            parts.append("\n".join(lines))
        return "\n\n".join(parts)
    return str(entry)


def _log_table(samples: list[Any], gen_step: int, step: int | None = None) -> None:
    n = len(samples)
    table = {
        "generation_step": [gen_step] * n,
        "model_version": [int(s.model_version) for s in samples],
        "prompt": [format_entry(s.prompt) for s in samples],
        "completion": [format_entry(s.completion) for s in samples],
        "reward": [float(s.metrics.get("reward_kept", s.metrics.get("reward", 0.0))) for s in samples],
        "advantage": [float(s.advantage) for s in samples],
    }
    df = pd.DataFrame(table)
    if step is not None:
        trackio.log({"completions": trackio.Table(dataframe=df)}, step=step)
    else:
        trackio.log({"completions": trackio.Table(dataframe=df)})


def patch_rollout_dataset(log_every: int = 1) -> None:
    logger.info(f"[log_completions_patch] Patching RolloutQueueDataset to log table every {log_every} steps")
    orig_init = agt.RolloutQueueDataset.__init__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._table_queue: queue.Queue = queue.Queue(maxsize=4)
        self._table_log_every = log_every
        self._table_buf: list[Any] = []
        self._last_table_step = -1
        self._get_global_step = lambda: 0
        threading.Thread(target=self._drain_table, name="rollout-table-logger", daemon=True).start()

    def _drain_table(self) -> None:
        while True:
            samples, gen_step, step, ctx = self._table_queue.get()
            ctx.run(_log_table, samples, gen_step, step)

    def __iter__(self):
        while True:
            t0 = time.time()
            while True:
                try:
                    sample = self.queue.get(timeout=self.poll_interval_s)
                    break
                except queue.Empty:
                    self.check_health_fn(self.stale_after_s)
            now = time.time()
            wait_s = now - t0
            self.wait_s += wait_s
            if wait_s > 1.0:
                agt_logger.info(f"waited {wait_s:.1f}s for sample (qsize={self.queue.qsize()})")

            version = self.model_version_fn()
            staleness = version - sample.model_version
            if staleness > self.max_staleness:
                agt_logger.info(f"dropping stale sample (staleness={staleness}, max={self.max_staleness})")
                self.metrics["sample/dropped_stale_total"].append(1.0)
                continue

            self.metrics["sample/rollout_queue_size"].append(float(self.queue.qsize()))
            if sample.enqueued_at is not None:
                self.metrics["sample/time_in_queue_s"].append(now - sample.enqueued_at)
            self.metrics["sample/staleness_mean"].append(float(staleness))
            self.metrics["sample/staleness_max"].append(float(staleness))

            if version % self._trace_log_interval == 0 and version != self._last_traced_version:
                self._trace_buf.append(sample)
                if len(self._trace_buf) >= self._traces_per_log:
                    try:
                        self._trace_queue.put_nowait((self._trace_buf, version, contextvars.copy_context()))
                    except queue.Full:
                        pass
                    self._trace_buf = []
                    self._last_traced_version = version

            gen_step = sample.group_id
            if self._table_buf and self._table_buf[0].group_id != gen_step:
                self._last_table_step = self._table_buf[0].group_id
                try:
                    self._table_queue.put_nowait(
                        (list(self._table_buf), self._last_table_step, self._get_global_step(), contextvars.copy_context())
                    )
                except queue.Full:
                    pass
                self._table_buf = []

            if gen_step % self._table_log_every == 0 and gen_step != self._last_table_step:
                self._table_buf.append(sample)
                if len(self._table_buf) >= 8:
                    self._last_table_step = gen_step
                    try:
                        self._table_queue.put_nowait(
                            (list(self._table_buf), self._last_table_step, self._get_global_step(), contextvars.copy_context())
                        )
                    except queue.Full:
                        pass
                    self._table_buf = []

            yield {
                "input_ids": sample.input_ids,
                "completion_mask": sample.completion_mask,
                "old_log_probs": sample.old_log_probs,
                "sampling_mask": sample.sampling_mask,
                "sampler_ids": sample.sampler_ids,
                "sampler_log_probs": sample.sampler_log_probs,
                "advantage": sample.advantage,
                "group_id": sample.group_id,
                "metrics": sample.metrics,
            }

    agt.RolloutQueueDataset.__init__ = __init__
    agt.RolloutQueueDataset._drain_table = _drain_table
    agt.RolloutQueueDataset.__iter__ = __iter__
