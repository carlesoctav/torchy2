from __future__ import annotations

import asyncio
import logging
import queue
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import aiohttp
from accelerate.state import PartialState
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    RolloutGroup,
    RolloutSample,
    TrainingSequence,
)

from torchy_core.rollout.thinking_budget_rollout import ThinkingBudgetRolloutLoop

logger = logging.getLogger(__name__)


class _EvalRolloutLoop(ThinkingBudgetRolloutLoop):
    """Finite single-pass `_AsyncRolloutLoop` for evaluation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["score_queue_maxsize"] = 0
        super().__init__(**kwargs)
        self._gen_stop_event = asyncio.Event()
        if len(self.dataset) == 0 or self.num_generations == 0:
            self._gen_stop_event.set()

    def _repeat_iterator(self) -> Iterator[tuple[int, dict[str, Any]]]:
        total = len(self.dataset) * self.num_generations
        emitted = 0
        for group_id, row in enumerate(self.dataset):
            for _ in range(self.num_generations):
                emitted += 1
                if emitted >= total:
                    self._gen_stop_event.set()
                yield group_id, row

    async def _score_group(self, group: RolloutGroup) -> list[RolloutSample]:
        group.completions_sequences = [
            seqs[:1] if seqs else [TrainingSequence([], [], [], "")]
            for seqs in group.completions_sequences
        ]
        return await super()._score_group(group)

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        async def _forward_parent_stop() -> None:
            await stop_event.wait()
            self._gen_stop_event.set()

        forward_task = asyncio.create_task(_forward_parent_stop())
        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self.max_inflight_tasks)
            ) as session:
                self.session = session
                await asyncio.gather(
                    asyncio.create_task(self._generate_loop(stop_event=self._gen_stop_event)),
                    asyncio.create_task(self._score_loop(stop_event=stop_event)),
                )
        finally:
            forward_task.cancel()


class _EvalRolloutWorker(AsyncRolloutWorker):
    _loop_cls = _EvalRolloutLoop


class AsyncEval:
    def __init__(
        self,
        vllm_server: str,
        model_name: str,
        env_client: dict[str, Any],
        *,
        output_dir: str | Path = "outputs",
        project_name: str = "default",
        exp_name: str = "eval",
        lora_name: str | None = None,
        num_generations: int = 1,
        max_inflight_tasks: int = 32,
        max_completion_length: int = 1024,
        max_tool_calling_iterations: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        request_timeout: int = 120,
        chat_template_kwargs: dict[str, Any] | None = None,
        log_completions: bool = False,
        num_completions_to_print: int | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ) -> None:
        self.output_dir = Path(output_dir) / project_name / exp_name
        self.parquet_path = self.output_dir / "rollouts.parquet"

        dataset: Dataset = env_client["train_dataset"]
        environment_factory = env_client.get("environment_factory")
        reward_funcs = env_client.get("reward_funcs", [])
        if callable(reward_funcs):
            reward_funcs = [reward_funcs]
        tools = env_client.get("tools")
        if processing_class is None:
            processing_class = env_client.get("processing_class")

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_name)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        self.total_rollouts = len(dataset) * num_generations
        self._stale_after_s = max(300.0, float(request_timeout) * 3.0)

        PartialState()
        self.worker = _EvalRolloutWorker(
            model_name=model_name,
            lora_name=lora_name,
            dataset=dataset,
            reward_funcs=reward_funcs,
            processing_class=processing_class,
            tools=tools,
            environment_factory=environment_factory,
            num_generations=num_generations,
            max_inflight_tasks=max_inflight_tasks,
            queue_maxsize=0,
            vllm_server_url=vllm_server,
            max_tokens=max_completion_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            request_timeout=request_timeout,
            chat_template_kwargs=chat_template_kwargs,
            max_tool_calling_iterations=max_tool_calling_iterations,
            log_completions=log_completions,
            num_completions_to_print=num_completions_to_print,
            **kwargs,
        )

    def run(self) -> None:
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)

        if self.total_rollouts == 0:
            table = Dataset.from_dict({"rollout": [], "reward": []})
            table.to_parquet(str(self.parquet_path))
            logger.info(f"Saved 0 rollouts to {self.parquet_path}")
            return

        records: list[tuple[int, list[dict[str, Any]], float]] = []
        self.worker.start()
        try:
            while len(records) < self.total_rollouts:
                self.worker.check_health(stale_after_s=self._stale_after_s)
                try:
                    sample: RolloutSample = self.worker.rollout_buffer.get(timeout=0.5)
                except queue.Empty:
                    continue
                rollout = [*sample.prompt, *sample.completion]
                reward = float(sample.metrics["reward"])
                records.append((sample.group_id, rollout, reward))
        finally:
            records.sort(key=lambda item: item[0])
            table = Dataset.from_dict(
                {
                    "rollout": [r[1] for r in records],
                    "reward": [r[2] for r in records],
                }
            )
            table.to_parquet(str(self.parquet_path))

            rewards = table["reward"]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            logger.info(
                f"Saved {len(table)} rollouts to {self.parquet_path} (mean_reward={mean_reward:.4f})"
            )
            self.worker.stop()
