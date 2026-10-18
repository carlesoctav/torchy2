from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer
from trl.experimental.async_grpo.async_grpo_config import AsyncGRPOConfig
from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    _AsyncRolloutLoop,
)


class ThinkingBudgetRolloutLoop(_AsyncRolloutLoop):
    def __init__(
        self,
        *,
        thinking_budget: int | None = None,
        mask_force_think: bool = False,
        **kwargs: Any,
    ) -> None:
        self.thinking_budget = thinking_budget
        self.mask_force_think = mask_force_think
        super().__init__(**kwargs)

    async def _generate_one_turn(self, prompt_ids: list[int]) -> tuple[list[int], list[float]]:
        if self.thinking_budget is None:
            return await super()._generate_one_turn(prompt_ids)
        payload: dict[str, Any] = {
            "model": self._request_model,
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": 1,
            "return_token_ids": True,
            "logprobs": 0,
            "thinking_token_budget": self.thinking_budget,
        }
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        output = await self._retry(
            lambda: self._post("/v1/completions", payload, self.request_timeout),
            max_attempts=30,
            label="vllm /v1/completions",
        )
        choice = output["choices"][0]
        return choice["token_ids"], choice["logprobs"]["token_logprobs"]

    async def _generate_one(
        self,
        prompt: Any,
        tool_dict: dict[str, Any],
        tools: list[Any],
        group_id: int = 0,
    ) -> tuple[Any, ...]:
        result = await super()._generate_one(prompt, tool_dict=tool_dict, tools=tools, group_id=group_id)
        if self.mask_force_think:
            completion, completion_ids, sequences, tool_call_count, tool_failure_count, rollout_reward = result
            think_end_ids = set(self.tokenizer.encode("</think>", add_special_tokens=False))
            for seq in sequences:
                for i, (tok_id, lp) in enumerate(zip(seq.input_ids, seq.old_log_probs)):
                    if seq.completion_mask[i] == 1 and tok_id in think_end_ids and lp == 0.0:
                        seq.completion_mask[i] = 0
            sequences = [seq for seq in sequences if any(seq.completion_mask)]
            result = (completion, completion_ids, sequences, tool_call_count, tool_failure_count, rollout_reward)
        return result


class ThinkingBudgetRolloutWorker(AsyncRolloutWorker):
    _loop_cls = ThinkingBudgetRolloutLoop


def make(
    rollout_worker_config: dict[str, Any],
    model_config: dict[str, Any],
    env_config: dict[str, Any],
    args: AsyncGRPOConfig | None = None,
) -> ThinkingBudgetRolloutWorker:
    model_name = model_config["model"]
    lora_name = rollout_worker_config.pop("lora_name", "trl-policy" if "peft_config" in model_config else None)

    dataset = env_config["train_dataset"]
    reward_funcs = env_config.get("reward_funcs", [])
    if callable(reward_funcs):
        reward_funcs = [reward_funcs]

    processing_class = env_config.get("processing_class")
    if processing_class is None:
        processing_class = AutoTokenizer.from_pretrained(model_name)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

    tools = env_config.get("tools")
    environment_factory = env_config.get("environment_factory")

    if args is None:
        args = AsyncGRPOConfig()

    samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
    max_inflight_tasks = args.max_inflight_tasks
    if max_inflight_tasks < 0:
        max_inflight_tasks = max(args.max_staleness, 1) * samples_per_step

    thinking_budget = rollout_worker_config.pop("thinking_budget", None)
    mask_force_think = rollout_worker_config.pop("mask_force_think", False)

    worker_kwargs: dict[str, Any] = {
        "num_generations": args.num_generations,
        "max_inflight_tasks": max_inflight_tasks,
        "queue_maxsize": args.queue_maxsize,
        "vllm_server_url": args.vllm_server_base_url,
        "max_tokens": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "request_timeout": args.request_timeout,
        "chat_template_kwargs": args.chat_template_kwargs,
        "max_tool_calling_iterations": args.max_tool_calling_iterations,
        "log_completions": args.log_completions,
        "num_completions_to_print": args.num_completions_to_print,
        "fork_threshold_tokens": args.fork_threshold_tokens,
    }

    if "max_completion_length" in rollout_worker_config:
        worker_kwargs["max_tokens"] = rollout_worker_config.pop("max_completion_length")
    if "vllm_server_base_url" in rollout_worker_config:
        worker_kwargs["vllm_server_url"] = rollout_worker_config.pop("vllm_server_base_url")

    worker_kwargs.update(rollout_worker_config)

    return ThinkingBudgetRolloutWorker(
        model_name=model_name,
        lora_name=lora_name,
        dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=processing_class,
        tools=tools,
        environment_factory=environment_factory,
        thinking_budget=thinking_budget,
        mask_force_think=mask_force_think,
        **worker_kwargs,
    )
