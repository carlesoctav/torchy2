from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import queue
import random
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import aiohttp
from datasets import Dataset
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase
from trl.chat_template_utils import (
    _SUPPORTS_RESPONSE_TEMPLATE,
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)
from torch.utils._pytree import tree_map

logger = logging.getLogger(__name__)


class AsyncVLLMCompletions:
    def __init__(
        self,
        vllm_server: str,
        model_name: str,
        env_client: dict[str, Any],
        *,
        output_dir: str | Path = "outputs",
        lora_name: str | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        eos_token_ids: list[int] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        num_generations: int = 1,
        max_inflight_tasks: int = 32,
        max_completion_length: int = 1024,
        max_tool_calling_iterations: int | None = 8,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        request_timeout: int = 120,
        thinking_budget: int | None = None,
        extra_payload: dict[str, Any] | None = None,
        override: bool = False, 
    ) -> None:
        self.output_dir = Path(output_dir)
        is_not_empty = self.output_dir.exists() and any(self.output_dir.iterdir())

        if is_not_empty and not override:
            raise ValueError(f" {self.output_dir} must be an empty dir since override is {override}")

        self.jsonl_path = self.output_dir / "rollouts.jsonl"
        self.error_jsonl_path = self.output_dir / "errors.jsonl"
        self.reward_path = self.output_dir / "rewards.json"

        self.vllm_server_url = vllm_server.rstrip("/")
        self.model_name = model_name
        self.lora_name = lora_name
        self.chat_template_kwargs = chat_template_kwargs if chat_template_kwargs is not None else {}
        self.thinking_budget = thinking_budget
        self.extra_payload = extra_payload if extra_payload is not None else {}

        dataset: Dataset = env_client["train_dataset"]
        environment_factory = env_client.get("environment_factory")
        reward_funcs = env_client.get("reward_funcs", [])
        if callable(reward_funcs):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        tools = env_client.get("tools")
        if tools is None:
            tools = []
        elif not isinstance(tools, list):
            tools = [tools]
        self._standalone_tools = tools

        if processing_class is None:
            processing_class = env_client.get("processing_class")
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_name)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        if eos_token_ids is None:
            gen_eos = GenerationConfig.from_pretrained(model_name).eos_token_id
            if gen_eos is None:
                eos_token_ids = []
            elif isinstance(gen_eos, int):
                eos_token_ids = [gen_eos]
            else:
                eos_token_ids = list(gen_eos)
            if processing_class.eos_token_id not in eos_token_ids:
                eos_token_ids = [processing_class.eos_token_id, *eos_token_ids]

        has_template = getattr(processing_class, "response_template", None) is not None
        has_schema = getattr(processing_class, "response_schema", None) is not None
        if not has_template and not has_schema:
            processing_class = add_response_schema(processing_class)
        elif has_schema and not has_template and _SUPPORTS_RESPONSE_TEMPLATE:
            warnings.warn(
                "The tokenizer has a legacy `response_schema` set but no `response_template`. The installed "
                "transformers supports the new-style `response_template`; consider migrating, as `response_schema` "
                "support will eventually be removed. See the Transformers response-parsing docs.",
                FutureWarning,
            )
        self.tokenizer = processing_class
        self.eos_token_ids = eos_token_ids

        self.dataset = dataset
        self.num_generations = num_generations
        self.total_rollouts = len(dataset) * num_generations
        self.max_inflight_tasks = max_inflight_tasks
        self.max_completion_length = max_completion_length
        self.max_tool_calling_iterations = max_tool_calling_iterations
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.request_timeout = request_timeout
        self.max_retries = 3

        self._multi_environment = isinstance(environment_factory, dict)
        if environment_factory is None:
            self.environment_factories = None
        elif self._multi_environment:
            self.environment_factories = environment_factory
        else:
            self.environment_factories = {None: environment_factory}

        self._env_reward_types: list[type] = []
        self._env_method_names: dict[str | None, list[str]] = {}
        self._env_tools: dict[str | None, list[Any]] = {}
        self._standalone_tool_dict = {tool.__name__: tool for tool in self._standalone_tools}
        self._environment_pool: dict[str | None, list[Any]] = {}
        self.tools = list(self._standalone_tools)

        if self.environment_factories is not None:
            for name, factory in self.environment_factories.items():
                instance = factory()
                has_reset = False
                methods = []
                method_names = []
                for member_name, _ in inspect.getmembers(type(instance), predicate=inspect.isfunction):
                    if member_name == "reset":
                        has_reset = True
                    elif member_name == "get_reward":
                        if type(instance) not in self._env_reward_types:
                            self._env_reward_types.append(type(instance))
                    elif not member_name.startswith("_"):
                        methods.append(getattr(instance, member_name))
                        method_names.append(member_name)
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define a callable `reset`."
                    )
                self._env_method_names[name] = method_names
                self._env_tools[name] = self._standalone_tools + methods
                self._environment_pool[name] = [instance]
                self.tools += [method for method in methods if method not in self.tools]

        if not self.reward_funcs and not self._env_reward_types:
            raise ValueError(
                "No reward source provided. Pass `reward_funcs`, or an `environment_factory` whose environment "
                "defines a `get_reward` method."
            )

        if self.tools and not is_chat_template_prefix_preserving(self.tokenizer):
            self.chat_template = get_training_chat_template(self.tokenizer)
        else:
            self.chat_template = None

        self._tool_pool = ThreadPoolExecutor(
            max_workers=max(1, max_inflight_tasks),
            thread_name_prefix="vllm-tool",
        )
        self._rollout_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._error_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._drain_thread = threading.Thread(
            target=self._drain_rollout,
            name="rollout-jsonl-drain",
            daemon=True,
        )
        self._error_drain_thread = threading.Thread(
            target=self._drain_error,
            name="error-jsonl-drain",
            daemon=True,
        )
        self._total_rollouts = 0
        self._failed_rollouts = 0
        self._rewards: list[dict[str, float]] = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self.session: aiohttp.ClientSession | None = None

    @property
    def _request_model(self) -> str:
        if self.lora_name is None:
            return self.model_name
        return self.lora_name

    def _drain_rollout(self) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f , open(self.reward_path, "a", encoding = "utf-8")as f2:
            while True:
                item = self._rollout_queue.get()

                if item is None:
                    self._rollout_queue.task_done()
                    break

                rewards = {
                    "id": item["id"],
                    "group_id": item["group_id"],
                    "reward": item["reward"]
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f2.write(json.dumps(rewards, ensure_ascii=False) + "\n")
                f.flush()
                f2.flush()
                self._rollout_queue.task_done()

    def _drain_error(self) -> None:
        with open(self.error_jsonl_path, "a", encoding="utf-8") as f:
            while True:
                item = self._error_queue.get()
                if item is None:
                    self._error_queue.task_done()
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                self._error_queue.task_done()

    async def _generate_one_turn(
        self, prompt_ids: list[int]
    ) -> tuple[list[int], list[float]]:
        url = (
            self.vllm_server_url
            if self.vllm_server_url.endswith("/v1/completions")
            else f"{self.vllm_server_url}/v1/completions"
        )
        payload: dict[str, Any] = {
            "model": self._request_model,
            "prompt": prompt_ids,
            "max_tokens": self.max_completion_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": 1,
            "return_token_ids": True,
            "logprobs": 0,
            "seed": random.getrandbits(63),
        }
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.thinking_budget is not None:
            payload["thinking_token_budget"] = self.thinking_budget
        if self.extra_payload:
            payload.update(self.extra_payload)

        client_timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        data = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=payload, timeout=client_timeout) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status} error from {url}: {body}")
                    data = await resp.json()
                    if not isinstance(data, dict) or "choices" not in data or not data["choices"]:
                        raise RuntimeError(f"Unexpected response payload from {url}: {data}")
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                if attempt >= self.max_retries - 1:
                    raise
                await asyncio.sleep(min(2 ** attempt, 8))

        choice = data["choices"][0]
        return choice["token_ids"], choice["logprobs"]["token_logprobs"]

    async def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]], tool_dict: dict[str, Any]
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        tool_messages = []
        for tool_call in tool_calls:
            fn = tool_call["function"]
            name = fn["name"]
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = raw_args

            tool = tool_dict.get(name)
            if tool is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    if isinstance(arguments, dict):
                        call_fn = functools.partial(tool, **arguments)
                    elif isinstance(arguments, list):
                        call_fn = functools.partial(tool, *arguments)
                    else:
                        call_fn = functools.partial(tool, arguments)

                    if inspect.iscoroutinefunction(tool):
                        result = await call_fn()
                    else:
                        result = await loop.run_in_executor(self._tool_pool, call_fn)
                except Exception as e:
                    result = {"error": str(e)}

            tool_messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": str(result),
                }
            )
        return tool_messages

    async def _generate_score(
        self,
        environment: Any | None,
        prompt: list[dict[str, Any]],
        completion: list[dict[str, Any]],
        completion_ids: list[int],
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        rewards = {}
        if environment is not None and type(environment) in self._env_reward_types:
            get_reward = environment.get_reward
            if inspect.iscoroutinefunction(get_reward):
                r = await get_reward()
            else:
                r = await loop.run_in_executor(self._tool_pool, get_reward)
            rewards[environment.__class__.__name__] = r
        for rf in self.reward_funcs:
            if inspect.iscoroutinefunction(rf):
                r = await rf(prompts=[prompt], completions=[completion], completion_ids=[completion_ids])
                rewards[rf.__name__] = r
            else:
                r = await loop.run_in_executor(
                    self._tool_pool,
                    functools.partial(
                        rf,
                        prompts=[prompt],
                        completions=[completion],
                        completion_ids=[completion_ids],
                    ),
                )
                rewards[rf.__name__] = r
        return rewards

    async def _generate_one(
        self,
        prompt: list[dict[str, Any]],
        tool_dict: dict[str, Any],
        tools: list[Any],
        group_id: int,
        environment: Any | None,
        env_name: str | None,
    ) -> None:
        messages = [dict(m) for m in prompt]
        completion: list[dict[str, Any]] = []
        completion_ids: list[int] = []
        iteration = 0
        max_iterations = self.max_tool_calling_iterations
        rollout_prompt_tokens = 0
        rollout_completion_tokens = 0
        try:
            while True:
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_dict=False,
                    add_generation_prompt=True,
                    tools=tools or None,
                    chat_template=self.chat_template,
                    **self.chat_template_kwargs,
                )
                turn_ids, _ = await self._generate_one_turn(prompt_ids)
                assistant_message = parse_response(
                    self.tokenizer, turn_ids, prefix=prompt_ids, tools=tools
                )

                pt = len(prompt_ids)
                ct = len(turn_ids)
                rollout_prompt_tokens += pt
                rollout_completion_tokens += ct
                self._total_prompt_tokens += pt
                self._total_completion_tokens += ct

                completion.append(assistant_message)
                completion_ids.extend(turn_ids)
                messages.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls")
                if not tool_calls:
                    break

                if max_iterations is not None and iteration >= max_iterations:
                    break

                tool_messages = await self._execute_tool_calls(tool_calls, tool_dict)
                completion.extend(tool_messages)
                messages.extend(tool_messages)
                iteration += 1

            reward = await self._generate_score(environment, prompt, completion, completion_ids)
            rollout_id = str(uuid.uuid4())
            record = {
                "id": rollout_id,
                "group_id": group_id,
                "prompt": prompt,
                "completion": completion,
                "reward": reward,
                "usage": {
                    "prompt_tokens": rollout_prompt_tokens,
                    "completion_tokens": rollout_completion_tokens,
                },
            }
            self._rollout_queue.put(record)
            self._total_rollouts += 1
            self._rewards.append(reward)

            logger.info(
                f"[Rollout {rollout_id} | Group {group_id}] "
                f"Reward: {reward} | Tool Turns: {iteration} | Messages: {len(completion)} | "
                f"Tokens: {rollout_prompt_tokens} in, {rollout_completion_tokens} out"
            )
        except Exception as e:
            rollout_id = str(uuid.uuid4())
            reward = -1.0
            logger.error(f"[Rollout {rollout_id} | Group {group_id}] FAILED: {e}")
            record = {
                "id": rollout_id,
                "group_id": group_id,
                "prompt": prompt,
                "completion": completion,
                "reward": reward,
                "error": str(e),
                "usage": {
                    "prompt_tokens": rollout_prompt_tokens,
                    "completion_tokens": rollout_completion_tokens,
                },
            }
            self._error_queue.put(record)
            self._failed_rollouts += 1
        finally:
            if environment is not None and self.environment_factories is not None:
                self._environment_pool[env_name].append(environment)

    async def _generate_loop(self) -> None:
        loop = asyncio.get_running_loop()
        inflight_tasks: set[asyncio.Task] = set()

        for group_id, row in enumerate(self.dataset):
            for _ in range(self.num_generations):
                name = row.get("environment") if self._multi_environment else None
                if self._multi_environment and name not in self.environment_factories:
                    raise ValueError(
                        f"Example has `environment={name!r}`, which is not among the environments passed to "
                        f"`environment_factory`. Expected one of: {list(self.environment_factories)}."
                    )

                environment = None
                observation = None
                if self.environment_factories is not None:
                    pool = self._environment_pool[name]
                    environment = pool.pop() if pool else self.environment_factories[name]()
                    reset_kwargs = {
                        k: v for k, v in row.items() if k not in ("environment", "prompt")
                    }
                    if inspect.iscoroutinefunction(environment.reset):
                        observation = await environment.reset(**reset_kwargs)
                    else:
                        observation = await loop.run_in_executor(
                            self._tool_pool, functools.partial(environment.reset, **reset_kwargs)
                        )

                if "prompt" not in row and self.environment_factories is not None:
                    prompt = [{"role": "user", "content": ""}]
                else:
                    prompt = list(row["prompt"])

                if observation is not None:
                    last = prompt[-1]
                    content = last["content"]
                    if isinstance(observation, list) and isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    if isinstance(observation, str) and isinstance(content, list):
                        observation = [{"type": "text", "text": observation}]
                    prompt = prompt[:-1] + [{**last, "content": content + observation}]

                tool_dict = dict(self._standalone_tool_dict)
                if environment is not None and name in self._env_method_names:
                    for member_name in self._env_method_names[name]:
                        tool_dict[member_name] = getattr(environment, member_name)
                tools = self._env_tools[name] if self.environment_factories is not None else self.tools

                task = asyncio.create_task(
                    self._generate_one(
                        prompt=prompt,
                        tool_dict=tool_dict,
                        tools=tools,
                        group_id=group_id,
                        environment=environment,
                        env_name=name,
                    )
                )
                inflight_tasks.add(task)

                if len(inflight_tasks) >= self.max_inflight_tasks:
                    done, inflight_tasks = await asyncio.wait(
                        inflight_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        if t.exception() is not None:
                            logger.error(f"Task unhandled error: {t.exception()}", exc_info=t.exception())

        if inflight_tasks:
            done, _ = await asyncio.wait(inflight_tasks)
            for t in done:
                if t.exception() is not None:
                    logger.error(f"Task unhandled error: {t.exception()}", exc_info=t.exception())

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.jsonl_path.exists():
            self.jsonl_path.unlink()
        if self.error_jsonl_path.exists():
            self.error_jsonl_path.unlink()
        if self.reward_path.exists():
            self.reward_path.unlink()

        if self.total_rollouts == 0:
            self.jsonl_path.touch()
            logger.info(f"Saved 0 rollouts to {self.jsonl_path}")
            return

        self._drain_thread.start()
        self._error_drain_thread.start()
        try:
            connector = aiohttp.TCPConnector(limit=self.max_inflight_tasks + 1)
            async with aiohttp.ClientSession(connector=connector) as session:
                self.session = session
                await self._generate_loop()
        finally:
            self._rollout_queue.put(None)
            self._error_queue.put(None)
            self._drain_thread.join()
            self._error_drain_thread.join()
            self._tool_pool.shutdown(wait=True)

        def _f(*leaf):
            return sum(leaf) / len(leaf)

        mean_reward = tree_map(_f,  *self._rewards)
        total_p = self._total_prompt_tokens
        total_c = self._total_completion_tokens
        logger.info(
            f"Saved {len(self._rewards)} rollouts to {self.jsonl_path}, failed={self._failed_rollouts}) | "
            f"mean rewards {mean_reward} | "
            f"Prompt Tokens: {total_p} | Completion Tokens: {total_c}"

        )
        if self._failed_rollouts > 0:
            logger.warning(f"Recorded {self._failed_rollouts} failed rollouts to {self.error_jsonl_path}")



