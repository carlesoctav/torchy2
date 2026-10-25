from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import aiohttp
from datasets import Dataset
from transformers.utils.chat_template_utils import get_json_schema

logger = logging.getLogger(__name__)


class AsyncChatCompletions:
    def __init__(
        self,
        endpoints: str,
        model_name: str,
        env_client: dict[str, Any], 
        *,
        output_dir: str | Path = "outputs",
        headers: dict[str, Any] | None = None,
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
        reasoning_effort: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:


        self.output_dir = Path(output_dir)
        self.jsonl_path = self.output_dir / "rollouts.jsonl"
        self.error_jsonl_path = self.output_dir / "error.jsonl"
        self.parquet_path = self.output_dir / "rollouts.parquet"
        self.endpoints = endpoints
        self.model_name = model_name
        self.headers = headers if headers is not None else {}
        self.reasoning_effort = reasoning_effort
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
        self.max_retries =  3

        self._multi_environment = isinstance(environment_factory, dict)
        if environment_factory is None:
            self.environment_factories = None
        elif self._multi_environment:
            self.environment_factories = environment_factory
        else:
            self.environment_factories = {None: environment_factory}

        self._env_reward_types: list[type] = []
        self._env_method_names: dict[str | None, list[str]] = {}
        self._env_tool_schemas: dict[str | None, list[dict[str, Any]] | None] = {}
        self._standalone_tool_dict = {tool.__name__: tool for tool in self._standalone_tools}
        self._environment_pool: dict[str | None, list[Any]] = {}

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
                all_tools = self._standalone_tools + methods
                self._env_tool_schemas[name] = [get_json_schema(tool) for tool in all_tools] if all_tools else None
                self._environment_pool[name] = [instance]

        if not self.reward_funcs and not self._env_reward_types:
            raise ValueError(
                "No reward source provided. Pass `reward_funcs`, or an `environment_factory` whose environment "
                "defines a `get_reward` method."
            )

        self._tool_pool = ThreadPoolExecutor(
            max_workers=max(1, max_inflight_tasks),
            thread_name_prefix="chat-tool",
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
        self._rewards: list[float] = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cached_tokens = 0
        self.session: aiohttp.ClientSession | None = None

    def _drain_rollout(self) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            while True:
                item = self._rollout_queue.get()
                if item is None:
                    self._rollout_queue.task_done()
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
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
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        url = self.endpoints if self.endpoints.endswith("/chat/completions") else f"{self.endpoints.rstrip('/')}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.headers:
            headers.update(self.headers)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_completion_length,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.top_k > 0:
            payload["top_k"] = self.top_k
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.extra_payload:
            payload.update(self.extra_payload)

        client_timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        data = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=payload, headers=headers, timeout=client_timeout) as resp:
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
        return choice["message"], data.get("usage")

    async def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]], tool_dict: dict[str, Any]
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        tool_messages = []
        for tool_call in tool_calls:
            fn = tool_call["function"]
            name = fn["name"]
            raw_args = fn.get("arguments", "{}")
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

            content = result if isinstance(result, str) else json.dumps(result)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": name,
                    "content": content,
                }
            )
        return tool_messages

    async def _generate_score(
        self,
        environment: Any | None,
        prompt: list[dict[str, Any]],
        completion: list[dict[str, Any]],
    ) -> float:
        loop = asyncio.get_running_loop()
        if environment is not None and type(environment) in self._env_reward_types:
            get_reward = environment.get_reward
            if inspect.iscoroutinefunction(get_reward):
                reward = await get_reward()
            else:
                reward = await loop.run_in_executor(self._tool_pool, get_reward)
            return float(reward)

        total_reward = 0.0
        for rf in self.reward_funcs:
            if inspect.iscoroutinefunction(rf):
                r = await rf(prompts=[prompt], completions=[completion])
            else:
                r = await loop.run_in_executor(
                    self._tool_pool,
                    functools.partial(rf, prompts=[prompt], completions=[completion]),
                )
            if isinstance(r, (list, tuple)):
                r = r[0]
            total_reward += float(r)
        return total_reward

    async def _generate_one(
        self,
        prompt: list[dict[str, Any]],
        tool_dict: dict[str, Any],
        tool_schemas: list[dict[str, Any]] | None,
        group_id: int,
        environment: Any | None,
        env_name: str | None,
    ) -> None:
        try:
            messages = [dict(m) for m in prompt]
            completion: list[dict[str, Any]] = []
            iteration = 0
            max_iterations = self.max_tool_calling_iterations
            rollout_prompt_tokens = 0
            rollout_completion_tokens = 0
            rollout_cached_tokens = 0

            while True:
                assistant_message, usage = await self._generate_one_turn(messages, tool_schemas)
                if usage:
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)
                    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    rollout_prompt_tokens += pt
                    rollout_completion_tokens += ct
                    rollout_cached_tokens += cached
                    self._total_prompt_tokens += pt
                    self._total_completion_tokens += ct
                    self._total_cached_tokens += cached

                messages.append(assistant_message)
                completion.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls")
                if not tool_calls:
                    break

                if max_iterations is not None and iteration >= max_iterations:
                    break

                tool_messages = await self._execute_tool_calls(tool_calls, tool_dict)
                messages.extend(tool_messages)
                completion.extend(tool_messages)
                iteration += 1

            reward = await self._generate_score(environment, prompt, completion)
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
                    "cached_tokens": rollout_cached_tokens,
                },
            }
            self._rollout_queue.put(record)
            self._total_rollouts += 1
            self._rewards.append(reward)

            cache_rate = (
                (rollout_cached_tokens / rollout_prompt_tokens * 100)
                if rollout_prompt_tokens > 0
                else 0.0
            )
            logger.info(
                f"[Rollout {rollout_id} | Group {group_id}] "
                f"Reward: {reward:.4f} | Tool Turns: {iteration} | Messages: {len(completion)} | "
                f"Tokens: {rollout_prompt_tokens} in ({cache_rate:.1f}% cached), {rollout_completion_tokens} out"
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
                    "cached_tokens": rollout_cached_tokens,
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
                tool_schemas = self._env_tool_schemas.get(name)

                task = asyncio.create_task(
                    self._generate_one(
                        prompt=prompt,
                        tool_dict=tool_dict,
                        tool_schemas=tool_schemas,
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

        mean_reward = sum(self._rewards) / len(self._rewards) if self._rewards else 0.0
        total_p = self._total_prompt_tokens
        total_c = self._total_completion_tokens
        total_cached = self._total_cached_tokens
        overall_cache_rate = (total_cached / total_p * 100) if total_p > 0 else 0.0
        logger.info(
            f"Saved {len(self._rewards)} rollouts to {self.jsonl_path} (mean_reward={mean_reward:.4f}, failed={self._failed_rollouts}) | "
            f"Prompt Tokens: {total_p} ({overall_cache_rate:.1f}% cached) | Completion Tokens: {total_c}"
        )
        if self._failed_rollouts > 0:
            logger.warning(f"Recorded {self._failed_rollouts} failed rollouts to {self.error_jsonl_path}")
