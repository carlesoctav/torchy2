from __future__ import annotations

import sws


def get_config() -> sws.Config:
    config = sws.Config()

    config.exp_name = "qwen3-1.7b-wordle-bench-before-with-budget-forcing"
    config.project_name = "bench-wordle"
    config.output_dir = lambda: f"evals/{config.project_name}/{config.exp_name}"
    config.remote_root = "/content/torchy2"
    config.bucket = "carlesoctav/budget-think"

    config.inference.infra.session = "vllm"
    config.inference.infra.gpu = "l4"
    config.inference.infra.local_forwards = ["8000:localhost:8000"]
    config.inference.infra.volumes = lambda : [f"{config.bucket}:/content/lora"]
    config.inference.infra.stop_vm = False

    config.inference.vllm.model = "/content/lora/wordle-sft/wordle-sst/checkpoint-45"
    config.inference.vllm.served_model_name = "Qwen/Qwen3-1.7B"
    config.inference.vllm.host = "0.0.0.0"
    config.inference.vllm.port = 8000
    config.inference.vllm.max_model_len = 4096
    config.inference.vllm.gpu_memory_utilization = 0.9
    config.inference.vllm.logprobs_mode = "processed_logprobs"
    config.inference.vllm.generation_config = "vllm"
    config.inference.vllm.reasoning_parser = "qwen3"
    # config.inference.vllm.safetensors_load_strategy = "prefetch"

    config.env.name = "wordle"
    config.env.client.env_url = "http://127.0.0.1:8001"
    config.env.client.dataset_size = 100
    config.env.client.seed = 67

    config.env.server.host = "127.0.0.1"
    config.env.server.port = 8001
    config.env.server.max_sessions = 256
    config.env.server.error_allowance = 3
    config.env.server.enable_web_interface = False

    config.eval.output_dir = lambda: config.output_dir
    config.eval.max_completion_length = 2048
    config.eval.max_tool_calling_iterations = 6 + 3
    config.eval.max_inflight_tasks = 32
    config.eval.chat_template_kwargs = {"enable_thinking": True}
    # config.eval.thinking_budget = 768
    config.eval.temperature = 0.7
    config.eval.top_p = 0.8
    config.eval.top_k = 20
    config.eval.extra_payload.presence_penalty = 1.5

    return config
