from __future__ import annotations

import sws


def get_config() -> sws.Config:
    config = sws.Config()

    config.exp_name = "qwen3-1.7b-wordle-bench"
    config.project_name = "wordle-qwen3"
    config.output_dir = "outputs"
    config.remote_root = "/content/torchy2"
    # config.rollout_worker_name = "thinking_budget_rollout"
    config.rollout_worker.thinking_budget = 768
    config.rollout_worker.mask_force_think = True

    config.inference.infra.session = "vllm"
    config.inference.infra.gpu = "l4"
    config.inference.infra.local_forwards = ["8000:localhost:8000", "8001:localhost:8001"]
    config.inference.infra.stop_vm = False

    config.inference.vllm.model = "Qwen/Qwen3-1.7B"
    config.inference.vllm.host = "0.0.0.0"
    config.inference.vllm.port = 8000
    config.inference.vllm.max_model_len = 8192
    config.inference.vllm.gpu_memory_utilization = 0.9
    config.inference.vllm.logprobs_mode = "processed_logprobs"
    config.inference.vllm.generation_config = "vllm"
    config.inference.vllm.reasoning_parser = "qwen3"

    config.env.name = "wordle"
    config.env.infra = lambda: config.inference.infra
    config.env.client.env_url = "http://127.0.0.1:8001"
    config.env.client.dataset_size = 100
    config.env.client.seed = 42

    config.env.server.host = "0.0.0.0"
    config.env.server.port = 8001
    config.env.server.max_sessions = 256
    config.env.server.enable_web_interface = True

    config.eval.output_dir = lambda: config.output_dir
    config.eval.project_name = lambda: config.project_name
    config.eval.exp_name = lambda: config.exp_name
    config.eval.max_completion_length = 1024
    config.eval.max_tool_calling_iterations = 8
    config.eval.max_inflight_tasks = 12 
    config.eval.log_completions = False
    config.eval.num_completions_to_print = 0
    config.eval.chat_template_kwargs = {"enable_thinking": True}
    config.eval.temperature= 0.6
    config.eval.top_p= 0.95
    config.eval.top_k= 20
    config.eval.min_p= None
    # config.eval.repetition_penalty= 1.0

    return config
