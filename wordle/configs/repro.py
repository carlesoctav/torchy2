from __future__ import annotations

import sws


def get_config() -> sws.Config:
    config = sws.Config()

    config.exp_name = "qwen3-1.7b-grpo-wordle"
    config.project_name = "wordle-qwen3"
    config.bucket_id = "carlesoctav/multi-serverasync-grpo"
    config.output_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"
    config.seed = 3407
    config.remote_root = "/content/torchy2"
    config.rollout_worker_name = "thinking_budget_rollout"
    config.rollout_worker.thinking_budget = 800
    config.rollout_worker.mask_force_think = True

    config.inference.infra.session = "vllm"
    config.inference.infra.gpu = "l4"
    config.inference.infra.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.inference.infra.local_forwards = ["8000:localhost:8000"]
    config.inference.infra.stop_vm = False

    config.inference.vllm.model = "Qwen/Qwen3-1.7B"
    config.inference.vllm.host = "0.0.0.0"
    config.inference.vllm.port = 8000
    config.inference.vllm.max_model_len = 4096
    config.inference.vllm.gpu_memory_utilization = 0.85
    config.inference.vllm.max_lora_rank = 8
    config.inference.vllm.max_loras = 6
    config.inference.vllm.logprobs_mode = "processed_logprobs"
    config.inference.vllm.generation_config = "vllm"
    config.inference.vllm.reasoning_parser = "qwen3"
    config.inference.vllm.enable_lora = True
    config.inference.vllm.env_vars = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
        "VLLM_SERVER_DEV_MODE": "1",
    }

    config.train.infra.session = "trainer"
    config.train.infra.gpu = "l4"
    config.train.infra.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.train.infra.remote_forwards = ["8000:localhost:8000"]
    config.train.infra.local_forwards = ["8001:localhost:8001"]
    config.train.infra.stop_vm = False

    config.train.output_dir = lambda: config.output_dir

    config.train.model.model = "Qwen/Qwen3-1.7B"
    config.train.model.peft_config.r = 8
    config.train.model.peft_config.lora_alpha = 16
    config.train.model.peft_config.target_modules = "all-linear"

    config.train.grpo.run_name = lambda: config.exp_name
    config.train.grpo.project = lambda: config.project_name
    config.train.grpo.output_dir = lambda: config.train.output_dir
    config.train.grpo.bf16 = True
    config.train.grpo.learning_rate = 1e-5
    config.train.grpo.adam_beta2 = 0.95
    config.train.grpo.max_completion_length = 1024
    config.train.grpo.max_tool_calling_iterations = 6
    config.train.grpo.token_budget = 0
    config.train.grpo.gradient_accumulation_steps = 4
    config.train.grpo.max_inflight_tasks = 128
    config.train.grpo.queue_maxsize = 512
    config.train.grpo.weight_sync_steps = 4
    config.train.grpo.max_steps = 1000
    config.train.grpo.save_steps = 100
    config.train.grpo.vllm_server_base_url = "http://127.0.0.1:8000"
    config.train.grpo.report_to = "trackio"
    config.train.grpo.log_completions = True
    config.train.grpo.num_completions_to_print = 2
    config.train.grpo.chat_template_kwargs = {"enable_thinking": False}

    config.env.name = "wordle"
    config.env.infra = lambda: config.train.infra
    config.env.client.env_url = "http://127.0.0.1:8001"
    config.env.client.dataset_size = 2000
    config.env.client.seed = 42

    config.env.server.host = "0.0.0.0"
    config.env.server.port = 8001
    config.env.server.max_sessions = 256
    config.env.server.enable_web_interface = True

    return config
