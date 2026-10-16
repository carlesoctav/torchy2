from __future__ import annotations

import sws


def get_config() -> sws.Config:
    config = sws.Config()

    config.exp_name = "qwen3-1.7b-grpo-wordle"
    config.project_name = "wordle-qwen3"
    config.bucket_id = "carlesoctav/multi-serverasync-grpo"
    config.output_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"
    config.seed = 3407

    config.infra.inference.session = "vllm"
    config.infra.inference.gpu = "l4"
    config.infra.inference.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.infra.inference.local_forwards = ["8000:localhost:8000"]
    config.infra.inference.stop_vm = False

    config.infra.train.session = "trainer"
    config.infra.train.gpu = "l4"
    config.infra.train.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.infra.train.remote_forwards = ["8000:localhost:8000"]
    config.infra.train.local_forwards = ["8001:localhost:8001"]
    config.infra.train.stop_vm = False

    config.infra.remote_root = "/content/torchy2"

    config.inference.model = "Qwen/Qwen3-1.7B"
    config.inference.host = "0.0.0.0"
    config.inference.port = 8000
    config.inference.max_model_len = 4096
    config.inference.gpu_memory_utilization = 0.85
    config.inference.max_lora_rank = 8
    config.inference.max_loras = 6
    config.inference.logprobs_mode = "processed_logprobs"
    config.inference.generation_config = "vllm"
    config.inference.enable_lora = True
    config.inference.env_vars = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
        "VLLM_SERVER_DEV_MODE": "1",
    }

    config.env_server.host = "0.0.0.0"
    config.env_server.port = 8001
    config.env_server.max_sessions = 256
    config.env_server.enable_web_interface = True

    config.model.model = "Qwen/Qwen3-1.7B"
    config.model.peft_config.r = 8
    config.model.peft_config.lora_alpha = 16
    config.model.peft_config.target_modules = "all-linear"

    config.grpo.run_name = lambda: f"{config.project_name}_{config.exp_name}"
    config.grpo.project = lambda: config.project_name
    config.grpo.output_dir = lambda: config.output_dir
    config.grpo.bf16 = True
    config.grpo.learning_rate = 1e-5
    config.grpo.lr_scheduler_type = "constant"
    config.grpo.adam_beta2 = 0.95
    config.grpo.num_generations = 8
    config.grpo.temperature = 1.0
    config.grpo.top_p = 1.0
    config.grpo.max_completion_length = 1024
    config.grpo.max_tool_calling_iterations = 6
    config.grpo.token_budget = 0
    config.grpo.per_device_train_batch_size = 8
    config.grpo.gradient_accumulation_steps = 4
    config.grpo.gradient_checkpointing = True
    config.grpo.max_staleness = 4
    config.grpo.max_inflight_tasks = 128
    config.grpo.queue_maxsize = 512
    config.grpo.weight_sync_steps = 4
    config.grpo.max_steps = 1000
    config.grpo.save_strategy = "steps"
    config.grpo.save_steps = 100
    config.grpo.vllm_server_base_url = "http://127.0.0.1:8000"
    config.grpo.report_to = "trackio"
    config.grpo.logging_steps = 1
    config.grpo.log_completions = True
    config.grpo.num_completions_to_print = 2
    config.grpo.chat_template_kwargs = {"enable_thinking": False}

    config.env_name = "wordle"
    config.env.env_url = "http://127.0.0.1:8001"
    config.env.dataset_size = 2000
    config.env.seed = 42

    return config
