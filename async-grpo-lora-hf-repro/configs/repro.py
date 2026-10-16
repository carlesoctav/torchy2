from __future__ import annotations

import sws


def get_config() -> sws.Config:
    config = sws.Config()

    config.exp_name = "qwen2.5-math-1.5b-r1-b128"
    config.project_name = "async-grpo-lora-hf-repro"
    config.bucket_id = "carlesoctav/multi-serverasync-grpo"
    config.output_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"
    config.seed = 3407
    config.remote_root = "/content/torchy2"

    config.inference.infra.session = "vllm"
    config.inference.infra.gpu = "l4"
    config.inference.infra.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.inference.infra.local_forwards = ["8000:localhost:8000"]
    config.inference.infra.stop_vm = False

    config.inference.vllm.model = "Qwen/Qwen2.5-Math-1.5B"
    config.inference.vllm.host = "0.0.0.0"
    config.inference.vllm.port = 8000
    config.inference.vllm.max_model_len = 4096
    config.inference.vllm.gpu_memory_utilization = 0.85
    config.inference.vllm.max_lora_rank = 1
    config.inference.vllm.max_loras = 6
    config.inference.vllm.logprobs_mode = "processed_logprobs"
    config.inference.vllm.generation_config = "vllm"
    config.inference.vllm.enable_lora = True
    config.inference.vllm.env_vars = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
        "VLLM_SERVER_DEV_MODE": "1",
    }

    config.train.infra.session = "trainer"
    config.train.infra.gpu = "l4"
    config.train.infra.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.train.infra.remote_forwards = ["8000:localhost:8000"]
    config.train.infra.stop_vm = False

    config.train.output_dir = lambda: config.output_dir

    config.train.model.model = "Qwen/Qwen2.5-Math-1.5B"
    config.train.model.peft_config.r = 1
    config.train.model.peft_config.lora_alpha = 2
    config.train.model.peft_config.target_modules = "all-linear"

    config.train.grpo.run_name = lambda: config.exp_name
    config.train.grpo.project = lambda: config.project_name
    config.train.grpo.output_dir = lambda: config.train.output_dir
    config.train.grpo.bf16 = True
    config.train.grpo.learning_rate = 4e-5
    config.train.grpo.adam_beta2 = 0.95
    config.train.grpo.max_completion_length = 3000
    config.train.grpo.token_budget = 0
    config.train.grpo.gradient_accumulation_steps = 16
    config.train.grpo.max_inflight_tasks = 128
    config.train.grpo.queue_maxsize = 512
    config.train.grpo.weight_sync_steps = 4
    config.train.grpo.max_steps = 500
    config.train.grpo.save_steps = 50
    config.train.grpo.vllm_server_base_url = "http://127.0.0.1:8000"
    config.train.grpo.report_to = "trackio"
    config.train.grpo.log_completions = True
    config.train.grpo.num_completions_to_print = 2

    config.env.name = "sanity_math"
    config.env.infra = lambda: config.train.infra
    config.env.client.dataset_name = "sail/Sanity-Test-R1D-1.5B"
    config.env.client.split = "train"
    config.env.client.model_id = lambda: config.train.model.model
    config.env.client.max_completion_length = lambda: config.train.grpo.max_completion_length
    config.env.client.limit = 0

    return config
