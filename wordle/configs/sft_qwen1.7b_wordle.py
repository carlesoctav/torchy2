from __future__ import annotations

import sws


def get_config() -> sws.Config:
    config = sws.Config()

    config.exp_name = "qwen-wordle-dft"
    config.project_name = "wordle-sft"
    config.bucket = "carlesoctav/budget-think"
    config.output_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"

    config.seed = 3407
    config.remote_root = "/content/torchy2"
    config.resume_checkpoint = False

    config.train.infra.session = "trainer"
    config.train.infra.gpu = "l4"
    config.train.infra.volumes = lambda : [f"{config.bucket}:/content/lora"]
    # config.train.infra.local_forwards = ["8001:localhost:8001"]
    # config.train.infra.remote_forwards = ["8000:localhost:8000"]
    config.train.infra.stop_vm = False
    config.train.infra.env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


    config.train.model.model = "Qwen/Qwen3-1.7B"
    config.train.sft.model_init_kwargs = {
        "dtype": "bfloat16",
        "attn_implementation": "kernels-community/flash-attn2",
    }
    # config.train.model.peft_config.r = 32
    # config.train.model.peft_config.lora_alpha = 64
    # config.train.model.peft_config.target_modules = "all-linear"

    config.train.sft.run_name = lambda: config.exp_name
    config.train.sft.project = lambda: config.project_name
    config.train.sft.output_dir = lambda: config.output_dir
    config.train.sft.report_to = "trackio"
    config.train.sft.logging_steps = 1

    config.train.sft.bf16 = True
    config.train.sft.learning_rate = 5e-5
    config.train.sft.max_length = 4096
    config.train.sft.per_device_train_batch_size = 4
    config.train.sft.gradient_accumulation_steps = 4
    config.train.sft.packing = True
    config.train.sft.assistant_only_loss = True
    config.train.sft.loss_type =  "nll"
    config.train.sft.max_grad_norm = 1

    config.train.data.path = "carlesoctav/f4.1-wordle"
    config.train.data.split = "train"


    return config

