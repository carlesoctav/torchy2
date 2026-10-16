# Multi-Server Async GRPO with LoRA Reproduction

Reproduces the Hugging Face reference recipe for multi-server Asynchronous GRPO with LoRA training across separate generation and training nodes using a shared Hugging Face Storage Bucket.

- **Reference Blog**: [Hugging Face Async GRPO with LoRA & HF Jobs](https://huggingface.co/blog/asyncgrpo-lora-hfjobs)
- **Reference Code**: [AmineDiro/hfjobs-lora-buckets](https://github.com/AmineDiro/hfjobs-lora-buckets)
- **Trackio Dashboard**: [https://trackio.intelligences.systems](https://trackio.intelligences.systems) (`async-grpo-lora-hf-repro_qwen2.5-math-1.5b-r1-b128`)

## Directory Layout

```text
async-grpo-lora-hf-repro/
├── pyproject.toml        # Experiment dependencies + editable ../common (torchy-core)
├── run.py                # 2-VM Colab orchestrator (vLLM + Trainer)
├── train.py              # AsyncGRPOTrainer entrypoint
├── configs/
│   └── repro.py          # Unified 2-VM + AsyncGRPO configuration
├── env/
│   ├── __init__.py       # make_env dispatcher
│   └── sanity_math.py    # sail/Sanity-Test-R1D-1.5B + QwenMathReward
└── exp/
    └── qwen2.5-math-1.5b-r1-b128/
        └── README.md     # Run logbook & 500-step results
```

## Running on Google Colab

```bash
cd async-grpo-lora-hf-repro

# Dry run to inspect resolved commands and sync mapping
uv run python run.py --config configs/repro.py --dry-run

# Launch both Colab VMs (vLLM + Trainer) and run training
uv run python run.py --config configs/repro.py
```

## Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-Math-1.5B` |
| **Dataset** | `sail/Sanity-Test-R1D-1.5B` |
| **LoRA Rank / Alpha** | `r=1`, `lora_alpha=2`, `target_modules="all-linear"` |
| **Micro-batch / Grad Accum** | `per_device_train_batch_size=8`, `gradient_accumulation_steps=16` (128 completions/step) |
| **Token Budget** | `0` (`FixedCountBatcher`) |
| **Max Steps** | `500` |
| **Learning Rate** | `4e-5` (constant) |
| **Generations per Prompt** | `8` (`max_completion_length=3000`, `max_model_len=4096`) |
| **Max Staleness / Sync Steps** | `max_staleness=4`, `weight_sync_steps=4` |

## Results Summary (500 Steps)

| Step Range | Min Reward | Mean Reward | Max Peak | Checkpoint |
| :--- | :---: | :---: | :---: | :--- |
| **001 - 050** | 0.0703 | **0.1520** | 0.3047 | `checkpoint-50` |
| **051 - 100** | 0.0703 | **0.2167** | 0.4844 | `checkpoint-100` |
| **101 - 150** | 0.1250 | **0.2970** | 0.5625 | `checkpoint-150` |
| **151 - 200** | 0.1484 | **0.3517** | 0.6250 | `checkpoint-200` |
| **201 - 250** | 0.1406 | **0.3866** | 0.5703 | `checkpoint-250` |
| **251 - 300** | 0.1875 | **0.4047** | 0.6719 | `checkpoint-300` |
| **301 - 350** | 0.1641 | **0.4111** | 0.6094 | `checkpoint-350` |
| **351 - 400** | 0.1797 | **0.4238** | 0.5938 | `checkpoint-400` |
| **401 - 450** | 0.2109 | **0.4311** | 0.6406 | `checkpoint-450` |
| **451 - 500** | 0.1641 | **0.4509** | 0.6406 | `checkpoint-500` & `final-adapter/` |
