# torchy2

Self-contained RL experiments and shared infrastructure for Google Colab and Hugging Face Storage Buckets.

## Structure

Each experiment directory is an isolated project with its own `pyproject.toml`, `run.py`, `train.py`, `configs/`, `env/`, and `exp/<run_name>/` run logbooks. Shared infrastructure lives in `common/` (`torchy-core`) and links into each experiment via `[tool.uv.sources]`.

| Directory | Purpose |
| :--- | :--- |
| [`common/`](./common/) | Shared `torchy-core` package: Colab SSH/tmux orchestration (`infra`), SWS config loader (`sws_utils`), runtime patches (`patch`), LoRA reverse proxy (`proxy`), and bucket sync callback (`callback`). |
| [`async-grpo-lora-hf-repro/`](./async-grpo-lora-hf-repro/) | Multi-server Asynchronous GRPO + LoRA reproduction (`Qwen/Qwen2.5-Math-1.5B` on `sail/Sanity-Test-R1D-1.5B`) across two Google Colab L4 VMs. |

## Quickstart

```bash
cd async-grpo-lora-hf-repro
uv run python run.py --config configs/repro.py --dry-run
```
