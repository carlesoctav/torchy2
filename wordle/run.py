#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

import sws
from torchy_core.infra import ColabSession, ProcessGroup, wait_for_http_ready
from torchy_core.sws_utils import _split_config_args, load_config_builder, run as sws_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


def get_config() -> sws.Config:
    return load_config_builder(str(exp_dir / "configs/repro.py"))


def build_vllm_command(cfg: sws.FinalConfig) -> list[str]:
    cmd = [
        "uv", "run", "vllm", "serve", cfg.inference.model,
        "--host", str(cfg.inference.host),
        "--port", str(cfg.inference.port),
        "--max-model-len", str(cfg.inference.max_model_len),
        "--gpu-memory-utilization", str(cfg.inference.gpu_memory_utilization),
        "--logprobs-mode", str(cfg.inference.logprobs_mode),
        "--generation-config", str(cfg.inference.generation_config),
    ]
    if getattr(cfg.inference, "enable_lora", False):
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", str(cfg.inference.max_lora_rank),
            "--max-loras", str(cfg.inference.max_loras),
        ])
    return cmd


def build_env_server_command(cfg: sws.FinalConfig) -> list[str]:
    return [
        "uv", "run", "python", "-m", "env.server",
    ]


def build_trainer_command(config_rel_path: str) -> list[str]:
    return [
        "uv", "run", "python", "train.py",
        "--config", config_rel_path,
    ]


def main(cfg: sws.FinalConfig, unused_argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run 2-VM AsyncGRPO + LoRA Reproduction")
    parser.add_argument("--stop-vm", action="store_true", help="Stop Colab VMs upon completion or interrupt")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing Colab sessions without recreating them")
    parser.add_argument("--gpu", default=None, help="Override GPU accelerator (e.g. t4, l4, a100)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting VMs")
    extra_args, _ = parser.parse_known_args(unused_argv or [])

    config_paths, _ = _split_config_args(sys.argv[1:], config_flag="--config")
    raw_config_path = config_paths[0] if config_paths else "configs/repro.py"
    config_full_path = (exp_dir / raw_config_path).resolve() if not Path(raw_config_path).is_absolute() else Path(raw_config_path)
    try:
        config_rel_path = str(config_full_path.relative_to(exp_dir))
    except ValueError:
        config_rel_path = str(raw_config_path)

    remote_root = getattr(cfg.infra, "remote_root", "/content/torchy2").rstrip("/")
    remote_exp_dir = f"{remote_root}/{exp_dir.name}"
    sync_dirs = {
        str(repo_root): remote_root,
    }

    inf_session_kwargs = cfg.infra.inference.to_dict()
    inf_session_env = inf_session_kwargs.pop("env_vars", {})
    train_session_kwargs = cfg.infra.train.to_dict()
    trainer_session_env = train_session_kwargs.pop("env_vars", {})
    for kwargs in (inf_session_kwargs, train_session_kwargs):
        if extra_args.reuse:
            kwargs["reuse"] = True
        if extra_args.stop_vm:
            kwargs["stop_vm"] = True
        if extra_args.gpu:
            kwargs["gpu"] = extra_args.gpu

    logger.info(f"Config path: {config_rel_path}")
    logger.info(f"Experiment: {cfg.exp_name} | Project: {cfg.project_name}")
    logger.info(f"Sync mapping: {sync_dirs}")

    vllm_cmd = build_vllm_command(cfg)
    env_server_cmd = build_env_server_command(cfg)
    trainer_cmd = build_trainer_command(config_rel_path)

    if extra_args.dry_run:
        logger.info("[Dry Run] vLLM command: " + " ".join(vllm_cmd))
        logger.info("[Dry Run] Env Server command: " + " ".join(env_server_cmd))
        logger.info("[Dry Run] Trainer command: " + " ".join(trainer_cmd))
        return

    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "vllm.log").unlink(missing_ok=True)
    (log_dir / "env_server.log").unlink(missing_ok=True)
    (log_dir / "trainer.log").unlink(missing_ok=True)

    vllm_session = ColabSession(**inf_session_kwargs)
    trainer_session = ColabSession(**train_session_kwargs)

    with ProcessGroup() as pg:
        vllm_env = {**inf_session_env, **cfg.inference.to_dict().get("env_vars", {})}
        vllm_proc = vllm_session.Popen(
            cmd=vllm_cmd,
            log_path=log_dir / "vllm.log",
            dir=sync_dirs,
            work_dir=remote_exp_dir,
            env=vllm_env,
        )
        pg.add("vllm", vllm_proc, session=vllm_session)

        env_server_cfg = getattr(cfg, "env_server", None)
        env_server_dict = env_server_cfg.to_dict() if hasattr(env_server_cfg, "to_dict") else {}
        env_port = int(env_server_dict.get("port", 8001))
        env_server_env = {
            **trainer_session_env,
            "ENABLE_WEB_INTERFACE": "true" if env_server_dict.get("enable_web_interface", True) else "false",
            "WORDLE_PORT": str(env_port),
            "WORDLE_HOST": str(env_server_dict.get("host", "0.0.0.0")),
            "WORDLE_MAX_SESSIONS": str(env_server_dict.get("max_sessions", 256)),
        }
        env_server_proc = trainer_session.Popen(
            cmd=env_server_cmd,
            log_path=log_dir / "env_server.log",
            dir=sync_dirs,
            work_dir=remote_exp_dir,
            env=env_server_env,
        )
        pg.add("env_server", env_server_proc, session=trainer_session)

        vllm_port = int(cfg.inference.port)
        wait_for_http_ready(url=f"http://127.0.0.1:{vllm_port}/health", timeout=300)
        wait_for_http_ready(url=f"http://127.0.0.1:{env_port}/health", timeout=180)

        trainer_proc = trainer_session.Popen(
            cmd=trainer_cmd,
            log_path=log_dir / "trainer.log",
            dir=sync_dirs,
            work_dir=remote_exp_dir,
            env=trainer_session_env,
        )
        pg.add("trainer", trainer_proc, session=trainer_session)

        pg.wait("trainer")
        logger.info("Trainer finished.")


def main_cli():
    sws_run(main, forward_extras=True)


if __name__ == "__main__":
    main_cli()

