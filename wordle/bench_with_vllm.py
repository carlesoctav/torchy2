#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import  time
from pathlib import Path


start = time.monotonic()

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

import sws
from env import make_env
from torchy_core import AsyncVLLMCompletions, build_env_server_command, build_vllm_command
from torchy_core.infra import ColabSession, ProcessGroup, wait_for_http_ready
from torchy_core.sws_utils import load_config_builder, sws_run_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_with_vllm")


def get_config() -> sws.Config:
    return load_config_builder(str(exp_dir / "configs/bench_qwen_vllm.py"))


async def main(cfg: sws.FinalConfig, unused_argv: list[str] | None = None):
    inside_main = time.monotonic() - start
    print(f"DEBUGPRINT[12]: bench_with_vllm.py:38: inside_main={inside_main}")
    parser = argparse.ArgumentParser(description="Run Wordle Benchmark with AsyncVLLMCompletions")
    parser.add_argument("--stop-vm", action="store_true", help="Stop Colab VMs upon completion or interrupt")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing Colab sessions without recreating them")
    parser.add_argument("--gpu", default=None, help="Override GPU accelerator (e.g. t4, l4, a100)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting VMs")
    extra_args, _ = parser.parse_known_args(unused_argv)

    inference_vllm_config = cfg.inference.vllm.to_dict()
    inference_infra_config = cfg.inference.infra.to_dict()
    env_server_config = cfg.env.server.to_dict()
    env_client_dict = cfg.env.client.to_dict()
    eval_config = cfg.eval.to_dict()

    remote_root = cfg.remote_root.rstrip("/")
    remote_exp_dir = f"{remote_root}/{exp_dir.name}"
    sync_dirs = {
        str(repo_root): remote_root,
    }

    if extra_args.reuse:
        inference_infra_config["reuse"] = True
    if extra_args.stop_vm:
        inference_infra_config["stop_vm"] = True
    if extra_args.gpu:
        inference_infra_config["gpu"] = extra_args.gpu

    model_name = inference_vllm_config.get("served_model_name") or inference_vllm_config["model"]
    print(f"DEBUGPRINT[11]: bench_with_vllm.py:60: model_name={model_name}")
    vllm_port = int(inference_vllm_config["port"])
    vllm_env = inference_vllm_config.pop("env_vars", None)
    vllm_cmd = build_vllm_command(inference_vllm_config)

    env_server_proc = None
    env_port = int(env_server_config["port"])
    env_url = f"http://127.0.0.1:{env_port}"
    env_server_cmd = build_env_server_command(cfg.env.name, env_server_config)

    if extra_args.dry_run:
        logger.info("[Dry Run] vLLM command: " + " ".join(vllm_cmd))
        logger.info("[Dry Run] Env Server command: " + " ".join(env_server_cmd))
        logger.info(f"[Dry Run] Eval config: {eval_config}")
        return

    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = log_dir / f"{cfg.exp_name}_vllm.log"
    vllm_log.unlink(missing_ok=True)

    vllm_session = ColabSession(**inference_infra_config)

    try:
        if not wait_for_http_ready(url=f"{env_url}/health", timeout=1):
            logger.info(f"Starting local environment server on port {env_port}...")
            env_server_proc = subprocess.Popen(
                env_server_cmd,
                # stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not wait_for_http_ready(url=f"{env_url}/health", timeout=30):
                raise TimeoutError(f"Local env server at {env_url}/health failed to start.")

        with ProcessGroup() as pg:
            vllm_proc = vllm_session.Popen(
                cmd=vllm_cmd,
                log_path=vllm_log,
                dir=sync_dirs,
                work_dir=remote_exp_dir,
                env=vllm_env,
            )
            pg.add("vllm", vllm_proc, session=vllm_session)

            if not wait_for_http_ready(url=f"http://127.0.0.1:{vllm_port}/health", timeout=300):
                raise TimeoutError(f"vLLM server at http://127.0.0.1:{vllm_port}/health failed to start.")

            env_client = make_env(cfg.env.name, env_client_dict)
            evaluator = AsyncVLLMCompletions(
                vllm_server=f"http://127.0.0.1:{vllm_port}",
                model_name=model_name,
                env_client=env_client,
                **eval_config,
            )
            await evaluator.run()
            logger.info(f"Finished benchmark: saved={evaluator.jsonl_path}")
    finally:
        if env_server_proc is not None:
            logger.info("Stopping local environment server...")
            env_server_proc.terminate()
            env_server_proc.wait(timeout=5)


if __name__ == "__main__":
    sws_run_async(main, forward_extras=True)
