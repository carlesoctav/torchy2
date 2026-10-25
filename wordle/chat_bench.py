#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

import sws
from env import make_env
from torchy_core import AsyncChatCompletions, build_env_server_command
from torchy_core.infra import wait_for_http_ready
from torchy_core.sws_utils import load_config_builder, sws_run_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chat_bench")


def get_config() -> sws.Config:
    return load_config_builder(str(exp_dir / "configs/bench_deepseek.py"))


async def main(cfg: sws.FinalConfig, unused_argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run GPU-free Chat Completions Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print config and endpoints without running")
    extra_args, _ = parser.parse_known_args(unused_argv)

    endpoints = cfg.inference.endpoints
    model_name = cfg.inference.model
    headers = cfg.inference.headers.to_dict()
    extra_payload = cfg.inference.extra_payload.to_dict() if "extra_payload" in cfg.inference else {}

    env_server_config = cfg.env.server.to_dict()
    env_client_dict = cfg.env.client.to_dict()
    eval_config = cfg.eval.to_dict()

    env_server_proc = None
    env_port = int(env_server_config["port"])
    env_url = f"http://127.0.0.1:{env_port}"

    if extra_args.dry_run:
        logger.info(f"[Dry Run] Endpoints: {endpoints}")
        logger.info(f"[Dry Run] Model: {model_name}")
        logger.info(f"[Dry Run] Headers: {headers}")
        logger.info(f"[Dry Run] Extra Payload: {extra_payload}")
        logger.info(f"[Dry Run] Eval config: {eval_config}")
        logger.info("[Dry Run] Env Server command: " + " ".join(build_env_server_command(cfg.env.name, env_server_config)))
        return

    try:
        if not wait_for_http_ready(url=f"{env_url}/health", timeout=1):
            logger.info(f"Starting local environment server on port {env_port}...")
            env_server_cmd = build_env_server_command(cfg.env.name, env_server_config)
            env_server_proc = subprocess.Popen(env_server_cmd)
            if not wait_for_http_ready(url=f"{env_url}/health", timeout=30):
                raise TimeoutError(f"Local env server at {env_url}/health failed to start.")

        env_client = make_env(cfg.env.name, env_client_dict)
        evaluator = AsyncChatCompletions(
            endpoints=endpoints,
            model_name=model_name,
            env_client=env_client,
            headers=headers,
            extra_payload=extra_payload,
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
