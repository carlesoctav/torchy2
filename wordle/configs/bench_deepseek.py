from __future__ import annotations

import os
from pathlib import Path

import sws

exp_dir = Path(__file__).resolve().parent.parent


def get_config() -> sws.Config:
    config = sws.Config()

    config.exp_name = "deepseek-v4.1-wordle-bench-4"
    config.project_name = "bench-wordle"
    config.output_dir = lambda : f"evals/{config.project_name}/{config.exp_name}" 

    api_key = (
        os.environ.get("X_GOOG_API_KEY")

    )

    config.inference.endpoints = (
        "https://aiplatform.googleapis.com/v1/projects/helpful-kingdom-509908-c3/locations/global/endpoints/openapi"
    )
    config.inference.model = "deepseek-ai/deepseek-v4.1-flash-maas"
    config.inference.headers = {
        "X-Goog-Api-Key": api_key,
        "api_key": api_key,
    }
    config.inference.extra_payload = {"thinking": {"type": "enabled"}}

    config.env.name = "wordle"
    config.env.client.env_url = "http://127.0.0.1:8001"
    config.env.client.dataset_size = 1000
    config.env.client.seed = 67

    config.env.server.host = "127.0.0.1"
    config.env.server.port = 8001
    config.env.server.max_sessions = 50
    config.env.server.error_allowance = 3
    config.env.server.enable_web_interface = False

    config.eval.output_dir = lambda: config.output_dir

    config.eval.max_completion_length = 2048
    config.eval.max_tool_calling_iterations = 6+3
    config.eval.max_inflight_tasks = 50

    config.eval.reasoning_effort = "low"
    config.eval.temperature = 1.0
    config.eval.top_p = 0.95

    return config
