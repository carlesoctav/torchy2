from __future__ import annotations

from typing import Any


def build_vllm_command(vllm_config: dict[str, Any]) -> list[str]:
    model = vllm_config.pop("model")
    cmd = ["uv", "run", "vllm", "serve", str(model)]

    vllm_config.pop("env_vars", None)

    for key, value in vllm_config.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cmd.extend([flag, str(item)])
        else:
            cmd.extend([flag, str(value)])

    return cmd
