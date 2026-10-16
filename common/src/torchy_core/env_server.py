from __future__ import annotations

from typing import Any


def build_env_server_command(env_name: str, server_config: dict[str, Any]) -> list[str]:
    cmd = ["uv", "run", "python", "-m", f"env.{env_name}.server"]

    for key, value in server_config.items():
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
