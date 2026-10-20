from __future__ import annotations

import argparse
import os
from uuid import uuid4

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import nltk

nltk.download("words", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

import uvicorn
from openenv.core.env_server.http_server import create_app
from textarena_env.models import TextArenaAction, TextArenaObservation
from textarena_env.server.environment import TextArenaEnvironment
from textarena_env.server.gradio_ui import build_textarena_gradio_app


class OpenEnvTextArenaEnvServer(TextArenaEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> TextArenaObservation:
        env = self._ta_env
        while hasattr(env, "env"):
            if hasattr(env, "full_observations"):
                env.full_observations = {}
            env = env.env
        if hasattr(env, "full_observations"):
            env.full_observations = {}

        self._ta_env.reset(num_players=self.num_players, seed=seed)

        for provider in self._reward_providers:
            provider.reset()

        self._state.episode_id = episode_id if episode_id is not None else str(uuid4())
        self._state.step_count = 0
        self._state.turn = 0
        self._state.last_reward = 0.0
        self._state.last_info = {}
        self._state.raw_state = self._snapshot_state()
        self._last_reward_signals = {}

        observation = self._build_observation()
        observation.reward = 0.0
        observation.done = False

        return observation


def create_textarena_environment() -> OpenEnvTextArenaEnvServer:
    return OpenEnvTextArenaEnvServer(
        env_id="Wordle-v0",
        num_players=1,
        max_turns=None,
        download_nltk=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--max-sessions", type=int, default=256, help="Maximum concurrent environment sessions")
    parser.add_argument("--enable-web-interface", action=argparse.BooleanOptionalAction, default=True, help="Enable Gradio web UI")
    args, _ = parser.parse_known_args()

    gradio_builder = build_textarena_gradio_app if args.enable_web_interface else None

    app = create_app(
        create_textarena_environment,
        TextArenaAction,
        TextArenaObservation,
        env_name="textarena_env",
        max_concurrent_envs=args.max_sessions,
        gradio_builder=gradio_builder,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
