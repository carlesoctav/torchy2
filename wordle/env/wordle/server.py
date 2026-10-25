from __future__ import annotations

import argparse
import functools
import os
from uuid import uuid4

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import nltk

nltk.download("words", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from textarena.envs.Wordle.env import WordleEnv

# Pre-cache word list and dictionary once at module load to avoid 1.5s+ POS-tagging per session.
_warmup_wordle = WordleEnv()
_WORDLE_WORD_LIST = _warmup_wordle.word_list
_WORDLE_DICTIONARY = _warmup_wordle.dictionary


def _cached_wordle_init(self, word_length: int = 5, num_guesses: int = 6, hardcore: bool = False) -> None:
    super(WordleEnv, self).__init__()
    self.word_length = word_length
    self.num_guesses = num_guesses
    self.word_list = _WORDLE_WORD_LIST
    self.dictionary = _WORDLE_DICTIONARY


WordleEnv.__init__ = _cached_wordle_init

import uvicorn
from openenv.core.env_server.http_server import create_app
from textarena_env.models import TextArenaAction, TextArenaObservation
from textarena_env.server.environment import TextArenaEnvironment
from textarena_env.server.gradio_ui import build_textarena_gradio_app


class OpenEnvTextArenaEnvServer(TextArenaEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *args,
        error_allowance: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.error_allowance = error_allowance

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

        base_env = self._ta_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        base_env.state.error_allowance = self.error_allowance

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

    def step(self, action: TextArenaAction) -> TextArenaObservation:
        obs = super().step(action)
        game_info = self._ta_env.state.game_info.get(0, {})
        if game_info.get("invalid_move"):
            obs.done = True
            obs.reward = -1.0
            obs.info["invalid_move"] = True
            self._ta_env.state.rewards = {0: -1.0}
            self._ta_env.state.done = True
            reason = game_info.get("reason", "Invalid move.")
            game_over_msg = (
                f"\n[GAME] You attempted an invalid move. Reason: {reason}\n"
                "[GAME] Game is over."
            )
            if obs.messages:
                obs.messages[0].content += game_over_msg
            else:
                obs.prompt += game_over_msg
        return obs


def create_textarena_environment(error_allowance: int = 0) -> OpenEnvTextArenaEnvServer:
    return OpenEnvTextArenaEnvServer(
        env_id="Wordle-v0",
        num_players=1,
        max_turns=None,
        download_nltk=False,
        error_allowance=error_allowance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--max-sessions", type=int, default=256, help="Maximum concurrent environment sessions")
    parser.add_argument("--error-allowance", type=int, default=0, help="Error allowance before game termination")
    parser.add_argument("--enable-web-interface", action=argparse.BooleanOptionalAction, default=True, help="Enable Gradio web UI")
    args, _ = parser.parse_known_args()

    gradio_builder = build_textarena_gradio_app if args.enable_web_interface else None

    app = create_app(
        functools.partial(create_textarena_environment, error_allowance=args.error_allowance),
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
