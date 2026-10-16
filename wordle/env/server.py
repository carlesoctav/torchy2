from __future__ import annotations

import os
import random
from uuid import uuid4

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

import nltk

nltk.download("words", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

import uvicorn
from openenv.core.env_server.http_server import create_app
from textarena_env.models import TextArenaAction, TextArenaObservation
from textarena_env.server.environment import TextArenaEnvironment
from textarena_env.server.gradio_ui import build_textarena_gradio_app
import textarena as ta
from textarena.envs.Wordle.env import WordleEnv

TextArenaEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True


def _wordle_reset(self, num_players: int = 1, seed: int | None = None):
    self.state = ta.SinglePlayerState(num_players=num_players, seed=None)
    rng = random.Random(seed) if seed is not None else random
    game_state = {
        "secret_word": rng.choice(self.word_list),
        "guess_history": [],
        "word_length": self.word_length,
        "num_guesses": self.num_guesses,
    }
    self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)


WordleEnv.reset = _wordle_reset


def _env_reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs):
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


TextArenaEnvironment.reset = _env_reset


def create_textarena_environment():
    env_id = os.getenv("TEXTARENA_ENV_ID", "Wordle-v0")
    num_players = int(os.getenv("TEXTARENA_NUM_PLAYERS", "1"))
    max_turns_env = os.getenv("TEXTARENA_MAX_TURNS")
    max_turns = int(max_turns_env) if max_turns_env is not None else None
    download_nltk = os.getenv("TEXTARENA_DOWNLOAD_NLTK", "1") in {"1", "true", "True"}
    return TextArenaEnvironment(
        env_id=env_id,
        num_players=num_players,
        max_turns=max_turns,
        download_nltk=download_nltk,
    )


app = create_app(
    create_textarena_environment,
    TextArenaAction,
    TextArenaObservation,
    env_name="textarena_env",
    max_concurrent_envs=int(os.getenv("WORDLE_MAX_SESSIONS", "256")),
    gradio_builder=build_textarena_gradio_app,
)


def main() -> None:
    uvicorn.run(
        app,
        host=os.getenv("WORDLE_HOST", "0.0.0.0"),
        port=int(os.getenv("WORDLE_PORT", "8001")),
    )


if __name__ == "__main__":
    main()
