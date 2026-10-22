from __future__ import annotations

import functools
import random
from typing import Any

from datasets import Dataset
from textarena_env import TextArenaAction, TextArenaEnv as OpenEnvTextArenaEnvClient

DEFAULT_ENV_URL = "http://127.0.0.1:8001"

PROMPT = """You are playing Wordle.
A secret 5-letter English word has been chosen. Your goal is to guess it within 6 attempts.

After each guess, you receive color-coded feedback:
- G (green): Letter is correct and in the correct position
- Y (yellow): Letter exists in the word but in the wrong position
- X (wrong): Letter is not in the word

Rules:
1. All guesses must be valid 5-letter English words.
2. You cannot reuse a word you have already guessed.
3. Use the tool `guess` to make each guess.
"""


class TRLWordleEnv:
    def __init__(self, env_url: str = DEFAULT_ENV_URL):
        self.env_url = env_url
        self._client = OpenEnvTextArenaEnvClient(base_url=self.env_url).sync()
        self._last_full_feedback = ""
        self.won = False
        self._max_partial_reward = 0.0
        self.done = False

    def reset(self, **kwargs: Any) -> str | None:
        seed = kwargs.get("seed")
        result = self._client.reset(seed=int(seed)) if seed is not None else self._client.reset()
        self._last_full_feedback = (
            result.observation.messages[0].content
            if result.observation.messages
            else result.observation.prompt
        )
        self.won = False
        self._max_partial_reward = 0.0
        self.done = False
        return None

    def guess(self, guess: str) -> str:
        """Make a guess in the Wordle game.

        Args:
            guess: The 5-letter word to guess (e.g. 'crane')

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            return "Game is over. No more guesses allowed."

        formatted_guess = guess.strip()
        if not (formatted_guess.startswith("[") and formatted_guess.endswith("]")):
            formatted_guess = f"[{formatted_guess}]"

        result = self._client.step(TextArenaAction(message=formatted_guess))
        full_feedback = (
            result.observation.messages[0].content
            if result.observation.messages
            else result.observation.prompt
        )
        feedback = full_feedback[len(self._last_full_feedback) :]
        self._last_full_feedback = full_feedback

        if result.reward == 1.0:
            self.won = True
            self._max_partial_reward = 1.0
        elif (
            "reward_signals" in result.observation.info
            and "wordle.greens" in result.observation.info["reward_signals"]
        ):
            current_partial = (
                result.observation.info["reward_signals"]["wordle.greens"] * 5 * 0.05
                + result.observation.info["reward_signals"]["wordle.yellows"] * 5 * 0.025
            )
            self._max_partial_reward = max(self._max_partial_reward, current_partial)

        self.done = bool(result.done)
        if self.done:
            if "game is over" not in feedback.lower():
                if self.won:
                    feedback += "\n[GAME] Game is over. You won!"
                elif "turn limit" in feedback.lower():
                    feedback += "\n[GAME] Game is over. Turn limit reached."
                else:
                    feedback += "\n[GAME] Game is over."
        return feedback

    def get_reward(self) -> float:
        if self.won:
            return 1.0
        return self._max_partial_reward


def make(
    prompt: str = PROMPT,
    dataset_size: int = 1000,
    env_url: str = DEFAULT_ENV_URL,
    seed: int | None = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    rng = random.Random(seed) if seed is not None else random
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(dataset_size)]
    train_dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": prompt}] for _ in range(dataset_size)],
            "seed": seeds,
        }
    )
    return {
        "train_dataset": train_dataset,
        "environment_factory": functools.partial(TRLWordleEnv, env_url=env_url),
    }
