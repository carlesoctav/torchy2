from __future__ import annotations

import functools
import random
from typing import Any

from datasets import Dataset
from textarena_env import TextArenaAction, TextArenaEnv

DEFAULT_ENV_URL = "http://127.0.0.1:8001"

PROMPT = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.
Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
"""


class WordleEnv:
    def __init__(self, env_url: str = DEFAULT_ENV_URL):
        self.env_url = env_url
        self._client = None
        self._last_full_feedback = ""
        self.reward = 0.0
        self.done = False

    def _ensure_client(self):
        if self._client is None:
            self._client = TextArenaEnv(base_url=self.env_url).sync()
        return self._client

    def reset(self, **kwargs: Any) -> str | None:
        seed = kwargs.get("seed")
        client = self._ensure_client()
        result = client.reset(seed=int(seed)) if seed is not None else client.reset()
        self._last_full_feedback = (
            result.observation.messages[0].content
            if result.observation.messages
            else result.observation.prompt
        )
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        """Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        client = self._ensure_client()
        result = client.step(TextArenaAction(message=guess))
        full_feedback = (
            result.observation.messages[0].content
            if result.observation.messages
            else result.observation.prompt
        )
        feedback = full_feedback[len(self._last_full_feedback) :]
        self._last_full_feedback = full_feedback
        if "You attempted an invalid move" in feedback:
            self.reward = 0.0
        else:
            self.reward = float(result.reward or 0.0)
        self.done = bool(result.done)
        return feedback

    def get_reward(self) -> float:
        return float(self.reward)


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
        "environment_factory": functools.partial(WordleEnv, env_url=env_url),
    }
