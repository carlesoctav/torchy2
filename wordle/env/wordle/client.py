from __future__ import annotations

import functools
import random
from typing import Any

from datasets import Dataset
from textarena_env import TextArenaAction, TextArenaEnv as OpenEnvTextArenaEnvClient

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


class TRLWordleEnv:
    def __init__(self, env_url: str = DEFAULT_ENV_URL):
        self.env_url = env_url
        self._client = OpenEnvTextArenaEnvClient(base_url=self.env_url).sync()
        self._last_full_feedback = ""
        self.obv_reward = None
        self.reward_signals = None
        self.done = False

    def reset(self, **kwargs: Any) -> str | None:
        seed = kwargs.get("seed")
        result = self._client.reset(seed=int(seed)) if seed is not None else self._client.reset()
        self._last_full_feedback = (
            result.observation.messages[0].content
            if result.observation.messages
            else result.observation.prompt
        )
        self.obv_reward = None
        self.reward_signals = None
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
            if self.obv_reward is None:
                return "game is over. too much making mistake"
            else:
                final_feedback = "you win" if self.obv_reward == 1.0 else "you lose"
                return f"game is over. {final_feedback}"

        result = self._client.step(TextArenaAction(message=guess))
        full_feedback = (
            result.observation.messages[0].content
            if result.observation.messages
            else result.observation.prompt
        )
        feedback = full_feedback[len(self._last_full_feedback) :]
        self._last_full_feedback = full_feedback
        if "Feedback:" not in feedback and result.reward != 1.0:
            self.obv_reward = None
            self.reward_signals = None
        else:
            self.obv_reward = float(result.reward)
            self.reward_signals = result.observation.info["reward_signals"]
        self.done = bool(result.done)
        return feedback

    def get_reward(self) -> float:
        if self.obv_reward is None:
            return 0.0

        if self.obv_reward != 1.0:
            return self.reward_signals["wordle.greens"] * 5 * 0.05 + self.reward_signals["wordle.yellows"] * 5 * 0.025
        else:
            return 1.0


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
