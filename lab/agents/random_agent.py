import numpy as np

from lab.core import Agent


class RandomAgent(Agent):
    """An agent that chooses actions at random."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self._num_actions: int = num_actions

    def _choose_action(self) -> int:
        return np.random.randint(self._num_actions)

    def begin_episode(self, observation: np.ndarray) -> int:
        return self._choose_action()

    def step(self, reward, observation):
        return self._choose_action()

    def end_episode(self, reward):
        pass
