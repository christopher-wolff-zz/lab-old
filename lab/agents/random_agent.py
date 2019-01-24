import numpy as np

from lab.core import Agent


class RandomAgent(Agent):
    """An agent that chooses actions at random."""
    def __init__(self, num_actions):
        super().__init__()
        self._num_actions = num_actions
        self._rng = np.random

    def seed(self, seed):
        self._rng.seed(seed)

    def act(self):
        return self._rng.randint(self._num_actions)
