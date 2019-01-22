import numpy as np

from lab import Agent


class RandomAgent(Agent):
    """An agent that chooses actions at random."""

    def __init__(self, num_actions):
        super()
        self._num_actions = num_actions

    def _choose_action(self):
        return np.random.randint(self._num_actions)

    def begin_episode(self, observation):
        return self._choose_action()

    def step(self, reward, observation):
        return self._choose_action()

    def end_episode(self, reward):
        pass
