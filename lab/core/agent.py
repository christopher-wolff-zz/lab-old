from abc import ABCMeta, abstractmethod

import numpy as np


class Agent(metaclass=ABCMeta):
    """An abstract base class for agents."""

    def __init__(self) -> None:
        self._seed: int

    @property
    def seed(self) -> int:
        """A seed for all randomness related to the agent.

        Returns:
            The seed of the agent.

        """
        return self._seed

    @abstractmethod
    def begin_episode(self, observation: np.ndarray) -> int:
        """Advance the agent for the first step of an episode.

        Args:
            observation: The initial observation from the environment.

        Returns:
            The first action to take.

        """

    @abstractmethod
    def step(self, reward: float, observation: np.ndarray) -> int:
        """Advance the agent by one step.

        Args:
            observation: An observation from the environment.
            reward: A reward from the environment.

        Returns:
            An action to take.

        """

    @abstractmethod
    def end_episode(self, reward: float) -> None:
        """Advance the agent for the last step of an episode.

        Args:
            reward: The final reward from the environment.

        """
