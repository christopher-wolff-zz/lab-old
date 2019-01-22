from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np


class Environment(metaclass=ABCMeta):
    """An abstract base class for environments."""

    def __init__(self) -> None:
        self._seed = None

    @property
    def seed(self) -> int:
        """A seed for all randomness related to the environment.

        Returns:
            The seed of the environment.

        """
        return self._seed

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Advance the environment by one step.

        Args:
            action: The action chosen by the agent.

        Returns:
            The next observation for the agent.
            The reward for the agents action.
            An indicator for whether the episode has ended.

        """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment.

        Returns:
            The initial observation of a new episode.

        """
