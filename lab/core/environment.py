from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    """An abstract base class for environments."""

    @abstractmethod
    def seed(self, seed):
        """Seed the random number generators of the environment.

        Args:
            seed: (int) The seed to use.

        """

    @abstractmethod
    def step(self, action):
        """Advance the environment by one step.

        Args:
            action: (int) The action chosen by the agent.

        Returns:
            (ndarray) The next observation for the agent.
            (float) The reward for the agents action.
            (bool) An indicator for whether the episode has ended (bool).
            (dict) Diagnostic information for debugging.

        """

    @abstractmethod
    def reset(self):
        """Reset the environment.

        Returns:
            (ndarray) The initial observation of a new episode.

        """
