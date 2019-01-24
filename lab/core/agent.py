from abc import ABCMeta, abstractmethod

import numpy as np


class Agent(metaclass=ABCMeta):
    """An abstract base class for agents.

    Attributes:
        eval_mode: (bool) Whether the agent is currently in evaluation mode as
            opposed to training mode.

    """

    def __init__(self):
        self.eval_mode = False

    @abstractmethod
    def seed(self, seed):
        """Seed the random number generators of the agent.

        Args:
            seed: (int) The seed to use.

        """

    @abstractmethod
    def act(self):
        """Choose an action.

        Returns:
            (int) The chosen action.

        """

    def learn(self, reward, observation):
        """Learn from the most recent observation and reward.

        This method is called immediately after the environment advances by one
        step and the resulting reward and observation are recorded.

        Args:
            reward: (float) The reward for the previous action.
            observation: (Observation) A new observation of the environment.

        """

    def begin_episode(self, observation):
        """Run a procedure at the beginning of an episode.

        Args:
            observation: (Observation) The initial observation of the episode.

        """

    def end_episode(self):
        """Run a procedure at the end of an episode."""
