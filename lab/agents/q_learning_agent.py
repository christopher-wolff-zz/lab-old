import numpy as np

from lab.core import Agent


class QLearningAgent(Agent):
    """An agent that implements tabular Q-Learning with TD updates."""

    def __init__(self,
                 action_space_n: int,
                 observation_space_n: int,
                 learning_rate: float = 0.5,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 0.1) -> None:
        super().__init__()

        self._action_space_n = action_space_n
        self._observation_space_n = observation_space_n

        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate

        self._step_count = 0
        self._last_observation = None

        # Initialize empty Q table
        self._q = np.zeros((self._observation_space_n, self._action_space_n))

    def _choose_action(self, observation: np.ndarray, eval_mode: bool = False):
        if eval_mode:
            return self._choose_greedy_action(observation)

        return self._choose_epsilon_greedy_action(observation)

    def _choose_greedy_action(self, observation):
        """Choose an action greedily.

        Args:
            observation: int, the newest observation.

        Returns:
            int, the chosen action.

        """
        return np.argmax(self._q[observation])

    def _choose_epsilon_greedy_action(self, observation):
        """Choose an action greedily.

        With probability self._epsilon, we choose a random action. Otherwise, we
        choose a greedy action.

        Args:
            observation: int, the newest observation.

        Returns:
            int, the chosen action.

        """
        if np.random.random() < self._exploration_rate:
            return np.random.randint(self._action_space_n)

        return self._choose_greedy_action(observation)

    def begin_episode(self, observation):
        """Choose the first action at the start of the episode.

        Args:
            observation: int, the initial observation.

        Returns:
            int, the chosen action.

        """
        self._step_count = 1
        self._last_observation = observation
        return self._choose_action(observation)

    def step(self, reward, observation):
        """Update the Q table and choose an action.

        Args:
            reward: float, the reward for the previous action from the
                environment.
            observation: int, the newest observation.

        Returns:
            int, the chosen action.

        """
        # Choose an action
        action = self._choose_action(observation)

        # TD update
        target = reward + self._discount_factor * np.max(self._q[observation])
        self._q[self._last_observation, action] += self._learning_rate * \
                (target - self._q[self._last_observation, action])

        # Update statistics
        self._step_count += 1
        self._last_observation = observation

        return action

    def end_episode(self, reward):
        self._last_observation = None
