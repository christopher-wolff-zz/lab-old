import numpy as np

from lab import Agent


class QLearningAgent(Agent):
    """An abstract base class for agents."""

    def __init__(self,
                 action_space_n,
                 observation_space_n,
                 alpha=0.5,  # learning rate
                 gamma=0.99,  # discount factor
                 epsilon_i=1,  # initial exploration rate
                 epsilon_f=0.05,  # final exploration rate
                 epsilon_n=50000):  # number of steps to decay exploration rate
        super()

        self._action_space_n = action_space_n
        self._observation_space_n = observation_space_n

        self._alpha = alpha
        self._gamma = gamma
        self._epsilon_i = epsilon_i
        self._epsilon_f = epsilon_f
        self._epsilon_n = epsilon_n

        self._step_count = 0
        self._last_observation = None

        # Initialize empty Q table
        self._q = np.zeros((self._observation_space_n, self._action_space_n))

    def _choose_action(self, observation, eval_mode=False):
        """Choose an action.

        Args:
            eval_mode: boolean, whether the experiment is currently in
                evaluation mode.

        Returns:
            int, the chosen action.
        """
        if eval_mode:
            return self._choose_greedy_action(observation)
        else:
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
        # Compute exploration rate
        if self._step_count < self._epsilon_n:
            epsilon = self._epsilon_i - self._step_count * \
                    (self._epsilon_i - self._epsilon_f) / self._epsilon_n
        else:
            epsilon = epsilon_f

        # Randomly decide which type of action to take
        if np.random.random() < epsilon:
            return np.random.randint(self._action_space_n)
        else:
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
        target = reward + self._gamma * np.max(self._q[observation])
        self._q[self._last_observation, action] += self._alpha * \
                (target - self._q[self._last_observation, action])

        # Update statistics
        self._step_count += 1
        self._last_observation = observation

        return action

    def end_episode(self, reward):
        self._last_observation = None
