import numpy as np

from lab.core import Agent


class QLearningAgent(Agent):
    """An agent that uses tabular Q-learning with TD updates.

    This implementation only works if the observation space is discrete and
    fully observable. We assume that each observation we receive represents the
    complete environment state.

    The agent uses an epsilon-greedy policy, which means that it mixes between
    choosing the action it thinks is best and a random action.

    """

    def __init__(self, num_actions, num_states, learning_rate=0.5,
                 exploration_rate=0.1, discount_factor=0.99):
        """Initialize a Q-learning agent.

        Args:
            num_actions: (int) The size of the action space.
            num_states: (int) The size of the state space.
            learning_rate: (float) The learning rate used in TD updates.
            exploration_rate: (float) The fraction of steps during which to
                explore the environment. This is the epsilon used in the
                epsilon-greedy policy.
            discount_factor: (float) The factor by which to discount future
                rewards internally. Used for TD updates.

        """
        super().__init__()

        self._num_actions = num_actions
        self._num_states = num_states

        self._learning_rate = learning_rate
        self._exploration_rate = exploration_rate
        self._discount_factor = discount_factor

        self._last_state = None
        self._rng = np.random
        self._Q = np.zeros((self._num_states, self._num_actions))

    def seed(self, seed):
        """See base class."""
        self._rng.seed(seed)

    def act(self):
        """See base class."""
        state = self._last_state
        if self.eval_mode:
            action = self._choose_greedy_action(state)
        else:
            action = self._choose_epsilon_greedy_action(state)
        self._last_action = action
        return action

    def _choose_greedy_action(self, state):
        return np.argmax(self._Q[state])

    def _choose_epsilon_greedy_action(self, state):
        if self._rng.random() < self._exploration_rate:
            return self._rng.randint(self._num_actions)
        else:
            return self._choose_greedy_action(state)

    def begin_episode(self, observation):
        """See base class."""
        self._last_state = observation

    def learn(self, reward, observation):
        """See base class.

        We update the Q-table using a TD update.

        """
        state = self._last_state
        action = self._last_action
        next_state = observation

        # TD update
        target = reward + self._discount_factor * np.max(self._Q[next_state])
        delta =  target - self._Q[state, action]
        self._Q[state, action] += self._learning_rate * delta

        self._last_state = observation
