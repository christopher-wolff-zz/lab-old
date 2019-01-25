import logging
import sys
import time


LOGGER = logging.getLogger('experiment')
LOGGER.setLevel(logging.DEBUG)


class Experiment:
    """An object that handles running experiments.

    An experiment controls interactions between an agent and an environment and
    keeps relevant statistics and logs.

    """

    def __init__(
        self,
        agent,
        environment,
        num_iterations,
        train_steps,
        eval_steps,
        max_steps_per_episode,
        seed=None,
        iteration_callback=None,
        episode_callback=None
    ):
        """Initialize an experiment.

        Args:
            environment: (Environment) The environment to test the agent in.
            agent: (Agent) The agent to act in the experiment.
            num_iterations: (int) The number of iterations to run.
            train_steps: (int) The number of training steps per iteration.
            eval_steps: (int) The number of evaluation steps per iteration.
            max_steps_per_episode: (int) The maximum number of steps after which
                an episode terminates.
            seed: (int) Optional. A seed for the experiment. If possible, this
                fixes all randomness related to the experiment.
            iteration_callback: (func) Optional. A function to be run after
                every iteration.
            episode_callback: (func) Optional. A function to be run after every
                episode.

        """
        self._environment = environment
        self._agent = agent

        self._num_iterations = num_iterations
        self._train_steps = train_steps
        self._eval_steps = eval_steps
        self._max_steps_per_episode = max_steps_per_episode

        self._seed = seed
        if seed is not None:
            self._agent.seed(seed)
            self._environment.seed(seed)

        def do_nothing(*args): pass
        self._iteration_callback = iteration_callback or do_nothing
        self._episode_callback = episode_callback or do_nothing

        self._stats = {}

    def run(self):
        """Run the experiment from start to finish.

        An experiment consists of repeated iterations, each of which has a
        training phase followed by an evaluation phase. During a phase, episodes
        of agent/environment interactions are simulated until a given number of
        steps is reached.

        Returns:
            (dict) Statistics about the experiment.

        """
        self._reset()
        LOGGER.info('Beginning the experiment...')
        for iteration in range(self._num_iterations):
            LOGGER.info('Starting iteration %d', iteration)
            self._run_train_phase()
            self._run_eval_phase()
            self._iteration_callback(self, self._stats)
        return self._stats

    def _reset(self):
        """Reset the experiment.

        Note: We do not reset any of the random number generators related to the
        agent or environment. This ensures that running the experiment multiple
        times in a row does not generate identical outcomes.

        """
        self._stats = {
            'train_average_returns': [],
            'train_episode_counts': [],
            'eval_average_returns': [],
            'eval_episode_counts': []
        }

    def _run_train_phase(self):
        """Run one training phase.

        Returns:
            (int) The number of episodes run in this phase.
            (float) The average return generated in this phase.

        """
        # Prepare phase
        self._agent.eval_mode = False

        # Run phase
        start_time = time.time()
        num_steps, total_return, num_episodes = self._run_one_phase(
            min_steps=self._train_steps,
            run_mode='train'
        )
        time_delta = time.time() - start_time

        # Statistics
        average_return = total_return / num_episodes if num_episodes > 0 else 0.
        self._stats['train_average_returns'].append(average_return)
        self._stats['train_episode_counts'].append(num_episodes)

        # Logging
        LOGGER.info(
            'Average undiscounted return per training episode: %.2f',
            average_return
        )
        LOGGER.info(
            'Average training steps per second: %.2f',
            num_steps / time_delta
        )

    def _run_eval_phase(self):
        """Run one evaluation phase.

        Returns:
            (int) The number of episodes run in this phase.
            (float) The average return generated in this phase.

        """
        # Prepare phase
        self._agent.eval_mode = True

        # Run phase
        _, total_return, num_episodes = self._run_one_phase(
            min_steps=self._eval_steps,
            run_mode='eval'
        )

        # Statistics
        average_return = total_return / num_episodes if num_episodes > 0 else 0.
        self._stats['eval_average_returns'].append(average_return)
        self._stats['eval_episode_counts'].append(num_episodes)

        # Logging
        LOGGER.info(
            'Average undiscounted return per evaluation episode: %.2f',
            average_return
        )

    def _run_one_phase(self, min_steps, run_mode):
        """Runs the agent/environment loop for a desired number of steps.

        When the desired number of steps is reached, the running episode is
        finished before stopping.

        Args:
            min_steps: (int) The minimum number of steps to generate.
            run_mode: (str) The run mode. Either 'train' or 'eval'.

        Returns:
            (int) The number of steps taken.
            (float) The total return accumulated.
            (int) The number of episodes performed.

        """
        step_count = 0
        num_episodes = 0
        total_return = 0.

        while step_count < min_steps:
            episode_length, episode_return = self._run_one_episode()
            self._episode_callback(self, self._stats)
            # TODO: Record episode length and return as statistics
            step_count += episode_length
            total_return += episode_return
            num_episodes += 1
            # We use sys.stdout.write instead of logger so as to flush
            # frequently without generating a line break.
            # sys.stdout.write(
            #     'Steps executed: {} '.format(step_count) +
            #     'Episode length: {} '.format(episode_length) +
            #     'Return: {}\r'.format(episode_return)
            # )
            # sys.stdout.flush()
        return step_count, total_return, num_episodes

    def _run_one_episode(self):
        """Run a single episode of agent/environment interactions.

        An episode ends when either the environment reaches a terminal state or
        a specified maximum number of steps is reached.

        Returns:
            (int) The number of steps taken.
            (float) The total reward.

        """
        initial_observation = self._environment.reset()
        self._agent.begin_episode(initial_observation)

        is_terminal = False
        step_count = 0
        total_reward = 0.

        while True:
            action = self._agent.act()
            observation, reward, is_terminal, _ = self._environment.step(action)
            self._agent.learn(reward, observation)

            total_reward += reward
            step_count += 1

            if is_terminal or step_count == self._max_steps_per_episode:
                break

        self._agent.end_episode()

        return step_count, total_reward
