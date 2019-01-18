"""Module defining classes and helper methods for running experiments."""

import logging
import sys
import time

from lab.common import iteration_statistics


logger = logging.getLogger('experiment')
logger.setLevel(logging.DEBUG)


class Experiment(object):
    """An object that handles running experiments.

    An experiment controls interactions between an agent and an environment and
    keeps relevant statistics and logs.
    """

    def __init__(self, agent, environment, num_iterations, training_steps,
                 evaluation_steps, max_steps_per_episode):
        """Initialize the Experiment object.

        Args:
            environment: The environment to test the agent in.
            agent: The agent to act in the experiment.
            num_iterations: int, the number of iterations to run.
            training_steps: int, the number of training steps to perform.
            evaluation_steps: int, the number of evaluation steps to perform.
            max_steps_per_episode: int, the maximum number of steps after which
                an episode terminates.
        """
        self._environment = environment
        self._agent = agent

        self._num_iterations = num_iterations
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps
        self._max_steps_per_episode = max_steps_per_episode

    def run(self):
        """Run a full experiment, spread over multiple iterations."""
        logger.info('Beginning training...')
        for iteration in range(self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            print(statistics)
            print('---')

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        Args:
            iteration: int, the current iteration number.

        Returns:
            A dict containing summary statistics for this iteration.
        """
        logger.info('Starting iteration %d', iteration)
        statistics = iteration_statistics.IterationStatistics()
        num_episodes_train, average_reward_train = self._run_train_phase(statistics)
        num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)
        return statistics.data_lists

    def _run_train_phase(self, statistics):
        """Run one training phase.

        Args:
            statistics: `IterationStatistics` object which records the
            experimental results. Note - This object is modified by this method.

        Returns:
            num_episodes: int, The number of episodes run in this phase.
            average_reward: The average reward generated in this phase.
        """
        self._agent.eval_mode = False
        start_time = time.time()
        number_steps, sum_returns, num_episodes = self._run_one_phase(
            self._training_steps, statistics, 'train')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        time_delta = time.time() - start_time
        logger.info('Average undiscounted return per training episode: %.2f',
                    average_return)
        logger.info('Average training steps per second: %.2f',
                    number_steps / time_delta)
        return num_episodes, average_return

    def _run_eval_phase(self, statistics):
        """Run one evaluation phase.

        Args:
            statistics: `IterationStatistics` object which records the
                experimental results. Note - This object is modified by this
                method.

        Returns:
            num_episodes: int, The number of episodes run in this phase.
            average_reward: float, The average reward generated in this phase.
        """
        self._agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_phase(
                self._evaluation_steps, statistics, 'eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        logger.info('Average undiscounted return per evaluation episode: %.2f',
                        average_return)
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.

        Args:
            min_steps: int, minimum number of steps to generate in this phase.
            statistics: `IterationStatistics` object which records the
                experimental results.
            run_mode_str: str, describes the run mode for this agent.

        Returns:
            Tuple containing the number of steps taken in this phase (int), the
                sum of returns (float), and the number of episodes performed
                (int).
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        while step_count < min_steps:
            episode_length, episode_return = self._run_one_episode()
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            # We use sys.stdout.write instead of logger so as to flush
            # frequently without generating a line break.
            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_length) +
                             'Return: {}\r'.format(episode_return))
            # sys.stdout.flush()
        return step_count, sum_returns, num_episodes

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
            The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

        # Initialize episode
        initial_observation = self._environment.reset()
        action = self._agent.begin_episode(initial_observation)
        is_terminal = False

        # Simulate interactions until we reach a terminal state.
        while True:
            observation, reward, is_terminal, _ = self._environment.step(action)

            total_reward += reward
            step_number += 1

            if is_terminal or step_number == self._max_steps_per_episode:
                break
            else:
                action = self._agent.step(reward, observation)

        # Finalize episode
        self._agent.end_episode(reward)

        return step_number, total_reward
