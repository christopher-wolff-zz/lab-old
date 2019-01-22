"""A brief, non-comprehensive test of the experiment framework."""

import gym

from lab.agents import QLearningAgent
from lab.core import Experiment


if __name__ == '__main__':
    frozen_lake = gym.make('FrozenLake-v0')
    random_agent = QLearningAgent(action_space_n=frozen_lake.action_space.n,
                                  observation_space_n=frozen_lake.observation_space.n,
                                  learning_rate=0.5,
                                  discount_factor=0.99)
    experiment = Experiment(environment=frozen_lake,
                            agent=random_agent,
                            num_iterations=100,
                            training_steps=1000,
                            evaluation_steps=100,
                            max_steps_per_episode=100)
    experiment.run()
