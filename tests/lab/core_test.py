import gym

from lab.agents.q_learning_agent import QLearningAgent
from lab.core.environment import Environment
from lab.core.experiment import Experiment


if __name__ == '__main__':
    frozen_lake = gym.make('FrozenLake-v0')
    random_agent = QLearningAgent(action_space_n=frozen_lake.action_space.n,
                                  observation_space_n=frozen_lake.observation_space.n)
    experiment = Experiment(environment=frozen_lake,
                            agent=random_agent,
                            num_iterations=100,
                            training_steps=1000,
                            evaluation_steps=100,
                            max_steps_per_episode=100)
    experiment.run()
