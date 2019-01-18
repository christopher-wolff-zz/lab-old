class Agent(object):
    """An abstract base class for agents."""

    def __init__(self):
        self._seed = None

    @property
    def seed(self, seed):
        return self._seed

    def begin_episode(self, observation):
        raise NotImplementedError('Must be implemented by subclass')

    def step(self, reward, observation):
        raise NotImplementedError('Must be implemented by subclass')

    def end_episode(self, reward):
        raise NotImplementedError('Must be implemented by subclass')
