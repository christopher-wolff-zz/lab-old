class Environment(object):
    """An abstract base class for environments."""

    def __init__(self):
        self._seed = None

    @property
    def seed(self):
        return self._seed

    def step(self):
        raise NotImplementedError('Must be implemented by subclass')

    def reset(self):
        raise NotImplementedError('Must be implemented by subclass')
