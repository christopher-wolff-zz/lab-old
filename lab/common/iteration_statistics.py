"""A class for storing iteration-specific metrics.
"""


class IterationStatistics:
    """A class for storing iteration-specific metrics.

    The internal format is as follows: we maintain a mapping from keys to lists.
    Each list contains all the values corresponding to the given key.

    For example, self.data_lists['train_episode_returns'] might contain the
    per-episode returns achieved during this iteration.

    Attributes:
        data_lists: dict mapping each metric_name (str) to a list of said metric
            across episodes.
    """

    def __init__(self):
        self.data_lists = {}

    def append(self, data_pairs):
        """Add the given values to their corresponding key-indexed lists.

        Args:
            data_pairs: A dictionary of key-value pairs to be recorded.
        """
        for key, value in data_pairs.items():
            if key not in self.data_lists:
                self.data_lists[key] = []
            self.data_lists[key].append(value)
