"""Utilities for plotting."""

import numpy as np


def smooth(x, window_length, method='mirror'):
    """Smooth an array of time series data.

    We use a moving average filter and extend the boundary values of `x` so that
    the result has the same length as `x`.

    Args:
        x: (ndarray) A one-dimensional array to be smoothed.
        window_length: (int) The length of the smoothing window.
        method: (str) The method with which to replicate boundary points before
            smoothing. Either 'mirror' or 'same'.

    Returns:
        (ndarray) The smoothed array.

    """
    num_left_vals = window_length // 2
    num_right_vals = window_length // 2
    if window_length % 2 == 0:
        num_left_vals -= 1
    if method == 'same':
        x_left = np.ones(num_left_vals) * x[0]
        x_right = np.ones(num_right_vals) * x[-1]
    elif method == 'mirror':
        x_left = x[num_left_vals-1::-1]
        x_right = x[-num_right_vals:]
    else:
        raise ValueError(f'Unknown method: {method}')
    x_extended = np.r_[x_left, x, x_right]
    window = np.ones(window_length, 'd') / window_length
    return np.convolve(x_extended, window, mode='valid')
