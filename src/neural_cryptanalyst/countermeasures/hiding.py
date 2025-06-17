"""Hiding-based countermeasures."""

import numpy as np


class HidingCountermeasure:
    """Operations for temporal hiding."""

    def add_random_delays(self, traces: np.ndarray, max_delay: int = 100) -> np.ndarray:
        """Insert random delays between operations.

        Parameters
        ----------
        traces : ndarray
            Power traces to modify.
        max_delay : int, default=100
            Maximum delay (number of samples) to insert.

        Returns
        -------
        ndarray
            Traces with random delays inserted.
        """

        augmented = []
        for trace in traces:
            delay = np.random.randint(0, max_delay)
            augmented_trace = np.concatenate([
                trace,
                np.zeros(delay, dtype=trace.dtype)
            ])
            augmented.append(augmented_trace)
        return np.array(augmented)

