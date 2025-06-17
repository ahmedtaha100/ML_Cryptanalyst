"""Simple masking countermeasures."""

import numpy as np


class MaskingCountermeasure:
    """Operations for Boolean masking of sensitive data."""

    def apply_boolean_masking(self, data: np.ndarray, mask: np.ndarray | None = None):
        """Apply Boolean masking to data.

        Parameters
        ----------
        data : ndarray
            Data to be masked.
        mask : ndarray, optional
            Mask to apply. If ``None``, a random mask is generated.

        Returns
        -------
        masked : ndarray
            Masked data.
        mask : ndarray
            The mask used.
        """

        if mask is None:
            mask = np.random.randint(0, 256, size=data.shape, dtype=data.dtype)
        return data ^ mask, mask

