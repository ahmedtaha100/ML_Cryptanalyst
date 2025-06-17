"""Oscilloscope interfaces."""

import numpy as np


class OscilloscopeInterface:
    """Base class for oscilloscope interfaces."""

    def capture_traces(self, num_traces: int, trigger_settings: dict):
        """Capture traces from the oscilloscope."""
        raise NotImplementedError("Subclass must implement")


class MockOscilloscope(OscilloscopeInterface):
    """Mock oscilloscope for testing purposes."""

    def capture_traces(self, num_traces: int, trigger_settings: dict):
        """Return synthetic traces for testing."""
        return np.random.randn(num_traces, 10000)

