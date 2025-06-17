"""Benchmarking utilities for attacks."""

import time

try:  # pragma: no cover - optional dependency
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - psutil is optional
    PSUTIL_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GPU_AVAILABLE = False


class AttackBenchmark:
    """Measure training performance of models."""

    def __init__(self):
        self.results = {}

    def benchmark_model(self, model, traces, labels):
        """Benchmark model performance."""
        start_time = time.time()
        start_memory = 0
        if PSUTIL_AVAILABLE:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        model.compile_model()
        history = model.train(traces, labels, epochs=10)

        train_time = time.time() - start_time
        memory_used = None
        if PSUTIL_AVAILABLE:
            memory_used = (
                psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            )

        gpu_memory = None
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory = gpus[0].memoryUsed

        return {
            "train_time": train_time,
            "memory_mb": memory_used,
            "gpu_memory_mb": gpu_memory,
            "final_accuracy": history.history.get("accuracy", [0])[-1],
        }

