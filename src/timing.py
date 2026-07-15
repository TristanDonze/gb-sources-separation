import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class StageTimer:
    """Collect and log timings while keeping instrumentation out of business logic."""

    def __init__(self, device, logger, prefix, enabled=True):
        self.device = torch.device(device)
        self.logger = logger
        self.prefix = prefix
        self.enabled = enabled
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)
        self._start = time.perf_counter()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _record(self, stage, elapsed):
        self._totals[stage] += elapsed
        self._counts[stage] += 1

    @contextmanager
    def measure(self, stage):
        if not self.enabled:
            yield
            return

        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._record(stage, time.perf_counter() - start)

    def iter_batches(self, dataloader):
        """Measure DataLoader construction and full processing time for each batch."""
        if not self.enabled:
            yield from dataloader
            return

        batch_build_start = time.perf_counter()
        iterator = iter(dataloader)
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                return

            self._record(
                "batch_build",
                time.perf_counter() - batch_build_start,
            )

            batch_process_start = time.perf_counter()
            yield batch
            self._sync()
            self._record(
                "batch_process",
                time.perf_counter() - batch_process_start,
            )
            batch_build_start = time.perf_counter()

    def log(self, renamed_averages=None, extra_averages=None):
        if not self.enabled:
            return

        renamed_averages = renamed_averages or {}
        extra_averages = extra_averages or {}

        self._sync()
        parts = [f"epoch={time.perf_counter() - self._start:.2f}s"]

        for stage, total in self._totals.items():
            label = renamed_averages.get(stage, stage)
            average = total / self._counts[stage]
            parts.append(f"{label}_avg={average:.4f}s")

        for label, (stage, denominator) in extra_averages.items():
            average = self._totals[stage] / denominator if denominator else 0.0
            parts.append(f"{label}_avg={average:.4f}s")

        self.logger.info(f"{self.prefix} timing: " + ", ".join(parts))
