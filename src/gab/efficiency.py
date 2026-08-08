"""Latency + NVML energy measurement core (M5, audited protocol C4/C5).

Measurement boundary (C4): canonical waveform batch -> final embedding, i.e.
adapter.embed_batch — model-specific feature extraction is INSIDE the timed
region for every model. Never time bare model.forward().

Energy (C5): NVML board power is authoritative. Fixed window >= 30 s per
configuration; J/clip divides integrated energy by ACTUAL CLIPS PROCESSED
(n_completed_batches * batch_size), never by iteration count. An idle
baseline (>= 30 s) is recorded and both total and idle-subtracted (dynamic)
J/clip are stored. All pieces take injectable time/power functions so the
accounting is unit-testable without a GPU.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass


@dataclass
class PowerTrace:
    times_s: list[float]
    watts: list[float]

    def energy_j(self) -> float:
        """Trapezoidal integral of power over time."""
        if len(self.times_s) < 2:
            raise ValueError("need >= 2 power samples to integrate energy")
        e = 0.0
        for i in range(1, len(self.times_s)):
            dt = self.times_s[i] - self.times_s[i - 1]
            e += 0.5 * (self.watts[i] + self.watts[i - 1]) * dt
        return e


def latency_stats(per_iter_seconds: list[float], batch_size: int) -> dict:
    """Median + IQR (p25/p75) of ms/clip over ALL timed iterations."""
    if not per_iter_seconds:
        raise ValueError("no timed iterations")
    ms_per_clip = sorted(1000.0 * t / batch_size for t in per_iter_seconds)
    n = len(ms_per_clip)
    return {
        "latency_ms_per_clip_median": statistics.median(ms_per_clip),
        "latency_ms_per_clip_p25": ms_per_clip[max(0, int(0.25 * (n - 1)))],
        "latency_ms_per_clip_p75": ms_per_clip[min(n - 1, int(0.75 * (n - 1)))],
        "n_timed_iterations": n,
    }


def energy_metrics(trace: PowerTrace, clips_processed: int, window_s: float,
                   idle_watts: float) -> dict:
    """J/clip accounting. Divisor is CLIPS, never iterations (audit C5)."""
    if clips_processed <= 0:
        raise ValueError("clips_processed must be positive")
    total_j = trace.energy_j()
    dynamic_j = total_j - idle_watts * window_s
    return {
        "energy_total_j": total_j,
        "clips_processed": clips_processed,
        "energy_window_s": window_s,
        "n_power_samples": len(trace.watts),
        "idle_power_w": idle_watts,
        "j_per_clip_total": total_j / clips_processed,
        "j_per_clip_dynamic": dynamic_j / clips_processed,
    }


class NvmlPowerSampler(threading.Thread):
    """Background 10 Hz NVML board-power sampler for device 0."""

    def __init__(self, hz: float = 10.0):
        super().__init__(daemon=True)
        self.hz = hz
        self.trace = PowerTrace([], [])
        # NOT self._stop: threading.Thread defines a private _stop() method on
        # Python <= 3.12 (Colab), which join() calls internally — shadowing it
        # with an Event raises "TypeError: 'Event' object is not callable"
        self._stop_event = threading.Event()

    def run(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        period = 1.0 / self.hz
        while not self._stop_event.is_set():
            self.trace.times_s.append(time.monotonic())
            self.trace.watts.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            self._stop_event.wait(period)
        pynvml.nvmlShutdown()

    def stop(self) -> PowerTrace:
        self._stop_event.set()
        self.join()
        return self.trace


def run_measured_window(work_fn, batch_size: int, min_window_s: float,
                        time_fn=time.monotonic, sync_fn=lambda: None) -> tuple[int, float]:
    """Run work_fn (one full batch) repeatedly for >= min_window_s wall time.

    Returns (clips_processed, actual_window_s). Only COMPLETED batches count:
    clips = n_completed_batches * batch_size.
    """
    sync_fn()
    t0 = time_fn()
    clips = 0
    while time_fn() - t0 < min_window_s:
        work_fn()
        sync_fn()
        clips += batch_size
    return clips, time_fn() - t0


def measure_idle_power(seconds: float, hz: float = 10.0) -> float:
    """Mean NVML board power over an idle window of `seconds`."""
    sampler = NvmlPowerSampler(hz=hz)
    sampler.start()
    time.sleep(seconds)
    trace = sampler.stop()
    return trace.energy_j() / (trace.times_s[-1] - trace.times_s[0])


def gpu_temperature_c() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        t = pynvml.nvmlDeviceGetTemperature(
            pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return float(t)
    except Exception:
        return None
