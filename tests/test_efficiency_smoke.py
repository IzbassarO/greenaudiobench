"""M5 accounting (C5): clip divisor, idle subtraction, latency stats — no GPU."""

import numpy as np
import pytest

from gab.efficiency import (
    PowerTrace, energy_metrics, latency_stats, run_measured_window,
)


def test_energy_integral_trapezoid():
    # constant 100 W over 10 s -> 1000 J
    trace = PowerTrace(times_s=[0.0, 5.0, 10.0], watts=[100.0, 100.0, 100.0])
    assert trace.energy_j() == pytest.approx(1000.0)
    # linear ramp 0->100 W over 10 s -> 500 J
    ramp = PowerTrace(times_s=[0.0, 10.0], watts=[0.0, 100.0])
    assert ramp.energy_j() == pytest.approx(500.0)


def test_j_per_clip_divisor_uses_actual_clips_not_iterations():
    # batch 32, 7 completed batches -> 224 clips (iterations would be 7)
    trace = PowerTrace(times_s=[0.0, 30.0], watts=[70.0, 70.0])  # 2100 J
    m = energy_metrics(trace, clips_processed=224, window_s=30.0, idle_watts=10.0)
    assert m["clips_processed"] == 224
    assert m["j_per_clip_total"] == pytest.approx(2100.0 / 224)
    assert m["j_per_clip_total"] != pytest.approx(2100.0 / 7)  # iteration divisor is wrong


def test_dynamic_energy_subtracts_idle_baseline():
    trace = PowerTrace(times_s=[0.0, 30.0], watts=[70.0, 70.0])  # 2100 J total
    m = energy_metrics(trace, clips_processed=100, window_s=30.0, idle_watts=10.0)
    # dynamic = (2100 - 10*30) / 100 = 18 J/clip
    assert m["j_per_clip_dynamic"] == pytest.approx(18.0)
    assert m["idle_power_w"] == 10.0


def test_run_measured_window_counts_completed_batches():
    fake_time = iter(np.arange(0.0, 100.0, 0.5))  # each call advances 0.5 s
    clock = {"t": 0.0}

    def time_fn():
        clock["t"] = next(fake_time)
        return clock["t"]

    calls = {"n": 0}

    def work():
        calls["n"] += 1

    clips, window = run_measured_window(work, batch_size=32, min_window_s=3.0,
                                        time_fn=time_fn)
    assert clips == calls["n"] * 32  # divisor material: clips, not iterations
    assert window >= 3.0


def test_latency_stats_median_and_iqr():
    timings = [0.010, 0.020, 0.030, 0.040, 0.050]  # seconds per batch of 10
    s = latency_stats(timings, batch_size=10)
    assert s["latency_ms_per_clip_median"] == pytest.approx(3.0)
    assert s["latency_ms_per_clip_p25"] == pytest.approx(2.0)
    assert s["latency_ms_per_clip_p75"] == pytest.approx(4.0)
    assert s["n_timed_iterations"] == 5


def test_energy_requires_enough_samples():
    with pytest.raises(ValueError, match="samples"):
        PowerTrace(times_s=[0.0], watts=[50.0]).energy_j()
    with pytest.raises(ValueError, match="positive"):
        energy_metrics(PowerTrace([0.0, 1.0], [1.0, 1.0]), 0, 1.0, 0.0)
