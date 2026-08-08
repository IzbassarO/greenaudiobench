"""Latency and energy measurement on GPU (Colab T4). Implemented at M5.

Planned interface:
    measure_latency(model, input_shape, batch_sizes=(1, 32)) -> dict
    - 20 warm-up iters, 100 timed iters, torch.cuda.synchronize(), 3 repeats
    measure_energy(...) -> dict
    - background pynvml power-sampling thread at 10 Hz, energy = integral(power dt) / clips
    - cross-checked with codecarbon session total
GPU-only: must never report CPU timings as GPU results (hard rule #6).
"""
