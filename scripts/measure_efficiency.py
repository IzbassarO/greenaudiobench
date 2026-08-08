#!/usr/bin/env python3
"""M5 — latency + NVML energy per model (audited protocol C4/C5/C6).

Boundary (C4): canonical waveform batch -> embedding (adapter.embed_batch),
model-specific feature extraction inside the timed path for every model.

Latency: 20 warmups, 100 timed iterations, torch.cuda.synchronize around
every timed iteration, 3 independent repeats -> median + IQR over all 300.

Energy (C5): NVML board power @10 Hz is authoritative; >=30 s idle baseline,
then a >=30 s measured window per configuration; J/clip divides integrated
energy by ACTUAL clips processed. Both total and idle-subtracted values are
stored. CodeCarbon is NOT authoritative and only runs with --codecarbon into
a clearly-named non-authoritative column.

Official runs (append to results/efficiency.csv): CUDA required, clean tree
(unless --allow-dirty), pinned checkpoints. --smoke: tiny CPU-friendly run,
writes ONLY data/smoke/efficiency_smoke.csv, no NVML needed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gab.audio import fix_duration, load_audio  # noqa: E402
from gab.datasets import SPECS, audio_path, dataset_root, load_meta  # noqa: E402
from gab.efficiency import (  # noqa: E402
    NvmlPowerSampler, energy_metrics, gpu_temperature_c, latency_stats,
    measure_idle_power, run_measured_window,
)
from gab.models.registry import MODEL_ORDER, assert_pinned, load_adapter  # noqa: E402
from gab.utils import assert_official_run, run_metadata, smoke_output_dir  # noqa: E402


def log(msg: str) -> None:
    print(f"[efficiency] {msg}", flush=True)


def make_batches(dataset: str, sample_rate: int, seconds: float,
                 batch_size: int, smoke: bool) -> list[np.ndarray]:
    """First batch_size real clips through the canonical decode path.

    Smoke mode falls back to synthetic noise when the dataset is absent.
    """
    spec = SPECS[dataset]
    try:
        meta = load_meta(spec, ROOT / "data").iloc[:batch_size]
        root = dataset_root(spec, ROOT / "data")
        wavs = [fix_duration(load_audio(audio_path(spec, root, row), sample_rate),
                             sample_rate, seconds)
                for _, row in meta.iterrows()]
    except (FileNotFoundError, OSError):
        if not smoke:
            raise
        rng = np.random.default_rng(0)
        wavs = [(0.05 * rng.standard_normal(int(seconds * sample_rate))
                 ).astype(np.float32) for _ in range(batch_size)]
    while len(wavs) < batch_size:
        wavs.append(wavs[len(wavs) % max(1, len(wavs) - 1)])
    return wavs[:batch_size]


def count_params_millions(adapter) -> float:
    return sum(p.numel() for p in adapter.torch_module().parameters()) / 1e6


def compute_gmacs(adapter) -> tuple[str, str]:
    """fvcore MAC count at batch 1; '' + note when uncountable (never estimated)."""
    import torch
    from fvcore.nn import FlopCountAnalysis

    x, shape_desc = adapter.example_tensor_input(batch_size=1)

    class _Shim(torch.nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.inner = adapter.torch_module()
            self._adapter = adapter

        def forward(self, t):
            return self._adapter._forward_tensor(t)

    try:
        with torch.inference_mode():
            flops = FlopCountAnalysis(_Shim(adapter), x).unsupported_ops_warnings(
                False).uncalled_modules_warnings(False).total()
        return f"{flops / 1e9:.3f}", shape_desc
    except Exception as exc:  # never fabricate a number
        return "", f"{shape_desc} | fvcore failed: {type(exc).__name__}: {exc}"


def measure_config(adapter, wavs, batch_size: int, device: str, smoke: bool,
                   warmup: int, timed: int, repeats: int,
                   min_window_s: float, idle_w: float | None) -> dict:
    import torch

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    for _ in range(warmup):
        adapter.embed_batch(wavs)
    sync()

    timings: list[float] = []
    for _ in range(repeats):
        for _ in range(timed):
            sync()
            t0 = time.monotonic()
            adapter.embed_batch(wavs)
            sync()
            timings.append(time.monotonic() - t0)
    row = latency_stats(timings, batch_size)
    row["n_repeats"] = repeats

    if device == "cuda" and not smoke:
        temp_before = gpu_temperature_c()
        sampler = NvmlPowerSampler(hz=10.0)
        sampler.start()
        clips, window_s = run_measured_window(
            lambda: adapter.embed_batch(wavs), batch_size, min_window_s,
            sync_fn=sync)
        trace = sampler.stop()
        row.update(energy_metrics(trace, clips, window_s, idle_w))
        row["gpu_temp_before_c"] = temp_before
        row["gpu_temp_after_c"] = gpu_temperature_c()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="esc50", choices=sorted(SPECS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER),
                        choices=list(MODEL_ORDER))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"],
                        choices=["fp32", "fp16"])
    parser.add_argument("--min-window-s", type=float, default=30.0)
    parser.add_argument("--idle-s", type=float, default=30.0)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny CPU-friendly run; writes ONLY data/smoke/")
    parser.add_argument("--codecarbon", action="store_true",
                        help="optional NON-authoritative CodeCarbon session total")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    models = [m for m in MODEL_ORDER if m in args.models]
    if args.smoke:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        meta_run = run_metadata()
        warmup, timed, repeats, min_window = 2, 5, 1, 2.0
        log(f"SMOKE MODE on {device}: output ONLY under data/smoke/")
    else:
        assert_pinned(models)
        meta_run = assert_official_run(require_cuda=True,
                                       allow_dirty=args.allow_dirty)
        device = "cuda"
        warmup, timed, repeats, min_window = 20, 100, 3, args.min_window_s
    log(f"run metadata: {meta_run}")

    tracker = None
    if args.codecarbon and not args.smoke:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()

    idle_w = None
    if device == "cuda" and not args.smoke:
        log(f"measuring idle GPU baseline for {args.idle_s:.0f}s ...")
        idle_w = measure_idle_power(args.idle_s)
        log(f"idle power: {idle_w:.2f} W")

    rows = []
    for name in models:
        for dtype in args.dtypes:
            fp16 = dtype == "fp16"
            adapter = load_adapter(name)
            base = {"model": name, "dtype": dtype, "dataset": args.dataset,
                    "smoke": args.smoke}
            try:
                adapter.load(device=device, fp16=fp16)
            except NotImplementedError as exc:
                log(f"{name}/{dtype}: unsupported ({exc})")
                rows.append({**base, "status": "fp16_unsupported", "note": str(exc)})
                continue
            except Exception as exc:
                log(f"{name}/{dtype}: load failed ({type(exc).__name__}: {exc})")
                rows.append({**base, "status": "load_error", "note": str(exc)})
                continue
            params_m = count_params_millions(adapter)
            gmacs, gmacs_note = ("", "skipped in smoke") if args.smoke \
                else compute_gmacs(adapter)
            seconds = float(adapter.info.duration_policy.split(":")[1].rstrip("s"))
            for bs in args.batch_sizes:
                wavs = make_batches(args.dataset, adapter.info.sample_rate,
                                    seconds, bs, args.smoke)
                try:
                    r = measure_config(adapter, wavs, bs, device, args.smoke,
                                       warmup, timed, repeats, min_window, idle_w)
                    status = "ok"
                    note = ""
                except Exception as exc:
                    r, status, note = {}, "measure_error", f"{type(exc).__name__}: {exc}"
                    log(f"{name}/{dtype}/bs{bs}: FAILED {note}")
                rows.append({**base, "batch_size": bs, "status": status,
                             "params_millions": round(params_m, 3),
                             "gmacs_fvcore_b1": gmacs, "gmacs_input": gmacs_note,
                             "note": note, **r})
                if status == "ok":
                    log(f"{name}/{dtype}/bs{bs}: "
                        f"{r['latency_ms_per_clip_median']:.2f} ms/clip median"
                        + (f", {r['j_per_clip_total']:.3f} J/clip total"
                           if "j_per_clip_total" in r else ""))
            del adapter

    df = pd.DataFrame(rows)
    for col, val in meta_run.items():
        df[col] = val
    if tracker is not None:
        df["codecarbon_kwh_nonauthoritative"] = tracker.stop()

    out = (smoke_output_dir() / "efficiency_smoke.csv") if args.smoke \
        else ROOT / "results" / "efficiency.csv"
    header = not out.exists()
    df.to_csv(out, mode="a", header=header, index=False)
    log(f"appended {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
