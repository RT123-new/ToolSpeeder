"""Reproducible synthetic benchmark for tool-call latency mechanisms.

Run:
    python tool_speed_lab.py --out results --trials 120000 --seed 20260825

This is a mechanism sanity-check. Replace the synthetic latency samplers with
instrumented real model/tool adapters before drawing production conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LatencyProfile:
    model_decision_ms: float = 450.0
    model_final_ms: float = 300.0
    tool_ms: float = 600.0
    draft_model_ms: float = 70.0
    program_runtime_overhead_ms: float = 80.0
    cache_lookup_ms: float = 8.0
    sigma: float = 0.45


def samples(rng, median_ms, sigma, shape):
    return rng.lognormal(np.log(median_ms), sigma, shape)


def summary(baseline, candidate):
    b50, c50 = np.percentile(baseline, 50), np.percentile(candidate, 50)
    b95, c95 = np.percentile(baseline, 95), np.percentile(candidate, 95)
    return {
        "baseline_p50_ms": float(b50),
        "candidate_p50_ms": float(c50),
        "p50_speedup": float(b50 / c50),
        "baseline_p95_ms": float(b95),
        "candidate_p95_ms": float(c95),
        "p95_speedup": float(b95 / c95),
    }


def parallel(profile, n, seed):
    rows = []
    for calls in (2, 4, 8, 16):
        rng = np.random.default_rng(seed + calls)
        decision = samples(rng, profile.model_decision_ms, profile.sigma, n)
        final = samples(rng, profile.model_final_ms, profile.sigma, n)
        tool = samples(rng, profile.tool_ms, profile.sigma, (n, calls))
        baseline = decision + tool.sum(axis=1) + final
        candidate = decision + tool.max(axis=1) + final
        row = {"independent_calls": calls}
        row.update(summary(baseline, candidate))
        rows.append(row)
    return pd.DataFrame(rows)


def fusion(profile, n, seed):
    rows = []
    for steps in (2, 4, 8, 16):
        rng = np.random.default_rng(seed + 100 + steps)
        decisions = samples(
            rng, profile.model_decision_ms, profile.sigma, (n, steps)
        )
        final = samples(rng, profile.model_final_ms, profile.sigma, n)
        tools = samples(rng, profile.tool_ms, profile.sigma, (n, steps))
        overhead = samples(
            rng, profile.program_runtime_overhead_ms, profile.sigma / 2, n
        )
        baseline = decisions.sum(axis=1) + tools.sum(axis=1) + final
        candidate = decisions[:, 0] + overhead + tools.sum(axis=1) + final
        row = {"dependent_steps": steps}
        row.update(summary(baseline, candidate))
        rows.append(row)
    return pd.DataFrame(rows)


def speculation(profile, n, seed):
    rows = []
    for mode_index, mode in enumerate(
        ("no_contention", "cancellable", "single_slot")
    ):
        for accuracy in np.linspace(0.0, 1.0, 21):
            rng = np.random.default_rng(
                seed + 1000 + mode_index * 100 + int(accuracy * 1000)
            )
            model = samples(rng, profile.model_decision_ms, profile.sigma, n)
            tool = samples(rng, profile.tool_ms, profile.sigma, n)
            wrong_tool = samples(rng, profile.tool_ms, profile.sigma, n)
            draft = samples(
                rng, profile.draft_model_ms, profile.sigma / 2, n
            )
            correct = rng.random(n) < accuracy
            baseline = model + tool
            correct_latency = np.maximum(model, draft + tool)

            if mode == "no_contention":
                wrong_latency = model + tool
            elif mode == "cancellable":
                occupied_until = np.minimum(
                    draft + wrong_tool, model + 30.0
                )
                wrong_latency = np.maximum(model, occupied_until) + tool
            else:
                occupied_until = draft + wrong_tool
                wrong_latency = np.maximum(model, occupied_until) + tool

            candidate = np.where(correct, correct_latency, wrong_latency)
            row = {
                "contention_mode": mode,
                "prediction_accuracy": float(accuracy),
                "mean_tool_cost_multiplier": float(2.0 - accuracy),
                "wasted_call_rate": float(1.0 - accuracy),
            }
            row.update(summary(baseline, candidate))
            rows.append(row)
    return pd.DataFrame(rows)


def cache(profile, n, seed):
    rows = []
    for stale_on_hit in (0.0, 0.001, 0.01):
        for hit_rate in np.linspace(0.0, 1.0, 11):
            rng = np.random.default_rng(
                seed
                + 2000
                + int(stale_on_hit * 1_000_000)
                + int(hit_rate * 1000)
            )
            model = samples(rng, profile.model_decision_ms, profile.sigma, n)
            tool = samples(rng, profile.tool_ms, profile.sigma, n)
            hit = rng.random(n) < hit_rate
            stale = hit & (rng.random(n) < stale_on_hit)
            baseline = model + tool
            candidate = model + np.where(
                hit, profile.cache_lookup_ms, tool
            )
            row = {
                "cache_hit_rate": float(hit_rate),
                "stale_probability_on_hit": float(stale_on_hit),
                "simulated_correctness": float(1.0 - stale.mean()),
            }
            row.update(summary(baseline, candidate))
            rows.append(row)
    return pd.DataFrame(rows)


def decode_compression():
    rows = []
    for decode_share in (0.10, 0.25, 0.50, 0.80):
        for factor in (2.0, 4.0, 6.0):
            speedup = 1.0 / (
                (1.0 - decode_share) + decode_share / factor
            )
            rows.append(
                {
                    "tool_call_decode_share": decode_share,
                    "decode_acceleration_factor": factor,
                    "end_to_end_speedup_upper_bound": speedup,
                }
            )
    return pd.DataFrame(rows)


def commit_horizon(profile, n, seed):
    rows = []
    for fraction in np.linspace(0.1, 1.0, 10):
        rng = np.random.default_rng(seed + 3000 + int(fraction * 1000))
        generation = samples(
            rng, profile.model_decision_ms, profile.sigma, n
        )
        tool = samples(rng, profile.tool_ms, profile.sigma, n)
        baseline = generation + tool
        candidate = np.maximum(
            generation, fraction * generation + tool
        )
        row = {"commit_fraction": float(fraction)}
        row.update(summary(baseline, candidate))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--trials", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.trials < 1000:
        raise SystemExit("--trials must be at least 1000")

    args.out.mkdir(parents=True, exist_ok=True)
    profile = LatencyProfile()

    outputs = {
        "parallel_fanout.csv": parallel(profile, args.trials, args.seed),
        "round_trip_fusion.csv": fusion(profile, args.trials, args.seed),
        "speculation.csv": speculation(profile, args.trials, args.seed),
        "cache.csv": cache(profile, args.trials, args.seed),
        "decode_compression.csv": decode_compression(),
        "commit_horizon.csv": commit_horizon(
            profile, args.trials, args.seed
        ),
    }

    for name, frame in outputs.items():
        frame.to_csv(args.out / name, index=False)

    spec = outputs["speculation.csv"]
    plt.figure(figsize=(8, 5))
    for mode, group in spec.groupby("contention_mode"):
        plt.plot(
            group["prediction_accuracy"],
            group["p50_speedup"],
            marker="o",
            label=mode,
        )
    plt.axhline(1.0, linewidth=1)
    plt.xlabel("Prediction accuracy")
    plt.ylabel("P50 latency speedup")
    plt.title("Speculative tool execution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "speculation_speedup.png", dpi=180)
    plt.close()

    par = outputs["parallel_fanout.csv"]
    plt.figure(figsize=(8, 5))
    plt.plot(
        par["independent_calls"], par["p50_speedup"], marker="o", label="P50"
    )
    plt.plot(
        par["independent_calls"], par["p95_speedup"], marker="o", label="P95"
    )
    plt.xlabel("Independent tool calls")
    plt.ylabel("Latency speedup")
    plt.title("Parallel fan-out")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "parallel_fanout_speedup.png", dpi=180)
    plt.close()

    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "profile": asdict(profile),
                "trials": args.trials,
                "seed": args.seed,
                "warning": (
                    "Synthetic mechanism sanity-check only. Replace sampled "
                    "latencies with real instrumented adapters."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
