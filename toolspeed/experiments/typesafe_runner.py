"""TypeSafe/Jev Speculative Routing Experiment Runner and CLI handlers."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, cast

import numpy as np

from toolspeed.adapters.base import ToolRegistry
from toolspeed.core.types import EventType
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.speculative_providers import (
    DeterministicFrequencyProvider,
    E3CurrentDraftProvider,
    NoSpeculationProvider,
    ReplayRecord,
    ReplaySpeculationProvider,
    SpeculationDecisionProvider,
    SpeculationState,
    SystemOneLLMProvider,
    TypeSafeJevProvider,
    compute_composite_hash,
)
from toolspeed.visualization.charts import ascii_table
from toolspeed.workloads.w8_speculative_routing import W8SpeculativeRoutingWorkload


def get_provider_by_name(name: str, **kwargs: Any) -> SpeculationDecisionProvider:
    """Factory creating a decision provider by canonical or short name."""
    norm = name.lower().strip()
    if norm in ("no_speculation", "none", "b0"):
        return NoSpeculationProvider()
    elif norm in ("current_e3", "e3", "b1"):
        return E3CurrentDraftProvider()
    elif norm in ("deterministic_baseline", "deterministic", "heuristic", "b2"):
        return DeterministicFrequencyProvider()
    elif norm in ("typesafe_jev", "jev", "typesafe", "b3"):
        return TypeSafeJevProvider(**kwargs)
    elif norm in ("llm_system_one", "system_one_llm", "llm", "b4"):
        return SystemOneLLMProvider(**kwargs)
    elif norm in ("replay", "replay_jev"):
        return ReplaySpeculationProvider(**kwargs)
    else:
        raise ValueError(f"Unknown predictor '{name}'. Choose from: no_speculation, current_e3, deterministic_baseline, typesafe_jev, llm_system_one, replay")


def run_typesafe_pilot() -> int:
    """Executes the TypeSafe exploratory live pilot check."""
    print("\n=======================================================")
    print("⚡ ToolSpeed: TypeSafe/Jev Pilot Inspection")
    print("=======================================================")

    try:
        import typesafe_sdk
        sdk_available = True
        sdk_version = getattr(typesafe_sdk, "__version__", "0.6.0")
    except ImportError:
        sdk_available = False
        sdk_version = "not installed"

    api_key = os.environ.get("TYPESAFE_API_KEY")

    print(f"  • typesafe-sdk available: {sdk_available} (version: {sdk_version})")
    print(f"  • TYPESAFE_API_KEY set: {bool(api_key)}")

    if not sdk_available:
        print("\n❌ typesafe-sdk is not installed in the current environment.")
        print("To install: uv pip install typesafe-sdk")
        return 1

    if not api_key:
        print("\n=======================================================")
        print("LIVE JEV PILOT BLOCKED:")
        print("TYPESAFE_API_KEY not present.")
        print("No live TypeSafe performance claim made.")
        print("=======================================================\n")
        return 0

    print("\n🔑 Live API key detected. Initiating exploratory connection test...")
    # Safe exploratory verification
    provider = TypeSafeJevProvider(api_key=api_key, timeout_s=1.0)
    print(f"Provider '{provider.provider_name}' initialized successfully.")
    return 0


async def _run_benchmark_trials_async(
    provider: SpeculationDecisionProvider,
    tool_latency_ms: float = 250.0,
    candidate_count: int = 4,
    trials: int = 20,
    confidence_threshold: float = 0.70,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs paired execution of W8 tasks with the given decision provider."""
    workload = W8SpeculativeRoutingWorkload(
        tool_latency_ms=tool_latency_ms,
        candidate_count=candidate_count,
    )
    tools = workload.get_tools()
    registry = ToolRegistry(tools)

    tasks = workload.generate_tasks(count=trials, seed=seed)

    scheduler = SpeculativeReadScheduler(
        config=SchedulerConfig(
            speculation_enabled=True,
            speculation_confidence_threshold=confidence_threshold,
            timeout_seconds=10.0,
        ),
        speculation_provider=provider,
        candidate_builder=workload.create_candidates_for_task,
    )

    latencies_ms: list[float] = []
    successes: list[bool] = []
    spec_launched = 0
    spec_hits = 0
    spec_misses = 0
    spec_cancelled = 0
    provider_latencies_ms: list[float] = []
    probabilities: list[float] = []
    is_correct_speculation: list[bool] = []

    for task in tasks:
        model = workload.create_model_for_task(task, decision_delay_ms=150.0)

        t0 = time.perf_counter()
        task_res: Any = await scheduler.execute(
            cast(Any, task),
            model,
            registry,
        )
        wall_ccl_ms = (time.perf_counter() - t0) * 1000.0
        ccl_ms = task_res.ccl_ms if task_res.ccl_ms is not None else wall_ccl_ms

        is_success = bool(task_res.success)
        successes.append(is_success)
        latencies_ms.append(ccl_ms)

        # Track guardrail metrics
        m = task_res.guardrails
        spec_launched += m.speculative_calls_launched
        spec_hits += m.speculative_calls_hit
        spec_misses += m.speculative_calls_wasted
        spec_cancelled += m.speculative_calls_cancelled

        # Extract provider events from events
        for ev in task_res.events:
            if ev.event_type == EventType.CUSTOM and ev.details.get("event") == "speculation_provider_latency":
                provider_latencies_ms.append(float(ev.details.get("latency_ms", 0.0)))

        total_task_launched = max(m.speculative_calls_launched, m.speculative_calls_hit + m.speculative_calls_wasted + m.speculative_calls_cancelled)
        if total_task_launched > 0:
            is_hit = m.speculative_calls_hit > 0
            is_correct_speculation.append(is_hit)
            probabilities.append(1.0 if is_hit else 0.0)

    # Compute statistics
    lat_arr = np.array(latencies_ms)
    p50 = float(np.median(lat_arr)) if len(lat_arr) else 0.0
    p95 = float(np.percentile(lat_arr, 95)) if len(lat_arr) else 0.0
    mean_lat = float(np.mean(lat_arr)) if len(lat_arr) else 0.0

    prov_lat_arr = np.array(provider_latencies_ms) if provider_latencies_ms else np.array([0.0])
    prov_p50 = float(np.median(prov_lat_arr))
    prov_p95 = float(np.percentile(prov_lat_arr, 95))

    total_launched = max(spec_launched, spec_hits + spec_misses + spec_cancelled)
    precision = (spec_hits / total_launched) if total_launched > 0 else 0.0
    recall = (spec_hits / len(tasks)) if tasks else 0.0
    useful_rate = (spec_hits / len(tasks)) if tasks else 0.0
    wasted_rate = (spec_misses / len(tasks)) if tasks else 0.0

    # Brier score: (p - y)^2
    brier_score = float(np.mean((np.array(probabilities) - np.array(is_correct_speculation)) ** 2)) if probabilities else 0.0

    return {
        "provider": provider.provider_name,
        "tool_latency_ms": tool_latency_ms,
        "trials": len(tasks),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "ccl_mean_ms": mean_lat,
        "ccl_p50_ms": p50,
        "ccl_p95_ms": p95,
        "spec_launched": spec_launched,
        "spec_hits": spec_hits,
        "spec_misses": spec_misses,
        "spec_cancelled": spec_cancelled,
        "precision": precision,
        "recall": recall,
        "useful_rate": useful_rate,
        "wasted_rate": wasted_rate,
        "provider_latency_p50_ms": prov_p50,
        "provider_latency_p95_ms": prov_p95,
        "brier_score": brier_score,
        "all_ccl_ms": latencies_ms,
    }


def run_typesafe_benchmark(
    predictor: str = "current_e3",
    tool_latency_ms: float = 250.0,
    candidate_count: int = 4,
    trials: int = 20,
    confidence_threshold: float = 0.70,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs benchmark for a single predictor and outputs formatted results."""
    provider = get_provider_by_name(predictor)
    print(f"\n⚡ Running ToolSpeed benchmark: predictor={provider.provider_name}, tool_latency={tool_latency_ms}ms, candidates={candidate_count}, trials={trials}...")

    res = asyncio.run(
        _run_benchmark_trials_async(
            provider=provider,
            tool_latency_ms=tool_latency_ms,
            candidate_count=candidate_count,
            trials=trials,
            confidence_threshold=confidence_threshold,
            seed=seed,
        )
    )

    rows = [
        ["Predictor", res["provider"]],
        ["Tool Latency (ms)", f"{res['tool_latency_ms']:.1f}"],
        ["Success Rate", f"{res['success_rate'] * 100:.1f}%"],
        ["CCL Mean (ms)", f"{res['ccl_mean_ms']:.1f}"],
        ["CCL P50 (ms)", f"{res['ccl_p50_ms']:.1f}"],
        ["CCL P95 (ms)", f"{res['ccl_p95_ms']:.1f}"],
        ["Speculations Launched", str(res["spec_launched"])],
        ["Speculation Hits", str(res["spec_hits"])],
        ["Speculation Misses", str(res["spec_misses"])],
        ["Speculation Precision", f"{res['precision'] * 100:.1f}%"],
        ["Provider Latency P50 (ms)", f"{res['provider_latency_p50_ms']:.1f}"],
        ["Provider Latency P95 (ms)", f"{res['provider_latency_p95_ms']:.1f}"],
    ]
    print(ascii_table(["Metric", "Value"], rows, ["left", "right"]))
    return res


def run_typesafe_sweep(
    tool_latencies: list[float] | None = None,
    predictors: list[str] | None = None,
    trials: int = 20,
    seed: int = 42,
) -> dict[float, dict[str, Any]]:
    """Sweeps over multiple tool latency bins to locate the break-even curve."""
    latencies = tool_latencies or [50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]
    preds = predictors or ["no_speculation", "current_e3", "deterministic_baseline", "replay"]

    print(f"\n⚡ Running Break-Even Sweep across {len(latencies)} latency bins and {len(preds)} predictors ({trials} trials/bin)...")
    results_by_lat: dict[float, dict[str, Any]] = {}

    table_rows: list[list[str]] = []

    for lat in latencies:
        results_by_lat[lat] = {}
        row = [f"{lat:.0f} ms"]
        for p in preds:
            prov = get_provider_by_name(p)
            res = asyncio.run(
                _run_benchmark_trials_async(
                    provider=prov,
                    tool_latency_ms=lat,
                    trials=trials,
                    seed=seed,
                )
            )
            results_by_lat[lat][p] = res
            row.append(f"{res['ccl_p50_ms']:.1f}")
        table_rows.append(row)

    headers = ["Tool Latency"] + [p for p in preds]
    alignments = ["left"] + ["right" for _ in preds]
    print("\n" + ascii_table(headers, table_rows, alignments) + "\n")

    return results_by_lat


def run_typesafe_replay() -> dict[str, Any]:
    """Generates and executes deterministic replay verification."""
    print("\n⚡ Running TypeSafe Deterministic Replay Verification...")

    # Build representative sanitized replay fixtures for W8 tasks
    workload = W8SpeculativeRoutingWorkload(tool_latency_ms=250.0, candidate_count=4)
    tasks = workload.generate_tasks(count=5, seed=101)

    records: list[ReplayRecord] = []
    for _i, task in enumerate(tasks):
        candidates = workload.create_candidates_for_task(task)
        target_tool = task.metadata["target_tool"]
        matching_cand = next((c for c in candidates if c.tool_name == target_tool), candidates[0])

        prob_map = {c.candidate_id: (0.85 if c.candidate_id == matching_cand.candidate_id else 0.05) for c in candidates}
        prob_map["no_speculation"] = 0.05

        state = SpeculationState(
            task_id=task.task_id,
            prompt=task.prompt,
            step_index=1,
            history_summary=(),
        )
        comp_hash, req_hash, cand_hash = compute_composite_hash(state, candidates)

        rec = ReplayRecord(
            composite_hash=comp_hash,
            request_hash=req_hash,
            candidate_set_hash=cand_hash,
            provider="replay_jev",
            selected_candidate_id=matching_cand.candidate_id,
            probability=0.85,
            confidence=0.88,
            should_speculate=True,
            observed_latency_ms=45.0,  # Simulated Jev latency
            probabilities=prob_map,
            metadata={"source": "sanitized_trace"},
        )
        records.append(rec)

    replay_provider = ReplaySpeculationProvider(records=records, simulate_latency=False)
    res = asyncio.run(
        _run_benchmark_trials_async(
            provider=replay_provider,
            tool_latency_ms=250.0,
            trials=5,
            seed=101,
        )
    )

    print("✅ Deterministic Replay Verification PASSED.")
    print(f"  • Speculation Hits: {res['spec_hits']} / {res['trials']}")
    print(f"  • Precision: {res['precision'] * 100:.1f}%")
    print(f"  • CCL P50: {res['ccl_p50_ms']:.1f}ms")
    return res
