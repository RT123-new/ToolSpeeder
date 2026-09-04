"""Independent hypothesis falsification evaluator with fail-closed trace recomputation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolspeed.benchmarks.bundle import validate_bundle_hashes_first
from toolspeed.benchmarks.recompute import recompute_bundle_metrics


def evaluate_falsification_independent(
    bundle_path: str | Path,
    strict_traces: bool = True,
    protocol_thresholds: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, Any]]:
    """Evaluates hypothesis falsification status independently from raw traces.

    Exit codes:
      0 = PASSED (all hypotheses met under verdict-eligible confirmatory evidence)
      1 = FALSIFIED (one or more hypotheses failed efficacy, safety, or speedup thresholds)
      2 = INCONCLUSIVE (synthetic simulation, underpowered sample size, or exploratory mode)
      3 = MALFORMED / ERROR (missing bundle, corrupted trace files, hash mismatch)
    """
    b_dir = Path(bundle_path).resolve()
    if not b_dir.exists():
        return 3, f"Bundle path '{b_dir}' does not exist", {}

    if b_dir.is_file():
        b_dir = b_dir.parent

    # 1. Verify bundle structural and hash integrity if manifest exists
    manifest_file = b_dir / "manifest.json"
    if manifest_file.exists():
        valid_hash, hash_errors = validate_bundle_hashes_first(b_dir)
        if not valid_hash:
            return 3, f"Bundle hash verification failed: {'; '.join(hash_errors)}", {}

    # 2. Check presence of raw traces
    traces_file = b_dir / "raw-traces.jsonl"
    if not traces_file.exists():
        traces_file = b_dir / "candidate-traces.jsonl"

    if strict_traces and (not traces_file.exists() or traces_file.stat().st_size == 0):
        return 3, "Missing or empty raw-traces.jsonl in bundle", {}

    # If traces are absent and strict_traces is False, cannot evaluate from traces
    if not traces_file.exists():
        return 3, "No raw trace files found in bundle", {}

    # Validate trace content is valid JSONL
    try:
        non_empty_lines = 0
        with open(traces_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    non_empty_lines += 1
        if non_empty_lines == 0:
            return 3, "raw-traces.jsonl contains no valid trace records", {}
    except Exception as e:
        return 3, f"Corrupted or invalid JSON in raw-traces.jsonl: {e}", {}

    # 3. Independent recomputation from raw traces
    try:
        recomputed = recompute_bundle_metrics(b_dir)
    except Exception as e:
        return 3, f"Failed to recompute metrics from raw traces: {e}", {}

    evaluations = recomputed.get("evaluations", [])
    if not evaluations:
        return 3, "No evaluations could be derived from raw traces", {}

    # 4. Check eligibility from manifest or protocol
    manifest_data: dict[str, Any] = {}
    if manifest_file.exists():
        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    ev_level = manifest_data.get("evidence_level") or recomputed.get("evidence_level", "synthetic")
    if ev_level == "synthetic":
        return 2, "Evidence level is 'synthetic' — real-world hypothesis claims are inconclusive", recomputed

    trial_count = manifest_data.get("trial_count", non_empty_lines)
    min_required = 1000 if ev_level == "replay_integration" else 200
    is_eligible = manifest_data.get("is_verdict_eligible", True)

    if not is_eligible or trial_count < min_required:
        return (
            2,
            f"Underpowered sample size or smoke trial (n={trial_count} < {min_required})",
            recomputed,
        )

    # 5. Evaluate recomputed metrics against hypothesis thresholds
    # Never reads summary metrics from result.json!
    for ev in evaluations:
        wl = ev.get("workload_id", "unknown")
        summ = ev.get("summary", {})

        p95_speedup = summ.get("p95_speedup")
        cand_succ = summ.get("candidate_success_rate")
        side_effects = summ.get("unapproved_side_effects", 0)

        # Safety: 0 unapproved side effects
        if side_effects > 0:
            return 1, f"Workload '{wl}' had {side_effects} unapproved side effect(s)", recomputed

        # Success rate threshold: min 95%
        if cand_succ is not None and cand_succ < 0.95:
            return (
                1,
                f"Workload '{wl}' candidate success rate {cand_succ:.1%} < 95.0% threshold",
                recomputed,
            )

        # Speedup threshold: min 1.0x (or mechanism threshold)
        min_sp = 1.0
        if protocol_thresholds and wl in protocol_thresholds:
            min_sp = float(protocol_thresholds[wl].get("min_p95_speedup", 1.0))

        if p95_speedup is not None and p95_speedup < min_sp:
            return (
                1,
                f"Workload '{wl}' p95 speedup {p95_speedup:.2f}x < required {min_sp:.2f}x threshold",
                recomputed,
            )

    # 6. Check confirmatory mode: exploratory runs cannot yield final confirmatory PASS
    mode = manifest_data.get("mode", "confirmatory")
    if mode == "exploratory":
        return 2, "Exploratory mode runs cannot produce final confirmatory PASS verdict", recomputed

    return 0, "All hypotheses passed under confirmatory empirical evidence", recomputed
