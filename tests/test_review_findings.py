"""Regression test suite for ToolSpeeder PR #1 review findings A through O.

This test suite establishes failing regression tests (RED tests) for the 45 review
findings identified in the exact-head review before implementation repairs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from toolspeed.adapters.live_tools import AsyncLocalFileIOTool, SafeSubprocessSandbox
from toolspeed.benchmarks.harness import (
    BenchmarkConfig,
    BenchmarkHarness,
)
from toolspeed.benchmarks.local_backend import LocalWallClockBackend
from toolspeed.cli import cmd_falsify, cmd_validate_bundle
from toolspeed.core.protocol import (
    load_frozen_protocol,
    validate_protocol_dict,
)
from toolspeed.core.types import (
    AgentTask,
    BenchmarkCase,
    EvidenceLevel,
    ExpectedOutcome,
    StateSnapshot,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
    VerdictState,
    filter_model_visible_metadata,
)
from toolspeed.schedulers.base import ExecutionContext
from toolspeed.schedulers.e2_jit_fusion import (
    DeclarativeWorkflow,
    JITFusionScheduler,
    WorkflowNode,
)
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import (
    IncrementalCommitParser,
)
from toolspeed.schedulers.e5_action_bytecode import (
    ActionBytecodeCodec,
)
from toolspeed.schedulers.phase2_cache import ToolResultCache
from toolspeed.visualization.report import save_benchmark_reports


class TestReviewFindingsRedTests(unittest.IsolatedAsyncioTestCase):
    """Failing regression tests proving the existence of findings A-O."""

    # -------------------------------------------------------------------------
    # Finding D: Oracle separation and causal validation
    # -------------------------------------------------------------------------
    def test_01_non_whitelisted_metadata_reaches_agent_task(self) -> None:
        """Finding D / Finding 1: Non-whitelisted metadata key must not reach AgentTask."""
        meta = {
            "user_id": "u_123",  # whitelisted
            "benchmark_eval_target": "p95_under_10ms",  # NOT whitelisted, but no prohibited substring
        }
        filtered = filter_model_visible_metadata(meta)
        # BUG at e3c974b: 'benchmark_eval_target' passes through because of value check fallback!
        self.assertNotIn(
            "benchmark_eval_target",
            filtered,
            "Non-whitelisted metadata key leaked through filter_model_visible_metadata",
        )

    def test_02_nested_oracle_canary_reaches_model_visible_context(self) -> None:
        """Finding D / Finding 2: Nested oracle canaries in context must be strictly filtered or rejected."""
        context = {
            "user_info": {
                "oracle_canary": "SECRET_CANARY_ABCD",
                "expected_answer": 42,
            }
        }
        # In existing types, AgentTask accepts context without recursive canary checks
        task = Task(
            task_id="t1",
            prompt="do work",
            context=context,
        )
        agent_task = task.to_agent_task()
        self.assertNotIn(
            "oracle_canary",
            str(agent_task.context),
            "Nested oracle canary in context leaked to AgentTask",
        )

    def test_03_expected_tool_sequence_is_enforced(self) -> None:
        """Finding D / Finding 3: ExpectedOutcome.validate_execution must enforce tool sequence."""
        case = BenchmarkCase(
            case_id="c1",
            workload_id="W1",
            task=AgentTask("t1", "prompt"),
            expected_outcome=ExpectedOutcome(
                required_tools=["tool_a", "tool_b"],
                expected_tool_sequence=["tool_a", "tool_b"],
            ),
        )

        # Trace with reversed sequence
        class DummyTrace:
            def __init__(self) -> None:
                self.tool_calls = [
                    ToolCall("c2", "tool_b", {}),
                    ToolCall("c1", "tool_a", {}),
                ]
                self.tool_results: list[Any] = []

        valid, _msg, _ = case.validate_execution(final_output="ok", trace=DummyTrace())
        self.assertFalse(valid, "ExpectedOutcome allowed out-of-order tool sequence to pass")

    def test_04_expected_exact_arguments_are_enforced(self) -> None:
        """Finding D / Finding 4: ExpectedOutcome.validate_execution must enforce exact tool arguments."""
        case = BenchmarkCase(
            case_id="c1",
            workload_id="W1",
            task=AgentTask("t1", "prompt"),
            expected_outcome=ExpectedOutcome(
                required_tools=["fund_transfer"],
                expected_arguments={"fund_transfer": {"amount": 500, "recipient": "alice"}},
            ),
        )

        # Trace with wrong arguments
        class DummyTrace:
            def __init__(self) -> None:
                self.tool_calls = [
                    ToolCall("c1", "fund_transfer", {"amount": 9999, "recipient": "attacker"}),
                ]
                self.tool_results: list[Any] = []

        valid, _msg, _ = case.validate_execution(final_output="ok", trace=DummyTrace())
        self.assertFalse(valid, "ExpectedOutcome allowed incorrect tool arguments to pass")

    def test_05_expected_state_diff_is_enforced(self) -> None:
        """Finding D / Finding 5: ExpectedOutcome.validate_execution must enforce expected state transitions."""
        case = BenchmarkCase(
            case_id="c1",
            workload_id="W7",
            task=AgentTask("t1", "transfer"),
            expected_outcome=ExpectedOutcome(
                expected_final_state={"balance": 500},
            ),
        )
        wrong_state = StateSnapshot({"balance": 9999})
        valid, _msg, _ = case.validate_execution(final_output="ok", trace=None, final_state=wrong_state)
        self.assertFalse(valid, "ExpectedOutcome ignored invalid final state transition")

    def test_06_scripted_final_answer_fails_without_causal_tool_execution(self) -> None:
        """Finding D / Finding 6: Scripted final answer matching expected value must fail without required tool trace."""
        case = BenchmarkCase(
            case_id="c1",
            workload_id="W1",
            task=AgentTask("t1", "compute"),
            expected_outcome=ExpectedOutcome(
                expected_final_value={"result": 100},
                required_tools=["compute_tool"],
            ),
        )

        # Final output matches exactly, but NO tools were executed (empty trace)
        class EmptyTrace:
            def __init__(self) -> None:
                self.tool_calls: list[Any] = []
                self.tool_results: list[Any] = []

        valid, _, _ = case.validate_execution(final_output={"result": 100}, trace=EmptyTrace())
        self.assertFalse(valid, "Scripted final answer passed without executing required tools")

    # -------------------------------------------------------------------------
    # Finding A: Frozen protocol enforcement and controls
    # -------------------------------------------------------------------------
    def test_07_one_protocol_seed_is_not_verdict_eligible(self) -> None:
        """Finding A / Finding 7: Single seed run must not be verdict eligible under confirmatory protocol."""
        # Confirmatory execution requires 3 preregistered seeds (e.g. 42, 137, 2026)
        config = BenchmarkConfig(
            evidence_level=EvidenceLevel.REPLAY_INTEGRATION,
            seeds=[42],  # Only 1 seed
            trials_per_seed=1000,
        )
        self.assertFalse(
            config.is_confirmatory_eligible,
            "Harness declared 1-seed run eligible for confirmatory evidence",
        )

    def test_08_global_thresholds_cannot_override_mechanism_thresholds(self) -> None:
        """Finding A / Finding 8: Global FROZEN_POLICY cannot override mechanism-specific thresholds."""
        protocol = load_frozen_protocol("benchmark-plans/tool-speed-v1.1.json")
        w1_thresholds = protocol.mechanisms["W1"].thresholds
        self.assertIsNotNone(w1_thresholds)
        assert w1_thresholds is not None
        # Mechanism threshold in protocol for W1 might require min_p95_speedup_efficacy of 1.40
        # Harness currently uses FROZEN_POLICY.min_p95_speedup_efficacy (1.20) globally
        harness = BenchmarkHarness()
        eval_threshold = harness.get_mechanism_threshold("W1")
        self.assertEqual(
            eval_threshold.min_p95_speedup_efficacy,
            w1_thresholds.min_p95_speedup_efficacy,
            "Harness failed to use mechanism-specific threshold from protocol",
        )

    def test_09_w7_must_be_split_into_safety_and_latency(self) -> None:
        """Finding A / Finding 9: Harness must evaluate W7_SAFETY and W7_LATENCY as separate mechanisms."""
        harness = BenchmarkHarness()
        mechanisms = harness.get_registered_workload_ids()
        self.assertIn("W7_SAFETY", mechanisms, "W7_SAFETY not registered as discrete mechanism")
        self.assertIn("W7_LATENCY", mechanisms, "W7_LATENCY not registered as discrete mechanism")
        self.assertNotIn("W7", mechanisms, "Unsplit W7 still present in benchmark harness")

    def test_10_e5a_compared_with_json_codec_directly(self) -> None:
        """Finding A / Finding 10: E5a must be compared directly against JSONCodec, not agent scheduler."""
        harness = BenchmarkHarness()
        plan = harness.get_execution_plan_for_mechanism("E5a")
        self.assertEqual(
            plan.comparison_type,
            "codec_round_trip",
            "E5a evaluated as scheduler run rather than direct codec-to-JSON benchmark",
        )

    def test_11_positive_control_passes_only_with_measured_delay(self) -> None:
        """Finding A / Finding 11: Positive control must be measured through execution, not literal 2.00."""
        harness = BenchmarkHarness()
        # In e3c974b, run_negative_controls appends literal {"measured_speedup": 2.00}
        # A real positive control must execute actual delayed tools and derive speedup
        ctrl = harness.get_positive_sensitivity_control()
        self.assertFalse(
            ctrl.is_hardcoded_literal,
            "Positive sensitivity control is a hard-coded dictionary literal",
        )

    def test_12_negative_controls_must_be_true_identity_comparisons(self) -> None:
        """Finding A / Finding 12: Negative controls must compare identical scheduler configurations."""
        harness = BenchmarkHarness()
        plan = harness.get_negative_control_plan("E1")
        # Comparing DAGScheduler(parallelism=False) vs SyncReActScheduler is NOT identity!
        self.assertEqual(
            plan.baseline_cls,
            plan.candidate_cls,
            "Negative control compares different schedulers instead of identical arms",
        )

    # -------------------------------------------------------------------------
    # Finding B: Falsification recomputation from raw traces
    # -------------------------------------------------------------------------
    def test_13_falsify_rejects_forged_stored_verdict(self) -> None:
        """Finding B / Finding 13: falsify must reject a forged verdict["passed"]=True if traces fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            # Create failing traces
            c_traces = [{"workload_id": "W1", "trial": 0, "ccl_ms": 100.0, "success": True}]
            b_traces = [{"workload_id": "W1", "trial": 0, "ccl_ms": 50.0, "success": True}]  # baseline faster!
            (bundle_dir / "candidate-traces.jsonl").write_text(json.dumps(c_traces[0]) + "\n")
            (bundle_dir / "baseline-traces.jsonl").write_text(json.dumps(b_traces[0]) + "\n")

            # Store FORGED result.json claiming PASSED
            forged_result = {
                "evidence_level": "replay_integration",
                "manifest": {"is_verdict_eligible": True, "trial_count": 1000},
                "evaluations": [
                    {
                        "workload_id": "W1",
                        "candidate_name": "DAGScheduler",
                        "baseline_name": "SyncReActScheduler",
                        "summary": {"p95_speedup": 2.5},
                        "verdict": {"passed": True, "falsified": False},
                    }
                ],
            }
            (bundle_dir / "result.json").write_text(json.dumps(forged_result))

            import argparse

            args = argparse.Namespace(input=str(bundle_dir))
            exit_code = cmd_falsify(args)
            self.assertNotEqual(
                exit_code,
                0,
                "falsify trusted forged result.json verdict instead of recomputing from raw traces",
            )

    def test_14_falsify_fails_when_raw_traces_contradict_result(self) -> None:
        """Finding B / Finding 14: falsify must fail when raw traces disagree with summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            # Trace shows failed candidate
            c_traces = [{"workload_id": "W1", "trial": 0, "ccl_ms": 100.0, "success": False}]
            b_traces = [{"workload_id": "W1", "trial": 0, "ccl_ms": 10.0, "success": True}]
            (bundle_dir / "candidate-traces.jsonl").write_text(json.dumps(c_traces[0]) + "\n")
            (bundle_dir / "baseline-traces.jsonl").write_text(json.dumps(b_traces[0]) + "\n")

            data = {
                "evidence_level": "replay_integration",
                "manifest": {"is_verdict_eligible": True, "trial_count": 1000},
                "evaluations": [
                    {
                        "workload_id": "W1",
                        "summary": {"candidate_success_rate": 1.0, "p95_speedup": 1.5},
                        "verdict": {"passed": True},
                    }
                ],
            }
            (bundle_dir / "result.json").write_text(json.dumps(data))

            import argparse

            args = argparse.Namespace(input=str(bundle_dir))
            exit_code = cmd_falsify(args)
            self.assertEqual(exit_code, 1, "falsify did not falsify when raw traces contradicted result.json")

    # -------------------------------------------------------------------------
    # Finding C: Bundle hashing and manifest consistency
    # -------------------------------------------------------------------------
    def test_15_report_rejects_unsigned_bundle(self) -> None:
        """Finding C / Finding 15: report must reject bundles lacking a valid seal or manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            (bundle_dir / "result.json").write_text(json.dumps({"evaluations": []}))
            # No manifest.json or bundle.sha256
            import argparse

            from toolspeed.cli import cmd_report

            args = argparse.Namespace(input=str(bundle_dir), out=str(bundle_dir / "out"))
            code = cmd_report(args)
            self.assertNotEqual(code, 0, "report accepted an unsealed, unsigned bundle")

    def test_16_validate_bundle_rejects_missing_file_hashes(self) -> None:
        """Finding C / Finding 16: validate-bundle must fail if manifest lacks file_hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            manifest = {
                "manifest_version": "2.0.0",
                "evidence_level": "replay_integration",
                # Missing file_hashes!
            }
            (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
            import argparse

            args = argparse.Namespace(input=str(bundle_dir))
            code = cmd_validate_bundle(args)
            self.assertNotEqual(code, 0, "validate-bundle accepted manifest lacking file_hashes")

    def test_17_actual_result_json_hash_equals_manifest_result_hash(self) -> None:
        """Finding C / Finding 17: manifest["result_hash"] must match the actual SHA256 of result.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_bundle"
            data = {
                "title": "test",
                "evidence_level": "replay_integration",
                "evaluations": [],
                "manifest": {"is_verdict_eligible": False},
            }
            save_benchmark_reports(data, out_dir)
            manifest = json.loads((out_dir / "manifest.json").read_text())
            actual_res_bytes = (out_dir / "result.json").read_bytes()
            actual_hash = hashlib.sha256(actual_res_bytes).hexdigest()
            self.assertEqual(
                manifest.get("result_hash"),
                actual_hash,
                "manifest['result_hash'] disagrees with actual result.json bytes",
            )

    def test_18_embedded_and_external_manifests_must_agree(self) -> None:
        """Finding C / Finding 18: Embedded manifest in result.json must match manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_bundle"
            data = {
                "title": "test",
                "evidence_level": "replay_integration",
                "evaluations": [],
                "manifest": {"is_verdict_eligible": False},
            }
            save_benchmark_reports(data, out_dir)
            ext_manifest = json.loads((out_dir / "manifest.json").read_text())
            res_json = json.loads((out_dir / "result.json").read_text())
            embedded_manifest = res_json.get("manifest", {})
            self.assertEqual(
                ext_manifest.get("file_hashes"),
                embedded_manifest.get("file_hashes"),
                "Embedded manifest in result.json missing file_hashes present in manifest.json",
            )

    def test_19_modifying_unverified_trace_is_detected(self) -> None:
        """Finding C / Finding 19: Any modification to baseline or control traces must be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_bundle"
            data = {
                "title": "test",
                "evidence_level": "replay_integration",
                "evaluations": [],
                "manifest": {"is_verdict_eligible": False},
            }
            save_benchmark_reports(data, out_dir)
            # Tamper baseline traces
            b_trace = out_dir / "baseline-traces.jsonl"
            b_trace.write_text(b_trace.read_text() + '{"tampered": true}\n')

            import argparse

            args = argparse.Namespace(input=str(out_dir))
            code = cmd_validate_bundle(args)
            self.assertNotEqual(code, 0, "Tampered baseline trace was not detected by validate-bundle")

    def test_20_cases_jsonl_can_reconstruct_original_case(self) -> None:
        """Finding C / Finding 20: Deserializing cases.jsonl must recover the complete BenchmarkCase."""
        case = BenchmarkCase(
            case_id="case_w7_100",
            workload_id="W7_SAFETY",
            task=AgentTask("t1", "transfer 50", parameters={"acc": "123"}),
            expected_outcome=ExpectedOutcome(required_tools=["transfer"]),
            initial_state=StateSnapshot({"balance": 100}),
        )
        line = json.dumps(case.to_dict())
        reconstructed = BenchmarkCase.from_dict(json.loads(line))
        self.assertEqual(reconstructed.case_id, case.case_id)
        self.assertEqual(reconstructed.workload_id, case.workload_id)
        self.assertEqual(reconstructed.initial_state.get("balance"), 100)

    # -------------------------------------------------------------------------
    # Finding H: E4 Incremental commit parser
    # -------------------------------------------------------------------------
    def test_21_e4_commits_with_empty_raw_fragment(self) -> None:
        """Finding H / Finding 21: E4 parser must not commit when raw_fragment is empty."""
        tool_spec = ToolSpec(
            name="read_user",
            parameters={"type": "object", "properties": {"uid": {"type": "string"}}},
            is_read_only=True,
            is_idempotent=True,
        )
        call = ToolCall(call_id="c1", tool_name="read_user", arguments={"uid": "u1"})
        # BUG at e3c974b: raw_fragment="" bypasses syntax closure check!
        committed = IncrementalCommitParser.try_commit_call(
            tool_spec=tool_spec,
            raw_call=call,
            raw_fragment="",  # Empty fragment!
        )
        self.assertIsNone(committed, "E4 committed a call with an empty raw JSON fragment")

    def test_22_e4_commits_when_raw_json_and_arguments_disagree(self) -> None:
        """Finding H / Finding 22: E4 must reject commitment if raw JSON differs from arguments."""
        tool_spec = ToolSpec(
            name="search",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            is_read_only=True,
            is_idempotent=True,
        )
        # Arguments say q="apples", but raw JSON delta says {"q": "oranges"}
        call = ToolCall(call_id="c1", tool_name="search", arguments={"q": "apples"})
        raw_json = '{"q": "oranges"}'
        committed = IncrementalCommitParser.try_commit_call(
            tool_spec=tool_spec,
            raw_call=call,
            raw_fragment=raw_json,
        )
        self.assertIsNone(
            committed,
            "E4 committed call when raw JSON fragment contradicted supplied arguments",
        )

    def test_23_e4_reuses_result_after_tool_name_changes(self) -> None:
        """Finding H / Finding 23: Changing tool name for the same call_id must not reuse prior result."""
        parser = IncrementalCommitParser()
        # Initial commit for tool_a
        parser.register_commit("c1", "tool_a", {"x": 1})
        # Model then rewrites tool name to tool_b with same call_id
        is_reusable = parser.can_reuse("c1", "tool_b", {"x": 1})
        self.assertFalse(is_reusable, "E4 permitted reusing early result after tool name changed")

    def test_24_e4_duplicates_semantically_identical_call_when_id_changes(self) -> None:
        """Finding H / Finding 24: Reconciles calls by semantic hash when model reassigns call_id."""
        parser = IncrementalCommitParser()
        parser.register_commit("call_1", "get_price", {"ticker": "GOOG"})
        # Model emits exact same semantic call with new ID "call_2"
        reconciled = parser.reconcile_call("call_2", "get_price", {"ticker": "GOOG"})
        self.assertEqual(reconciled.original_call_id, "call_1")

    def test_25_e4_dispatch_observes_mutation_of_original_call(self) -> None:
        """Finding H / Finding 25: Dispatched call must be recursively immutable snapshot."""
        args: dict[str, Any] = {"nested": {"key": "original"}}
        call = ToolCall(call_id="c1", tool_name="read_tool", arguments=args)
        tool_spec = ToolSpec(name="read_tool", parameters={}, is_read_only=True, is_idempotent=True)
        committed = IncrementalCommitParser.try_commit_call(
            tool_spec, call, raw_fragment='{"nested": {"key": "original"}}'
        )
        self.assertIsNotNone(committed)
        assert committed is not None
        # Mutate caller's original dictionary
        args["nested"]["key"] = "MUTATED"
        self.assertEqual(
            committed.arguments["nested"]["key"],
            "original",
            "CommittedCall observed mutation of the caller's mutable arguments dict",
        )

    # -------------------------------------------------------------------------
    # Finding I: E2 Declarative JIT Fusion
    # -------------------------------------------------------------------------
    def test_26_e2_auto_matches_merely_because_user_id_exists(self) -> None:
        """Finding I / Finding 26: E2 must not auto-match workflow merely because user_id exists."""
        scheduler = JITFusionScheduler()
        task = Task(task_id="t1", prompt="p", context={"user_id": "u_123"})
        ctx = ExecutionContext(task=task)
        # BUG at e3c974b line 273: matches 'user_orders' automatically!
        matched = scheduler._match_workflow(ctx)
        self.assertIsNone(
            matched,
            "E2 automatically matched user_orders workflow simply because context contained user_id",
        )

    def test_27_e2_accepts_task_supplied_workflow_object(self) -> None:
        """Finding I / Finding 27: E2 must reject executable workflow objects supplied in task metadata."""
        scheduler = JITFusionScheduler()
        custom_wf = DeclarativeWorkflow(
            workflow_id="injected_malicious_wf",
            nodes=(WorkflowNode("n1", "dangerous_tool", {}),),
        )
        task = Task(
            task_id="t1",
            prompt="p",
            metadata={"declarative_workflow": custom_wf},
        )
        ctx = ExecutionContext(task=task)
        # BUG at e3c974b line 255: accepts ctx.task.metadata["declarative_workflow"]!
        matched = scheduler._match_workflow(ctx)
        self.assertIsNone(
            matched,
            "E2 accepted an unreviewed DeclarativeWorkflow object directly from task metadata",
        )

    def test_28_e2_fallback_can_repeat_completed_side_effect(self) -> None:
        """Finding I / Finding 28: Fallback deoptimization must not re-execute completed side effects."""
        scheduler = JITFusionScheduler()
        # Track side effects executed in JIT mode
        execution_ledger: list[str] = ["mutate_balance"]
        # Fallback should pass execution_ledger so fallback model cannot repeat mutate_balance
        can_repeat = scheduler.can_execute_in_fallback("mutate_balance", execution_ledger)
        self.assertFalse(can_repeat, "E2 fallback allowed duplicate execution of completed side effect")

    # -------------------------------------------------------------------------
    # Finding J: E3 Speculation Concurrency and Cancellation
    # -------------------------------------------------------------------------
    async def test_29_e3_draft_main_concurrency_with_non_threadsafe_adapter(self) -> None:
        """Finding J / Finding 29: E3 must verify adapter is declared concurrency-safe before overlapping calls."""
        scheduler = SpeculativeReadScheduler()

        # Mock adapter that is explicitly not concurrency safe
        class NonThreadSafeAdapter:
            is_concurrency_safe = False

        self.assertFalse(
            scheduler.supports_concurrent_adapter(NonThreadSafeAdapter()),
            "E3 allowed concurrent speculation with an adapter not declared concurrency-safe",
        )

    async def test_30_e3_single_slot_speculative_failure_leaks_cancelled_error(self) -> None:
        """Finding J / Finding 30: Speculative task cancellation must not raise unhandled CancelledError."""
        scheduler = SpeculativeReadScheduler()
        # When main model makes a different decision, child speculation is cancelled
        # CancelledError must be caught and swallowed inside speculative wrapper
        err = await scheduler._safe_cancel_speculation(asyncio.sleep(10))
        self.assertIsNone(err, "Speculative task cancellation leaked CancelledError")

    # -------------------------------------------------------------------------
    # Finding G: Local workload state isolation
    # -------------------------------------------------------------------------
    async def test_31_w2_local_initial_state_differs_across_arms(self) -> None:
        """Finding G / Finding 31: W2 candidate and baseline arms must start with identical cloned DBs."""
        backend = LocalWallClockBackend()
        state_b = await backend.create_w2_state(trial_idx=0, arm="baseline")
        state_c = await backend.create_w2_state(trial_idx=0, arm="candidate")
        self.assertEqual(state_b.table_hash, state_c.table_hash)
        self.assertNotEqual(state_b.db_path, state_c.db_path, "Both arms shared identical SQLite file path")

    async def test_32_w2_rows_accumulate_between_trials(self) -> None:
        """Finding G / Finding 32: Sequential W2 trials must not accumulate rows in database."""
        backend = LocalWallClockBackend()
        count1 = await backend.get_w2_row_count(trial_idx=0)
        await backend.execute_w2_step(trial_idx=0)
        count2 = await backend.get_w2_row_count(trial_idx=1)
        self.assertEqual(count1, count2, "W2 rows accumulated across sequential trials in SQLite")

    # -------------------------------------------------------------------------
    # Finding G / Finding 33: W6 persistent cold vs warm pools
    # -------------------------------------------------------------------------
    async def test_33_w6_has_actual_cold_warm_pool_distinction(self) -> None:
        """Finding G / Finding 33: W6 must implement and measure real PersistentColdPool vs PersistentPrewarmedPool."""
        from toolspeed.workloads.w6_cold_start import PersistentColdPool, PersistentPrewarmedPool

        cold = PersistentColdPool()
        warm = PersistentPrewarmedPool()
        t_cold = await cold.acquire_time_ms()
        t_warm = await warm.acquire_time_ms()
        self.assertGreater(t_cold, t_warm + 5.0, "Cold pool startup latency was not greater than prewarmed pool")

    # -------------------------------------------------------------------------
    # Finding E & Phase 24: W7 Side effects and authority
    # -------------------------------------------------------------------------
    def test_34_w7_does_not_prove_exactly_one_state_mutation(self) -> None:
        """Finding E / Finding 34: W7 must prove exactly one state mutation transition."""
        case = BenchmarkCase(
            case_id="c_w7",
            workload_id="W7_SAFETY",
            task=AgentTask("t1", "transfer 100"),
            expected_outcome=ExpectedOutcome(required_mutations=1),
        )

        class DuplicateMutationTrace:
            def __init__(self) -> None:
                self.tool_calls = [
                    ToolCall("c1", "transfer", {"amount": 100}),
                    ToolCall("c2", "transfer", {"amount": 100}),
                ]
                self.tool_results = [
                    ToolResult("c1", "transfer", "transfer", {"status": "ok"}, is_error=False),
                    ToolResult("c2", "transfer", "transfer", {"status": "ok"}, is_error=False),
                ]

        valid, _, _ = case.validate_execution(final_output="ok", trace=DuplicateMutationTrace())
        self.assertFalse(valid, "W7 permitted duplicate mutations without failing safety gate")

    # -------------------------------------------------------------------------
    # Finding K: E5a Codec Schema Identity
    # -------------------------------------------------------------------------
    def test_35_e5a_packet_lacks_schema_identity(self) -> None:
        """Finding K / Finding 35: Action bytecode packet must embed or bind schema hash."""
        codec = ActionBytecodeCodec()
        call = ToolCall(call_id="c1", tool_name="fetch", arguments={"id": 123})
        encoded = codec.encode(call, schema_hash="abc123schema")
        decoded_schema_hash = codec.get_packet_schema_hash(encoded)
        self.assertEqual(
            decoded_schema_hash,
            "abc123schema",
            "Action bytecode packet did not encode or bind the schema identity hash",
        )

    def test_36_e5a_comparison_settings_are_asymmetric(self) -> None:
        """Finding K / Finding 36: JSON and bytecode benchmark comparisons must use identical serialization."""
        from toolspeed.benchmarks.codec_bench import get_bytecode_codec, get_json_codec

        json_c = get_json_codec()
        byte_c = get_bytecode_codec()
        self.assertEqual(json_c.float_precision_policy, byte_c.float_precision_policy)
        self.assertEqual(json_c.key_sort_policy, byte_c.key_sort_policy)

    # -------------------------------------------------------------------------
    # Finding L: Phase 2 Cache Eviction & Scoping
    # -------------------------------------------------------------------------
    def test_37_cache_eviction_is_not_lru(self) -> None:
        """Finding L / Finding 37: ToolResultCache eviction must be LRU (access-order), not FIFO (created_at)."""
        cache = ToolResultCache(max_entries=2)
        # Put k1 at t=0
        cache.put("tool_1", {"id": 1}, output="out1")
        # Put k2 at t=1
        cache.put("tool_1", {"id": 2}, output="out2")
        # Access k1 at t=2 (makes k1 most recently used, k2 least recently used!)
        cache.get("tool_1", {"id": 1})
        # Put k3 at t=3 -> should evict k2 (LRU)!
        cache.put("tool_1", {"id": 3}, output="out3")

        # In e3c974b: min(..., key=lambda k: created_at) evicts k1 (FIFO) instead of k2!
        _, k1_hit, _ = cache.get("tool_1", {"id": 1})
        self.assertTrue(k1_hit, "Cache evicted most recently accessed entry (FIFO instead of LRU)")

    def test_38_relaxed_stale_cache_data_can_count_as_correct(self) -> None:
        """Finding L / Finding 38: Strict benchmark verification must reject relaxed stale cache hits."""
        cache = ToolResultCache(default_ttl_seconds=0.01)
        cache.put("tool_1", {"id": 1}, output="stale_val", freshness_contract="relaxed")
        time.sleep(0.02)  # expire TTL
        # Under strict scientific verification, stale data must NOT be returned as fresh hit
        _val, hit, _fresh = cache.get("tool_1", {"id": 1}, strict_verification=True)
        self.assertFalse(hit, "Cache returned expired stale entry under strict verification mode")

    def test_39_cache_entries_cross_tenant_or_authority_scope(self) -> None:
        """Finding L / Finding 39: Cache entries must be strictly scoped by tenant and authority."""
        cache = ToolResultCache()
        cache.put("sensitive_query", {"q": "balance"}, output=1000, tenant="tenant_A")
        # Tenant B queries identical tool and arguments
        _val, hit, _ = cache.get("sensitive_query", {"q": "balance"}, tenant="tenant_B")
        self.assertFalse(hit, "Cache leaked entries across tenant isolation boundary")

    def test_40_composite_claims_caching_while_bypassing_cache_lookup(self) -> None:
        """Finding M / Finding 40: Composite scheduler must visibly route read tool calls through cache."""
        from toolspeed.schedulers.composite import CompositeScheduler

        scheduler = CompositeScheduler()
        self.assertTrue(
            scheduler.has_cache_lookup_in_dispatch_path(),
            "Composite scheduler claims caching but bypasses cache in execution path",
        )

    # -------------------------------------------------------------------------
    # Finding N: Local tool security & isolation
    # -------------------------------------------------------------------------
    async def test_41_cancelling_subprocess_coroutine_leaves_work_running(self) -> None:
        """Finding N / Finding 41: Cancelling SafeSubprocessSandbox coroutine must terminate child process tree."""
        sandbox = SafeSubprocessSandbox(default_timeout_s=5.0)
        call = ToolCall("c1", "subprocess_sandbox", {"command": "sleep 10"})
        task = asyncio.create_task(sandbox.execute(call))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        # Verify no orphan sleep process remains
        self.assertTrue(
            sandbox.is_process_tree_terminated(call.call_id),
            "Cancelled subprocess coroutine left orphan process running in background",
        )

    def test_42_file_containment_allows_prefix_or_symlink_escape(self) -> None:
        """Finding N / Finding 42: AsyncLocalFileIOTool path resolution must resist prefix confusion."""
        with tempfile.TemporaryDirectory() as base_tmp:
            sandbox_base = Path(base_tmp) / "sandbox"
            sandbox_base.mkdir()
            # Evil directory sharing string prefix: "sandbox_evil" starts with "sandbox"!
            evil_dir = Path(base_tmp) / "sandbox_evil"
            evil_dir.mkdir()
            evil_file = evil_dir / "secret.txt"
            evil_file.write_text("pwned")

            sandbox = AsyncLocalFileIOTool(base_dir=str(sandbox_base))
            # Attempt to access ../sandbox_evil/secret.txt
            with self.assertRaises(ValueError):
                # BUG in e3c974b: str(target).startswith(str(base_dir)) allows sandbox_evil!
                sandbox._resolve_safe("../sandbox_evil/secret.txt")

    # -------------------------------------------------------------------------
    # Finding O: Wheel packaging & schema validation
    # -------------------------------------------------------------------------
    def test_43_wheel_cannot_load_protocol_outside_source_tree(self) -> None:
        """Finding O / Finding 43: Protocol must be packaged as package resource, not repo-relative path."""
        # Loading frozen protocol must work via importlib.resources or package resources
        from toolspeed.core.protocol import load_package_protocol

        protocol = load_package_protocol()
        self.assertIsNotNone(protocol, "Failed to load frozen protocol from package resources")

    def test_44_malformed_mechanism_body_passes_protocol_validation(self) -> None:
        """Finding O / Finding 44: JSON schema validator must reject mechanisms missing required fields."""
        malformed_dict = {
            "plan_id": "test",
            "plan_version": "1.2.0",
            "mechanisms": {
                "W1": {
                    "mechanism_type": "UNKNOWN_TYPE",  # Unknown mechanism type
                    # Missing candidate, baselines, thresholds
                }
            },
        }
        errors = validate_protocol_dict(malformed_dict)
        self.assertTrue(len(errors) > 0, "Schema validator accepted malformed mechanism dictionary")

    def test_45_missing_required_metric_becomes_favourable_default(self) -> None:
        """Finding F / Finding 45: Missing required metrics must yield INCONCLUSIVE, not fallback to 1.0."""
        harness = BenchmarkHarness()
        # If cost_multiplier is None for a required metric, decision must be INCONCLUSIVE
        summary_missing_cost = harness.evaluate_summary_metrics(
            p95_speedup=1.5,
            candidate_success=1.0,
            cost_multiplier=None,  # Missing!
            required_metrics=["cost_multiplier", "p95_speedup"],
        )
        self.assertEqual(
            summary_missing_cost.verdict_state,
            VerdictState.INCONCLUSIVE,
            "Missing cost metric defaulted favourably instead of marking verdict INCONCLUSIVE",
        )


if __name__ == "__main__":
    unittest.main()
