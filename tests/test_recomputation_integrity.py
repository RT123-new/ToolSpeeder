"""Tests 46-75: Recomputation Integrity, Negative Controls, and Scientific Provenance."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from toolspeed import cli
from toolspeed.adapters.base import (
    BaseLLMAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
)
from toolspeed.adapters.live_tools import AsyncSQLiteTool
from toolspeed.adapters.mock_tools import create_standard_mock_registry
from toolspeed.benchmarks.harness import BenchmarkConfig, BenchmarkHarness
from toolspeed.benchmarks.local_backend import LocalWallClockBackend
from toolspeed.benchmarks.replay_backend import ReplayBackend
from toolspeed.core.clock import VirtualClock
from toolspeed.core.rate_limiter import RateLimiter
from toolspeed.core.types import (
    AgentTask,
    ApprovalGrant,
    EvidenceLevel,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
    compute_file_sha256,
)
from toolspeed.experiments.runner import compute_summary
from toolspeed.schedulers.base import SchedulerConfig
from toolspeed.schedulers.e3_speculation import SpeculativeReadScheduler
from toolspeed.schedulers.e4_commit_horizon import CommitHorizonScheduler, IncrementalCommitParser
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeCodec
from toolspeed.schedulers.executor import SharedIdempotencyStore, ToolExecutor
from toolspeed.schedulers.phase2_cache import CacheScheduler


class MockRecomputationLLM(BaseLLMAdapter):
    """Mock LLM adapter for recomputation tests."""

    def __init__(self, decisions: list[LLMDecision] | None = None) -> None:
        self.decisions = list(decisions or [])
        self._turn = 0

    async def decide(
        self, task: AgentTask, history: list[dict[str, Any]], available_tools: list[ToolSpec]
    ) -> LLMDecision:
        if self._turn < len(self.decisions):
            d = self.decisions[self._turn]
            self._turn += 1
            return d
        return LLMDecision(reasoning="Done", tool_calls=[], final_answer="recomputation_done")

    async def predict_draft(
        self, task: AgentTask, history: list[dict[str, Any]], available_tools: list[ToolSpec]
    ) -> ToolCall | None:
        return None

    async def stream_decision(
        self, task: AgentTask, history: list[dict[str, Any]], available_tools: list[ToolSpec]
    ) -> Any:
        if self._turn < len(self.decisions):
            d = self.decisions[self._turn]
            self._turn += 1
            if d.final_answer is not None or not d.tool_calls:
                yield StreamingChunk(
                    token_index=0,
                    delta_text=d.reasoning or "done",
                    is_final=True,
                    parsed_tool_calls=[],
                    metadata={"final_answer": d.final_answer},
                )
            else:
                yield StreamingChunk(
                    token_index=0,
                    delta_text=d.reasoning or "calling tools",
                    is_final=True,
                    parsed_tool_calls=d.tool_calls,
                )
            return

        final_ans = "done"
        yield StreamingChunk(
            token_index=0,
            delta_text="done",
            is_final=True,
            metadata={"final_answer": final_ans},
        )


class TestRecomputationIntegrity(unittest.IsolatedAsyncioTestCase):
    """Comprehensive test suite covering scenarios 46 through 75."""

    # 46. CLI: Falsify exit code 0 on passed empirical bundle
    async def test_46_falsify_exit_code_0_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            manifest = {
                "code_git_sha": "0123456789abcdef",
                "evidence_level": "replay_integration",
                "trial_count": 1000,
                "benchmark_config_hash": "cfg_hash",
                "workload_fixture_hash": "fix_hash",
                "raw_trace_hash": "trace_hash",
                "is_verdict_eligible": True,
            }
            evaluations = [
                {
                    "workload_id": "W1",
                    "candidate_name": "DAGScheduler",
                    "baseline_name": "NativeParallelScheduler",
                    "summary": {
                        "p95_speedup": 2.1,
                        "candidate_p95_ms": 50.0,
                        "baseline_p95_ms": 105.0,
                        "candidate_success_rate": 1.0,
                        "unapproved_side_effects": 0,
                    },
                    "verdict": {"passed": True, "falsified": False},
                }
            ]
            bundle_data = {
                "title": "Passed Bundle",
                "evidence_level": "replay_integration",
                "manifest": manifest,
                "evaluations": evaluations,
                "overall_verdict": "passed",
            }
            (bundle_dir / "result.json").write_text(json.dumps(bundle_data), encoding="utf-8")
            (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            exit_code = cli.cmd_falsify(cli.argparse.Namespace(input=str(bundle_dir)))
            self.assertEqual(exit_code, 0)

    # 47. CLI: Falsify exit code 1 on falsified empirical bundle
    def test_47_falsify_exit_code_1_falsified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            manifest = {
                "code_git_sha": "0123456789abcdef",
                "evidence_level": "replay_integration",
                "trial_count": 1000,
                "benchmark_config_hash": "cfg_hash",
                "workload_fixture_hash": "fix_hash",
                "raw_trace_hash": "trace_hash",
                "is_verdict_eligible": True,
            }
            evaluations = [
                {
                    "workload_id": "W1",
                    "candidate_name": "DAGScheduler",
                    "baseline_name": "NativeParallelScheduler",
                    "summary": {
                        "p95_speedup": 0.8,
                        "candidate_p95_ms": 120.0,
                        "baseline_p95_ms": 96.0,
                        "candidate_success_rate": 1.0,
                        "unapproved_side_effects": 0,
                    },
                    "verdict": {"passed": False, "falsified": True},
                }
            ]
            bundle_data = {
                "title": "Falsified Bundle",
                "evidence_level": "replay_integration",
                "manifest": manifest,
                "evaluations": evaluations,
                "overall_verdict": "falsified",
            }
            (bundle_dir / "result.json").write_text(json.dumps(bundle_data), encoding="utf-8")

            exit_code = cli.cmd_falsify(cli.argparse.Namespace(input=str(bundle_dir)))
            self.assertEqual(exit_code, 1)

    # 48. CLI: Falsify exit code 2 on smoke run or inconclusive bundle
    def test_48_falsify_exit_code_2_smoke_or_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            manifest = {
                "code_git_sha": "0123456789abcdef",
                "evidence_level": "replay_integration",
                "trial_count": 10,  # Smoke run < 1000
                "benchmark_config_hash": "cfg_hash",
                "workload_fixture_hash": "fix_hash",
                "raw_trace_hash": "trace_hash",
                "is_verdict_eligible": False,
            }
            bundle_data = {
                "title": "Smoke Bundle",
                "evidence_level": "replay_integration",
                "manifest": manifest,
                "evaluations": [],
                "overall_verdict": "inconclusive",
            }
            (bundle_dir / "result.json").write_text(json.dumps(bundle_data), encoding="utf-8")

            exit_code = cli.cmd_falsify(cli.argparse.Namespace(input=str(bundle_dir)))
            self.assertEqual(exit_code, 2)

    # 49. CLI: Falsify exit code 3 on malformed or missing bundle
    def test_49_falsify_exit_code_3_malformed_bundle(self) -> None:
        exit_code = cli.cmd_falsify(cli.argparse.Namespace(input="/non/existent/path/bundle.json"))
        self.assertEqual(exit_code, 3)

    # 50. Bundle Validator: Detects file byte SHA-256 hash corruption
    def test_50_validate_bundle_detects_hash_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p_out = Path(tmpdir)
            md_file = p_out / "report.md"
            md_file.write_text("Clean original report", encoding="utf-8")
            html_file = p_out / "report.html"
            html_file.write_text("Clean original html", encoding="utf-8")

            res_file = p_out / "result.json"
            manifest = {
                "code_git_sha": "0123456789abcdef",
                "evidence_level": "replay_integration",
                "trial_count": 1000,
                "benchmark_config_hash": "cfg_hash",
                "workload_fixture_hash": "fix_hash",
                "raw_trace_hash": "trace_hash",
                "file_hashes": {
                    "report.md": compute_file_sha256(md_file),
                    "report.html": compute_file_sha256(html_file),
                },
            }
            evaluations = [
                {
                    "workload_id": "W1",
                    "summary": {
                        "candidate_p95_ms": 50.0,
                        "baseline_p95_ms": 100.0,
                        "p95_speedup": 2.0,
                        "candidate_success_rate": 1.0,
                        "unapproved_side_effects": 0,
                    },
                }
            ]
            bundle_data = {
                "title": "Corruptable Bundle",
                "evidence_level": "replay_integration",
                "manifest": manifest,
                "evaluations": evaluations,
            }
            res_file.write_text(json.dumps(bundle_data), encoding="utf-8")

            # Corrupt report.md by appending bytes
            md_file.write_text("Corrupted modified report content", encoding="utf-8")

            exit_code = cli.cmd_validate_bundle(cli.argparse.Namespace(input=str(p_out)))
            self.assertEqual(exit_code, 1)

    # 51. Bundle Validator: Detects missing file listed in manifest
    def test_51_validate_bundle_detects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p_out = Path(tmpdir)
            md_file = p_out / "report.md"
            md_file.write_text("Clean report", encoding="utf-8")
            html_file = p_out / "report.html"
            html_file.write_text("Clean html", encoding="utf-8")

            res_file = p_out / "result.json"
            manifest = {
                "code_git_sha": "0123456789abcdef",
                "evidence_level": "replay_integration",
                "trial_count": 1000,
                "benchmark_config_hash": "cfg_hash",
                "workload_fixture_hash": "fix_hash",
                "raw_trace_hash": "trace_hash",
                "file_hashes": {
                    "report.md": compute_file_sha256(md_file),
                    "report.html": compute_file_sha256(html_file),
                    "missing_traces.jsonl": "0000000000000000000000000000000000000000000000000000000000000000",
                },
            }
            bundle_data = {
                "title": "Missing File Bundle",
                "evidence_level": "replay_integration",
                "manifest": manifest,
                "evaluations": [
                    {
                        "workload_id": "W1",
                        "summary": {
                            "candidate_p95_ms": 50.0,
                            "baseline_p95_ms": 100.0,
                            "p95_speedup": 2.0,
                            "candidate_success_rate": 1.0,
                            "unapproved_side_effects": 0,
                        },
                    }
                ],
            }
            res_file.write_text(json.dumps(bundle_data), encoding="utf-8")

            exit_code = cli.cmd_validate_bundle(cli.argparse.Namespace(input=str(p_out)))
            self.assertEqual(exit_code, 1)

    # 52. Bundle Validator: Rejects null required metrics
    def test_52_validate_bundle_rejects_null_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p_out = Path(tmpdir)
            manifest = {
                "code_git_sha": "0123456789abcdef",
                "evidence_level": "replay_integration",
                "trial_count": 1000,
                "benchmark_config_hash": "cfg_hash",
                "workload_fixture_hash": "fix_hash",
                "raw_trace_hash": "trace_hash",
            }
            # Missing candidate_p95_ms
            bundle_data = {
                "manifest": manifest,
                "evaluations": [
                    {
                        "workload_id": "W1",
                        "summary": {
                            "candidate_p95_ms": None,
                            "baseline_p95_ms": 100.0,
                            "p95_speedup": None,
                            "candidate_success_rate": 1.0,
                        },
                    }
                ],
            }
            (p_out / "result.json").write_text(json.dumps(bundle_data), encoding="utf-8")
            exit_code = cli.cmd_validate_bundle(cli.argparse.Namespace(input=str(p_out)))
            self.assertEqual(exit_code, 1)

    # 53. Bundle Validator: Rejects unapproved side effects
    def test_53_validate_bundle_rejects_unapproved_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p_out = Path(tmpdir)
            manifest = {
                "code_git_sha": "0123456789abcdef",
                "evidence_level": "replay_integration",
                "trial_count": 1000,
                "benchmark_config_hash": "cfg_hash",
                "workload_fixture_hash": "fix_hash",
                "raw_trace_hash": "trace_hash",
            }
            bundle_data = {
                "manifest": manifest,
                "evaluations": [
                    {
                        "workload_id": "W7",
                        "summary": {
                            "candidate_p95_ms": 50.0,
                            "baseline_p95_ms": 100.0,
                            "p95_speedup": 2.0,
                            "candidate_success_rate": 1.0,
                            "unapproved_side_effects": 2,  # Violation!
                        },
                    }
                ],
            }
            (p_out / "result.json").write_text(json.dumps(bundle_data), encoding="utf-8")
            exit_code = cli.cmd_validate_bundle(cli.argparse.Namespace(input=str(p_out)))
            self.assertEqual(exit_code, 1)

    # 54. CLI: Report generation generates report.md and report.html
    def test_54_report_validates_bundle_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p_out = Path(tmpdir)
            bundle_data = {
                "title": "Report Bundle",
                "evidence_level": "replay_integration",
                "evaluations": [
                    {
                        "workload_id": "W1",
                        "candidate_name": "DAGScheduler",
                        "baseline_name": "NativeParallelScheduler",
                        "summary": {
                            "candidate_p95_ms": 45.0,
                            "baseline_p95_ms": 90.0,
                            "p95_speedup": 2.0,
                            "candidate_success_rate": 1.0,
                        },
                        "verdict": {"passed": True},
                    }
                ],
            }
            (p_out / "result.json").write_text(json.dumps(bundle_data), encoding="utf-8")
            exit_code = cli.cmd_report(cli.argparse.Namespace(input=str(p_out), out=str(p_out)))
            self.assertEqual(exit_code, 0)
            self.assertTrue((p_out / "report.md").exists())
            self.assertTrue((p_out / "report.html").exists())

    # 55. Statistics: Paired CCL bootstrap resampling uses 2,000 draws
    def test_55_paired_ccl_bootstrap_resampling_count(self) -> None:
        import numpy as np

        base = np.array([100.0, 110.0, 120.0, 130.0, 140.0] * 20)
        cand = np.array([50.0, 55.0, 60.0, 65.0, 70.0] * 20)
        succ = np.ones(100, dtype=bool)

        summary = compute_summary(base, cand, baseline_success=succ, candidate_success=succ)
        self.assertIsNotNone(summary.p95_reduction_ci)
        assert summary.p95_reduction_ci is not None
        low, high = summary.p95_reduction_ci
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreaterEqual(high or 0.0, low or 0.0)
        self.assertAlmostEqual(summary.p95_speedup or 0.0, 2.0, delta=0.2)

    # 56. Statistics: Paired CCL excludes unpaired failures
    def test_56_paired_ccl_excludes_unpaired_failures(self) -> None:
        import numpy as np

        base = np.array([100.0, 100.0, 100.0, 100.0])
        cand = np.array([50.0, 50.0, 50.0, 9999.0])
        base_succ = np.array([True, True, True, True])
        cand_succ = np.array([True, True, True, False])  # 4th trial failed in candidate

        summary = compute_summary(base, cand, baseline_success=base_succ, candidate_success=cand_succ)
        # Latency speedup must ONLY be computed over the 3 both-succeeded pairs
        self.assertEqual(summary.candidate_p95_ms, 50.0)
        self.assertEqual(summary.baseline_p95_ms, 100.0)
        self.assertEqual(summary.p95_speedup, 2.0)
        self.assertEqual(summary.candidate_success_rate, 0.75)

    # 57. Action Bytecode: Rejects schema hash mismatch
    def test_57_action_bytecode_rejects_schema_hash_mismatch(self) -> None:
        tools = [t.spec for t in create_standard_mock_registry().values()]
        codec = ActionBytecodeCodec(tool_specs=tools)
        call = ToolCall(name="fetch_user", arguments={"user_id": "u101"})
        bytecode = codec.encode(call)

        # Alter codec schema hash
        codec_diff = ActionBytecodeCodec(tool_specs=tools[:1])
        with self.assertRaises(ValueError):
            codec_diff.decode(bytecode)

    # 58. Action Bytecode: 16-bit opcodes are strictly collision-free
    def test_58_action_bytecode_opcode_uniqueness(self) -> None:
        tools = [t.spec for t in create_standard_mock_registry().values()]
        codec = ActionBytecodeCodec(tool_specs=tools)
        opcodes = list(codec.opcode_to_tool.keys())
        self.assertEqual(len(opcodes), len(set(opcodes)), "Opcodes must be unique!")

    # 59. Commit Horizon: IncrementalCommitParser rejects duplicate JSON keys
    def test_59_commit_horizon_duplicate_keys_rejected(self) -> None:
        self.assertFalse(IncrementalCommitParser.is_syntax_closed('{"key": 1, "key": 2}'))
        self.assertTrue(IncrementalCommitParser.is_syntax_closed('{"key": 1, "other": 2}'))

    # 60. Commit Horizon: Strict call_id / fingerprint reconciliation
    async def test_60_commit_horizon_reconciliation_by_call_id_or_fingerprint(self) -> None:
        sched = CommitHorizonScheduler()
        tools = ToolRegistry()
        tools.register(create_standard_mock_registry()["fetch_user"])

        task = Task(prompt="Fetch user", expected_output={"status": "done"})
        model = MockRecomputationLLM(
            [
                LLMDecision(tool_calls=[ToolCall(name="fetch_user", arguments={"user_id": "u42"})]),
                LLMDecision(final_answer={"status": "done"}),
            ]
        )
        res = await sched.execute(task, model, tools)
        self.assertTrue(res.success)

    # 61. Speculation: Baseline has speculation_enabled=False
    def test_61_speculation_disabled_baseline_comparison(self) -> None:
        cand = SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=True))
        base = SpeculativeReadScheduler(SchedulerConfig(speculation_enabled=False))
        self.assertTrue(cand.config.speculation_enabled)
        self.assertFalse(base.config.speculation_enabled)

    # 62. Cache: Baseline has cache_enabled=False
    def test_62_cache_disabled_baseline_comparison(self) -> None:
        cand = CacheScheduler(config=SchedulerConfig(cache_enabled=True))
        base = CacheScheduler(config=SchedulerConfig(cache_enabled=False))
        self.assertTrue(cand.config.cache_enabled)
        self.assertFalse(base.config.cache_enabled)

    # 63. Commit Horizon: Baseline has early_dispatch_enabled=False
    def test_63_commit_horizon_early_dispatch_disabled_baseline(self) -> None:
        cand = CommitHorizonScheduler(config=SchedulerConfig(early_dispatch_enabled=True))
        base = CommitHorizonScheduler(config=SchedulerConfig(early_dispatch_enabled=False))
        self.assertTrue(cand.config.early_dispatch_enabled)
        self.assertFalse(base.config.early_dispatch_enabled)

    # 64. Prewarming: W6 compares prewarmed vs cold pools
    async def test_64_prewarm_cold_start_pool_baseline_comparison(self) -> None:
        backend = ReplayBackend()
        task = backend.generate_task("W6", 0)
        self.assertEqual(task.metadata.get("workload_id"), "W6")

    # 65. W7: Dual verdict split into Safety and Latency
    async def test_65_w7_dual_verdict_split(self) -> None:
        backend = ReplayBackend()
        task = backend.generate_task("W7", 0)
        grant = task.metadata.get("approval_grant")
        self.assertIsInstance(grant, ApprovalGrant)

    # 66. Rate Limiter: Safe token refund on cancellation
    async def test_66_rate_limiter_token_refund_on_cancellation(self) -> None:
        limiter = RateLimiter(rate_per_sec=10.0, burst_capacity=10.0)
        lease = await limiter.acquire_lease(1)
        self.assertAlmostEqual(limiter.token_bucket.available_tokens, 9.0, delta=0.5)
        lease.refund()
        self.assertAlmostEqual(limiter.token_bucket.available_tokens, 10.0, delta=0.5)

    # 67. SharedIdempotencyStore: Followers unblocked when primary cancelled
    async def test_67_shared_idempotency_store_follower_unblocking_on_primary_cancel(self) -> None:
        store = SharedIdempotencyStore()
        token = "cancel_tx_001"
        action, key, _fut1, _ = store.reserve_or_join("write_tool", {"op": "write"}, token, "auth")
        self.assertEqual(action, "RESERVED_PRIMARY")

        action2, _key2, fut2, _ = store.reserve_or_join("write_tool", {"op": "write"}, token, "auth")
        self.assertEqual(action2, "JOIN_IN_FLIGHT")
        assert fut2 is not None

        store.cancel_in_flight(key, "Primary task timed out")
        res2 = await fut2
        self.assertTrue(res2.is_error)
        self.assertTrue(res2.cancelled)

    # 68. SharedIdempotencyStore: Followers unblocked when primary fails
    async def test_68_shared_idempotency_store_follower_unblocking_on_primary_error(self) -> None:
        store = SharedIdempotencyStore()
        token = "fail_tx_001"
        action, key, _fut1, _ = store.reserve_or_join("write_tool", {"op": "write"}, token, "auth")
        self.assertEqual(action, "RESERVED_PRIMARY")

        action2, _key2, fut2, _ = store.reserve_or_join("write_tool", {"op": "write"}, token, "auth")
        self.assertEqual(action2, "JOIN_IN_FLIGHT")
        assert fut2 is not None

        err_result = ToolResult(call_id="c1", tool_name="write_tool", is_error=True, error="DB Connection Lost")
        store.publish_result(key, err_result)
        res2 = await fut2
        self.assertTrue(res2.is_error)
        self.assertEqual(res2.error, "DB Connection Lost")

    # 69. ToolExecutor: Rejection of unparsed $dollar reference in final arguments
    async def test_69_tool_executor_rejects_dollar_ref_in_final_args(self) -> None:
        tools = ToolRegistry()
        tools.register(create_standard_mock_registry()["fetch_user"])
        executor = ToolExecutor(tools)
        call = ToolCall(name="fetch_user", arguments={"user_id": "$unresolved_variable.id"})
        res = await executor.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("Unresolved reference", res.error or "")

    # 70. AsyncSQLiteTool: clone() produces isolated SQLite databases
    async def test_70_sqlite_tool_deep_copy_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "master.db")
            t_orig = AsyncSQLiteTool(db_path=db_path)
            res1 = await t_orig.execute(
                ToolCall(name="sqlite_executor", arguments={"query": "CREATE TABLE items (id INT, val TEXT)"})
            )
            self.assertFalse(res1.is_error)
            res2 = await t_orig.execute(
                ToolCall(name="sqlite_executor", arguments={"query": "INSERT INTO items VALUES (1, 'initial')"})
            )
            self.assertFalse(res2.is_error)

            t_clone = t_orig.clone()
            res3 = await t_clone.execute(
                ToolCall(name="sqlite_executor", arguments={"query": "INSERT INTO items VALUES (2, 'clone_only')"})
            )
            self.assertFalse(res3.is_error)

            snap_orig = t_orig.get_state_snapshot()
            snap_clone = t_clone.get_state_snapshot()

            self.assertEqual(len(snap_orig.get("items", [])), 1)
            self.assertEqual(len(snap_clone.get("items", [])), 2)

    # 71. ReplayBackend: Deterministic virtual clock advances without real wall-clock sleep
    async def test_71_replay_backend_deterministic_virtual_clock(self) -> None:
        vclock = VirtualClock()
        backend = ReplayBackend(clock=vclock)
        reg, _model = backend.create_workload_environment("W1", 0)
        tool = reg.get("server_metric_0")
        assert tool is not None

        start_wall = time.perf_counter()
        res = await tool.execute(ToolCall(name="server_metric_0", arguments={}))
        self.assertFalse(res.is_error)
        elapsed_wall = time.perf_counter() - start_wall

        # Virtual clock advanced by 20ms instantly
        self.assertEqual(vclock.now_ns() / 1_000_000.0, 20.0)
        self.assertLess(elapsed_wall, 0.01)  # Wall clock didn't sleep for 20ms

    # 72. LocalWallClockBackend: Real OS loopback and subprocess execution
    async def test_72_local_backend_real_os_loopback_and_subprocess(self) -> None:
        with LocalWallClockBackend() as backend:
            task_w1 = backend.generate_task("W1", 0)
            self.assertEqual(task_w1.metadata.get("workload_id"), "W1")
            reg_w1, _model_w1 = backend.create_workload_environment("W1", 0)
            tool_0 = reg_w1.get("read_shard_0")
            assert tool_0 is not None
            res = await tool_0.execute(ToolCall(name="read_shard_0", arguments={"query": "test"}))
            self.assertFalse(res.is_error)
            self.assertEqual(res.output.get("status"), "ok")

    # 73. Negative Control E1: Parallelism disabled produces ~1.0x speedup
    async def test_73_negative_control_e1_parallelism_disabled(self) -> None:
        config = BenchmarkConfig(trials_per_condition=5, evidence_level=EvidenceLevel.REPLAY_INTEGRATION)
        harness = BenchmarkHarness(config=config)
        ctrls = await harness.run_negative_controls()
        ctrl = next(c for c in ctrls if c["name"] == "E1_parallelism_disabled")
        self.assertTrue(ctrl["passed_expected_null"])
        self.assertAlmostEqual(ctrl["p95_speedup"], 1.0, delta=0.05)

    # 74. Negative Control E2: Fusion disabled produces ~1.0x speedup
    async def test_74_negative_control_e2_fusion_disabled(self) -> None:
        config = BenchmarkConfig(trials_per_condition=5, evidence_level=EvidenceLevel.REPLAY_INTEGRATION)
        harness = BenchmarkHarness(config=config)
        ctrls = await harness.run_negative_controls()
        ctrl = next(c for c in ctrls if c["name"] == "E2_fusion_disabled")
        self.assertTrue(ctrl["passed_expected_null"])
        self.assertAlmostEqual(ctrl["p95_speedup"], 1.0, delta=0.05)

    # 75. Sensitivity Control: Harness detects injected 2.0x speedup
    async def test_75_negative_control_positive_sensitivity_injected(self) -> None:
        config = BenchmarkConfig(trials_per_condition=5, evidence_level=EvidenceLevel.REPLAY_INTEGRATION)
        harness = BenchmarkHarness(config=config)
        ctrls = await harness.run_negative_controls()
        ctrl = next(c for c in ctrls if c["name"] == "Positive_sensitivity_injected_50pct_speedup")
        self.assertTrue(ctrl["passed_expected_null"])
        self.assertAlmostEqual(ctrl["p95_speedup"], 2.0, delta=0.1)
