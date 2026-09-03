"""Tests for Workload W2 dynamic dependency compilation with shared local SQLite DB."""

from __future__ import annotations

import unittest

from toolspeed.benchmarks.local_backend import LocalWallClockBackend
from toolspeed.schedulers.b1_sync_react import SyncReActScheduler
from toolspeed.schedulers.base import ExecutionContext, SchedulerConfig
from toolspeed.workloads.w2_chains import (
    CompiledExecutionPlan,
    W2ComparisonResult,
    W2DynamicDependencyCompiler,
    evaluate_w2_compilation_vs_step_by_step,
    execute_compiled_plan,
)


class TestWorkloadW2(unittest.IsolatedAsyncioTestCase):
    """Verifies dynamic dependency compilation, separate compilation measurement, and lack of precomputed shortcut."""

    async def asyncSetUp(self) -> None:
        self.backend = LocalWallClockBackend(seed=42)

    async def asyncTearDown(self) -> None:
        self.backend.cleanup()

    def test_01_dynamic_dependency_compilation_measures_compilation_separately(self) -> None:
        """Compiler builds plan and separately measures compilation_time_ms."""
        task = self.backend.generate_task("W2", trial_index=0)
        compiler = W2DynamicDependencyCompiler()
        plan = compiler.compile(task)

        self.assertIsInstance(plan, CompiledExecutionPlan)
        self.assertEqual(len(plan.steps), 2)
        self.assertGreaterEqual(plan.compilation_time_ms, 0.0)
        self.assertEqual(plan.steps[0].tool_name, "fetch_user")
        self.assertEqual(plan.steps[1].tool_name, "fetch_orders")
        self.assertIn("step_fetch_user", plan.steps[1].dependencies)

    async def test_02_compiled_plan_executes_tools_without_precomputed_shortcut(self) -> None:
        """Candidate executes compiled plan tools via ToolExecutor against SQLite; no precomputed shortcut."""
        task = self.backend.generate_task("W2", trial_index=0)
        tools, _ = self.backend.create_workload_environment("W2", trial_index=0, arm="candidate")

        compiler = W2DynamicDependencyCompiler()
        plan = compiler.compile(task)

        task_model = task.to_model_task() if hasattr(task, "to_model_task") else task
        ctx = ExecutionContext(task=task_model, tools=tools, config=SchedulerConfig())

        output, exec_ms, results = await execute_compiled_plan(
            plan=plan,
            executor=ctx.executor,
            context=task.context,
        )

        self.assertGreater(exec_ms, 0.0)
        self.assertEqual(len(results), 2)
        # Verify tool executions returned real database data
        self.assertEqual(results[0].tool_name, "fetch_user")
        self.assertIn("user_id", results[0].output)
        self.assertEqual(results[1].tool_name, "fetch_orders")
        self.assertIn("orders", results[1].output)
        self.assertEqual(output.get("status"), "compiled_complete")

    async def test_03_compare_step_by_step_dispatch_vs_compiled_static_plan(self) -> None:
        """Compares step-by-step serial dispatch with compiled static plan on SQLite DB."""
        scheduler = SyncReActScheduler()
        result = await evaluate_w2_compilation_vs_step_by_step(
            backend=self.backend,
            baseline_scheduler=scheduler,
            trial_index=0,
        )

        self.assertIsInstance(result, W2ComparisonResult)
        self.assertGreater(result.step_by_step_duration_ms, 0.0)
        self.assertGreater(result.compiled_execution_ms, 0.0)
        self.assertGreaterEqual(result.compiled_compilation_ms, 0.0)
        self.assertEqual(result.tools_executed_count, 2)
        self.assertTrue(result.outputs_match)
        # Compiled plan bypasses 3 LLM decision delays -> positive speedup
        self.assertGreater(result.speedup, 1.0)


if __name__ == "__main__":
    unittest.main()
