"""Unit tests for all 7 ToolSpeed workload families (W1 to W7)."""

import asyncio
import unittest

from toolspeed.core.types import ExecutionTrace, ToolCall, ToolResult
from toolspeed.workloads import (
    W1IndependentWorkload,
    W2ChainsWorkload,
    W3BranchingWorkload,
    W4LocalityWorkload,
    W5LargePayloadsWorkload,
    W6ColdStartWorkload,
    W7SideEffectsWorkload,
)


class TestWorkloadFamilies(unittest.IsolatedAsyncioTestCase):
    """Test task generation, tool execution, and exact validation across W1-W7."""

    async def test_w1_independent_fanout(self):
        w1 = W1IndependentWorkload(fan_out_widths=[2, 4], median_tool_ms=1.0)
        spec = w1.get_spec()
        self.assertEqual(spec.family, "w1_independent")

        tasks = w1.generate_tasks(count=5, seed=42)
        self.assertEqual(len(tasks), 5)

        tools = w1.get_tools()
        self.assertEqual(len(tools), 1)
        tool = tools[0]

        validator = w1.get_validator()

        # Run task with exact tool calls
        task = tasks[0]
        servers = task.parameters["servers"]
        results = []
        calls = []
        for s in servers:
            call = ToolCall(tool_name="query_server_load", arguments={"server_id": s})
            res = await tool.execute(call)
            calls.append(call)
            results.append(res)

        total_load = sum(r.result["load_pct"] for r in results)
        output = {"server_count": len(servers), "total_load": total_load, "servers": servers}

        trace = ExecutionTrace(
            task_id=task.task_id,
            tool_calls=calls,
            tool_results=results,
            success=True,
        )

        valid, msg, _ = validator.validate(task, output, trace)
        self.assertTrue(valid, msg)

        # Invalid output test
        bad_output = {"server_count": len(servers), "total_load": -999}
        valid_bad, _, _ = validator.validate(task, bad_output, trace)
        self.assertFalse(valid_bad)

    async def test_w2_dependent_chains(self):
        w2 = W2ChainsWorkload(chain_depths=[2, 4], median_step_ms=1.0)
        tasks = w2.generate_tasks(count=4, seed=42)
        task = tasks[0]
        depth = task.parameters["depth"]

        tools = w2.get_tools()
        tool = tools[0]
        validator = w2.get_validator()

        current_val = task.parameters["initial_input"]
        calls = []
        results = []
        for s in range(depth):
            call = ToolCall(tool_name="execute_pipeline_step", arguments={"step_index": s, "input_val": current_val})
            res = await tool.execute(call)
            calls.append(call)
            results.append(res)
            current_val = res.result["output_val"]

        output = {"final_value": current_val, "depth": depth}
        trace = ExecutionTrace(task_id=task.task_id, tool_calls=calls, tool_results=results, success=True)

        valid, msg, _ = validator.validate(task, output, trace)
        self.assertTrue(valid, msg)

    async def test_w3_branching_workflows(self):
        w3 = W3BranchingWorkload(median_tool_ms=1.0)
        tasks = w3.generate_tasks(count=6, seed=42)
        tools = {t.get_schema().name: t for t in w3.get_tools()}
        validator = w3.get_validator()

        for task in tasks:
            tx_id = task.parameters["tx_id"]
            # 1. Step 1: risk check
            risk_call = ToolCall(tool_name="check_transaction_risk", arguments={"tx_id": tx_id})
            risk_res = await tools["check_transaction_risk"].execute(risk_call)
            risk_score = risk_res.result["risk_score"]

            calls = [risk_call]
            if risk_score < 35:
                app_call = ToolCall(tool_name="approve_standard", arguments={"tx_id": tx_id})
                await tools["approve_standard"].execute(app_call)
                calls.append(app_call)
                output = {"tx_id": tx_id, "branch": "low", "final_status": "APPROVED", "risk_score": risk_score}
            elif risk_score < 75:
                s1_call = ToolCall(tool_name="request_stepup_auth", arguments={"tx_id": tx_id})
                s1_res = await tools["request_stepup_auth"].execute(s1_call)
                s2_call = ToolCall(tool_name="verify_stepup_response", arguments={"challenge_id": s1_res.result["challenge_id"], "code": "123456"})
                await tools["verify_stepup_response"].execute(s2_call)
                calls.extend([s1_call, s2_call])
                output = {"tx_id": tx_id, "branch": "medium", "final_status": "STEPUP_VERIFIED", "risk_score": risk_score}
            else:
                q_call = ToolCall(tool_name="quarantine_transaction", arguments={"tx_id": tx_id})
                await tools["quarantine_transaction"].execute(q_call)
                f_call = ToolCall(tool_name="notify_fraud_team", arguments={"tx_id": tx_id, "risk_score": risk_score})
                await tools["notify_fraud_team"].execute(f_call)
                calls.extend([q_call, f_call])
                output = {"tx_id": tx_id, "branch": "high", "final_status": "QUARANTINED_AND_FLAGGED", "risk_score": risk_score}

            trace = ExecutionTrace(task_id=task.task_id, tool_calls=calls, success=True)
            valid, msg, _ = validator.validate(task, output, trace)
            self.assertTrue(valid, msg)

    async def test_w4_plan_locality(self):
        w4 = W4LocalityWorkload(num_entities=10, median_tool_ms=1.0)
        tasks = w4.generate_tasks(count=5, seed=42)
        tools = {t.get_schema().name: t for t in w4.get_tools()}
        validator = w4.get_validator()

        task = tasks[0]
        user_id = task.parameters["user_id"]
        base_amount = task.parameters["base_amount"]

        # Call profile lookup
        p_call = ToolCall(tool_name="lookup_user_profile", arguments={"user_id": user_id})
        p_res = await tools["lookup_user_profile"].execute(p_call)
        profile = p_res.result

        # Call invoice calc
        i_call = ToolCall(
            tool_name="calculate_final_invoice",
            arguments={
                "user_id": user_id,
                "tier": profile["tier"],
                "base_amount": base_amount,
                "discount_pct": profile["discount_pct"],
            },
        )
        i_res = await tools["calculate_final_invoice"].execute(i_call)

        output = {
            "user_id": user_id,
            "tier": profile["tier"],
            "final_price": i_res.result["final_price"],
        }
        trace = ExecutionTrace(task_id=task.task_id, tool_calls=[p_call, i_call], success=True)
        valid, msg, _ = validator.validate(task, output, trace)
        self.assertTrue(valid, msg)

    async def test_w5_large_payloads(self):
        w5 = W5LargePayloadsWorkload(payload_sizes_kb=[5], median_tool_ms=1.0)
        tasks = w5.generate_tasks(count=2, seed=42)
        tools = {t.get_schema().name: t for t in w5.get_tools()}
        validator = w5.get_validator()

        task = tasks[0]
        g_call = ToolCall(
            tool_name="generate_heavy_dataset",
            arguments={"num_rows": task.parameters["num_rows"], "seed": task.parameters["seed"]},
        )
        g_res = await tools["generate_heavy_dataset"].execute(g_call)

        a_call = ToolCall(
            tool_name="aggregate_heavy_dataset",
            arguments={"rows": g_res.result["rows"]},
        )
        a_res = await tools["aggregate_heavy_dataset"].execute(a_call)

        output = {
            "processed_rows": a_res.result["processed_rows"],
            "total_val_b": a_res.result["total_val_b"],
            "mean_val_a": a_res.result["mean_val_a"],
        }
        trace = ExecutionTrace(task_id=task.task_id, tool_calls=[g_call, a_call], success=True)
        valid, msg, _ = validator.validate(task, output, trace)
        self.assertTrue(valid, msg)

    async def test_w6_cold_start(self):
        w6 = W6ColdStartWorkload(cold_start_ms=10.0, warm_execution_ms=1.0)
        tasks = w6.generate_tasks(count=2, seed=42)
        tools = w6.get_tools()
        tool = tools[0]
        validator = w6.get_validator()

        task = tasks[0]
        call = ToolCall(tool_name="sandbox_python_eval", arguments={"expression": task.parameters["expression"]})
        res = await tool.execute(call)

        output = {"result": res.result["result"], "expression": task.parameters["expression"]}
        trace = ExecutionTrace(task_id=task.task_id, tool_calls=[call], success=True)
        valid, msg, _ = validator.validate(task, output, trace)
        self.assertTrue(valid, msg)

    async def test_w7_side_effects_and_approvals(self):
        w7 = W7SideEffectsWorkload(median_tool_ms=1.0)
        tasks = w7.generate_tasks(count=2, seed=42)
        tools = w7.get_tools()
        tool = tools[0]
        validator = w7.get_validator()

        task = tasks[0]
        params = task.parameters

        # 1. Unapproved call must fail
        unapproved_call = ToolCall(
            tool_name="execute_fund_transfer",
            arguments=params,
            requires_approval=True,
            is_approved=False,
        )
        res_unapproved = await tool.execute(unapproved_call)
        self.assertTrue(res_unapproved.is_error)

        # 2. Approved call succeeds
        approved_call = ToolCall(
            tool_name="execute_fund_transfer",
            arguments=params,
            requires_approval=True,
            is_approved=True,
        )
        res_approved = await tool.execute(approved_call)
        self.assertFalse(res_approved.is_error)
        self.assertEqual(res_approved.result["status"], "TRANSFERRED")

        output = {
            "status": res_approved.result["status"],
            "from_account": params["from_account"],
            "to_account": params["to_account"],
            "amount": params["amount"],
            "idempotency_key": params["idempotency_key"],
        }
        trace = ExecutionTrace(task_id=task.task_id, tool_calls=[approved_call], success=True)
        valid, msg, _ = validator.validate(task, output, trace)
        self.assertTrue(valid, msg)

        # 3. Replay with same idempotency key returns cached without double mutation
        res_replay = await tool.execute(approved_call)
        self.assertEqual(res_replay.result["from_balance"], res_approved.result["from_balance"])


if __name__ == "__main__":
    unittest.main()
