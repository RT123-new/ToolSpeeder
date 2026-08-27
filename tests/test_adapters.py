"""Unit tests for ToolSpeed mock and live tool adapters and LLM simulators."""

import asyncio
import os
import shutil
import tempfile
import time
import unittest

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    StreamingChunk,
    ToolSchema,
)
from toolspeed.adapters.mock_tools import (
    MockToolAdapter,
    MockToolConfig,
    MockToolEngine,
)
from toolspeed.adapters.live_tools import (
    AsyncHTTPClientTool,
    AsyncLocalFileIOTool,
    AsyncSQLiteTool,
    MockHTTPServer,
    SafeSubprocessSandbox,
)
from toolspeed.adapters.mock_models import (
    ActionBytecodeCodec,
    DraftPredictorModel,
    ModelCostConfig,
    SimulatedLLM,
)
from toolspeed.core.types import LatencyProfile, ToolCall, ToolResult


class TestMockTools(unittest.IsolatedAsyncioTestCase):
    """Test MockToolAdapter and MockToolEngine behavior."""

    async def test_mock_tool_execution_and_caching(self):
        config = MockToolConfig(
            name="echo_tool",
            median_ms=5.0,
            sigma=0.1,
            cache_ttl_s=10.0,
            handler=lambda args: {"echo": args.get("val")},
        )
        adapter = MockToolAdapter(config, seed=42)
        call = ToolCall(tool_name="echo_tool", arguments={"val": "hello"})

        # First call: cache miss, executes handler
        res1 = await adapter.execute(call)
        self.assertFalse(res1.cached)
        self.assertFalse(res1.is_error)
        self.assertEqual(res1.result, {"echo": "hello"})

        # Second call: cache hit, 0 cost
        res2 = await adapter.execute(call)
        self.assertTrue(res2.cached)
        self.assertEqual(res2.cost_usd, 0.0)
        self.assertEqual(res2.result, {"echo": "hello"})

    async def test_mock_tool_cold_start_and_warmup(self):
        config = MockToolConfig(
            name="cold_tool",
            median_ms=5.0,
            cold_start_ms=50.0,
        )
        adapter = MockToolAdapter(config, seed=42)
        self.assertFalse(adapter._is_warm)

        # Sampling before warmup should include cold start
        lat1 = adapter.sample_latency_ms()
        self.assertGreaterEqual(lat1, 50.0)
        self.assertTrue(adapter._is_warm)

        # Next sample should be warm
        lat2 = adapter.sample_latency_ms()
        self.assertLess(lat2, 40.0)

        # Cool down and warm up explicitly
        adapter.cool_down()
        self.assertFalse(adapter._is_warm)
        adapter.warm_up()
        self.assertTrue(adapter._is_warm)

    async def test_mock_tool_approval_gate(self):
        config = MockToolConfig(
            name="delete_account",
            requires_approval=True,
            is_side_effect=True,
        )
        adapter = MockToolAdapter(config)

        unapproved_call = ToolCall(tool_name="delete_account", is_approved=False)
        res_unapproved = await adapter.execute(unapproved_call)
        self.assertTrue(res_unapproved.is_error)
        self.assertIn("requires explicit approval", res_unapproved.error or "")

        approved_call = ToolCall(tool_name="delete_account", is_approved=True)
        res_approved = await adapter.execute(approved_call)
        self.assertFalse(res_approved.is_error)

    async def test_mock_tool_cancellation(self):
        config = MockToolConfig(name="slow_tool", median_ms=500.0)
        adapter = MockToolAdapter(config)
        call = ToolCall(tool_name="slow_tool")

        task = asyncio.create_task(adapter.execute(call))
        await asyncio.sleep(0.01)
        cancelled = await adapter.cancel(call.call_id)
        self.assertTrue(cancelled)

        res = await task
        self.assertTrue(res.is_error)
        self.assertIn("cancelled", (res.error or "").lower())

    async def test_mock_tool_engine_parallel(self):
        engine = MockToolEngine(seed=123)
        cfg1 = MockToolConfig(name="tool_a", median_ms=5.0, handler=lambda a: 1)
        cfg2 = MockToolConfig(name="tool_b", median_ms=5.0, handler=lambda a: 2)
        engine.register_tool(cfg1)
        engine.register_tool(cfg2)

        calls = [
            ToolCall(tool_name="tool_a"),
            ToolCall(tool_name="tool_b"),
        ]
        results = await engine.execute_parallel(calls)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].result, 1)
        self.assertEqual(results[1].result, 2)


class TestLiveTools(unittest.IsolatedAsyncioTestCase):
    """Test real local live tools (SQLite, Subprocess, File I/O, Mock HTTP)."""

    async def test_async_sqlite_tool(self):
        sqlite_tool = AsyncSQLiteTool(db_path=":memory:")

        # Create table
        call_create = ToolCall(
            tool_name="sqlite_executor",
            arguments={"query": "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL)"},
        )
        res_create = await sqlite_tool.execute(call_create)
        self.assertFalse(res_create.is_error)

        # Insert items
        call_insert = ToolCall(
            tool_name="sqlite_executor",
            arguments={"query": "INSERT INTO items (name, price) VALUES (?, ?)", "params": ["widget", 19.99]},
        )
        res_insert = await sqlite_tool.execute(call_insert)
        self.assertFalse(res_insert.is_error)

        # Select items
        call_select = ToolCall(
            tool_name="sqlite_executor",
            arguments={"query": "SELECT * FROM items WHERE name = ?", "params": ["widget"]},
        )
        res_select = await sqlite_tool.execute(call_select)
        self.assertFalse(res_select.is_error)
        rows = res_select.result
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "widget")
        self.assertAlmostEqual(rows[0]["price"], 19.99)
        sqlite_tool.close()

    async def test_safe_subprocess_sandbox(self):
        temp_dir = tempfile.mkdtemp(prefix="test_subproc_")
        sandbox = SafeSubprocessSandbox(sandbox_dir=temp_dir, default_timeout_s=5.0)

        # Safe python execution
        call = ToolCall(
            tool_name="subprocess_sandbox",
            arguments={"command": 'python3 -c "print(100 * 2)"'},
        )
        res = await sandbox.execute(call)
        self.assertFalse(res.is_error)
        self.assertEqual(res.result["stdout"].strip(), "200")
        self.assertEqual(res.result["exit_code"], 0)

        # Timeout handling
        call_timeout = ToolCall(
            tool_name="subprocess_sandbox",
            arguments={"command": "sleep 2", "timeout_s": 0.1},
        )
        res_timeout = await sandbox.execute(call_timeout)
        self.assertTrue(res_timeout.is_error)
        self.assertIn("timed out", (res_timeout.error or "").lower())

        shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_async_local_file_io(self):
        temp_dir = tempfile.mkdtemp(prefix="test_fileio_")
        file_io = AsyncLocalFileIOTool(base_dir=temp_dir)

        # Write file
        call_write = ToolCall(
            tool_name="file_io",
            arguments={"action": "write", "path": "docs/note.txt", "content": "ToolSpeed test"},
        )
        res_write = await file_io.execute(call_write)
        self.assertFalse(res_write.is_error)

        # Read file
        call_read = ToolCall(
            tool_name="file_io",
            arguments={"action": "read", "path": "docs/note.txt"},
        )
        res_read = await file_io.execute(call_read)
        self.assertFalse(res_read.is_error)
        self.assertEqual(res_read.result, "ToolSpeed test")

        # Append file
        call_append = ToolCall(
            tool_name="file_io",
            arguments={"action": "append", "path": "docs/note.txt", "content": " appended"},
        )
        await file_io.execute(call_append)
        res_read2 = await file_io.execute(call_read)
        self.assertEqual(res_read2.result, "ToolSpeed test appended")

        # Path traversal security check
        call_traversal = ToolCall(
            tool_name="file_io",
            arguments={"action": "read", "path": "../../etc/passwd"},
        )
        res_traversal = await file_io.execute(call_traversal)
        self.assertTrue(res_traversal.is_error)
        self.assertIn("traversal", (res_traversal.error or "").lower())

        shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_mock_http_server_and_client(self):
        server = MockHTTPServer()
        server.add_route("GET", "/api/status", {"status": "ok", "version": "1.0"})
        server.add_route("POST", "/api/echo", {"message": "received"}, status_code=201)
        base_url = server.start()

        try:
            client = AsyncHTTPClientTool(base_url=base_url)

            # GET request
            call_get = ToolCall(tool_name="http_client", arguments={"url": "/api/status", "method": "GET"})
            res_get = await client.execute(call_get)
            self.assertFalse(res_get.is_error)
            self.assertEqual(res_get.result["status"], "ok")

            # POST request
            call_post = ToolCall(
                tool_name="http_client",
                arguments={"url": "/api/echo", "method": "POST", "body": {"data": 123}},
            )
            res_post = await client.execute(call_post)
            self.assertFalse(res_post.is_error)
            self.assertEqual(res_post.result["message"], "received")
        finally:
            server.stop()


class TestMockModelsAndBytecode(unittest.IsolatedAsyncioTestCase):
    """Test SimulatedLLM, ActionBytecodeCodec, and DraftPredictorModel."""

    def test_action_bytecode_codec(self):
        call = ToolCall(
            tool_name="query_database",
            arguments={"table": "users", "limit": 100, "active": True},
        )
        encoded = ActionBytecodeCodec.encode(call)
        self.assertIsInstance(encoded, bytes)
        self.assertEqual(encoded[0], ActionBytecodeCodec.OP_TOOL_CALL)

        decoded = ActionBytecodeCodec.decode(encoded)
        self.assertEqual(decoded.tool_name, "query_database")
        self.assertEqual(decoded.arguments, {"table": "users", "limit": 100, "active": True})

    async def test_draft_predictor_model(self):
        predictor = DraftPredictorModel(latency_ms=1.0, accuracy=1.0, confidence_threshold=0.5, seed=42)
        res = await predictor.predict(
            prompt="Find user 123",
            candidate_tools=["search_db", "fetch_url"],
            ground_truth_tool="search_db",
            ground_truth_args={"user_id": 123},
        )
        self.assertIsNotNone(res)
        predicted_call, conf = res
        self.assertEqual(predicted_call.tool_name, "search_db")
        self.assertEqual(predicted_call.arguments, {"user_id": 123})
        self.assertTrue(predicted_call.is_speculative)

    async def test_simulated_llm_generation_and_streaming(self):
        profile = LatencyProfile(model_decision_ms=5.0, model_final_ms=5.0, draft_model_ms=2.0)
        llm = SimulatedLLM(profile=profile, tokens_per_second=500.0, seed=42)

        # Generate tool calls
        expected_call = ToolCall(tool_name="test_tool", arguments={"a": 1})
        text, calls, tokens = await llm.generate(
            prompt="Call test tool",
            expected_calls=[expected_call],
            is_final=False,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool_name, "test_tool")
        self.assertGreater(tokens.total_tokens, 0)
        self.assertGreater(tokens.cost_usd, 0.0)

        # Stream generation and commit horizon
        chunks = []
        commit_reached = False
        async for chunk in llm.stream_generate(
            prompt="Call test tool",
            expected_calls=[expected_call],
            is_final=False,
        ):
            chunks.append(chunk)
            if chunk.metadata.get("commit_horizon_reached"):
                commit_reached = True

        self.assertTrue(len(chunks) > 0)
        self.assertTrue(commit_reached)


if __name__ == "__main__":
    unittest.main()
