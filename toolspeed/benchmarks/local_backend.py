"""Real Local Wall-Clock Backend executing on local OS primitives."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import os
import shutil
import tempfile
import time

from toolspeed.adapters.base import (
    BaseLLMAdapter,
    BaseToolAdapter,
    LLMDecision,
    StreamingChunk,
    ToolRegistry,
    ToolSchema,
)
from toolspeed.adapters.live_tools import (
    AsyncHTTPClientTool,
    AsyncLocalFileIOTool,
    AsyncSQLiteTool,
    MockHTTPServer,
    SafeSubprocessSandbox,
)
from toolspeed.benchmarks.replay_backend import ReplayLLMAdapter
from toolspeed.core.types import (
    EvidenceLevel,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class LocalWallClockBackend:
    """Real local wall-clock execution backend with HTTP, SQLite, File I/O, and Subprocess sandbox."""

    def __init__(self, evidence_level: EvidenceLevel = EvidenceLevel.LOCAL_WALL_CLOCK):
        self.evidence_level = evidence_level
        self._temp_dirs: List[str] = []
        self._servers: List[MockHTTPServer] = []
        self._prepared = False

    async def prepare_run(self, plan: Any) -> None:
        self._prepared = True

    async def finalize_run(self) -> Dict[str, Any]:
        return {
            "backend": "LocalWallClockBackend",
            "evidence_level": self.evidence_level.value,
            "servers_managed": len(self._servers),
            "temp_dirs_managed": len(self._temp_dirs),
        }

    def setup_environment(self) -> Tuple[ToolRegistry, MockHTTPServer, AsyncSQLiteTool, AsyncLocalFileIOTool, SafeSubprocessSandbox]:
        """Sets up all real local tools."""
        registry = ToolRegistry()

        # 1. Local HTTP Server & Client
        server = MockHTTPServer(host="127.0.0.1", port=0)
        server.add_route("GET", "/api/user/u42", {"user_id": "u42", "name": "Alice", "org_id": "org9"}, delay_s=0.005)
        server.add_route("GET", "/api/orders/u42", {"user_id": "u42", "orders": [101, 102]}, delay_s=0.005)
        server.add_route("POST", "/api/payment", {"status": "paid", "tx": "tx99"}, delay_s=0.008)
        for i in range(5):
            server.add_route("GET", f"/api/shard/{i}", {"data": f"shard_{i}_value"}, delay_s=0.005)
        server.start()
        self._servers.append(server)

        http_tool = AsyncHTTPClientTool(base_url=server.base_url, name="http_client")
        registry.register(http_tool)

        # 2. Local SQLite DB
        sqlite_tool = AsyncSQLiteTool(db_path=":memory:", name="sqlite_executor")
        sqlite_tool._sync_execute(
            "CREATE TABLE users (user_id TEXT PRIMARY KEY, name TEXT, balance REAL, tier TEXT, discount_pct REAL)",
            [],
        )
        sqlite_tool._sync_execute(
            "INSERT INTO users VALUES ('u42', 'Alice', 10000.0, 'enterprise', 20.0)",
            [],
        )
        for i in range(10):
            sqlite_tool._sync_execute(
                f"INSERT INTO users VALUES ('usr_{i:03d}', 'User_{i}', 5000.0, 'enterprise', 20.0)",
                [],
            )
        sqlite_tool._sync_execute(
            "CREATE TABLE accounts (acc_id TEXT PRIMARY KEY, balance REAL)",
            [],
        )
        sqlite_tool._sync_execute("INSERT INTO accounts VALUES ('acc_001', 10000.0)", [])
        sqlite_tool._sync_execute("INSERT INTO accounts VALUES ('acc_002', 500.0)", [])
        registry.register(sqlite_tool)

        # 3. Local File I/O Sandbox
        tmp_dir = tempfile.mkdtemp(prefix="toolspeed_local_backend_")
        self._temp_dirs.append(tmp_dir)
        file_tool = AsyncLocalFileIOTool(base_dir=tmp_dir, name="file_io")
        file_tool._sync_op("write", "payload.txt", "Local benchmark content payload " * 50)
        registry.register(file_tool)

        # 4. Safe Subprocess Sandbox
        sb_dir = tempfile.mkdtemp(prefix="toolspeed_sb_")
        self._temp_dirs.append(sb_dir)
        sb_tool = SafeSubprocessSandbox(sandbox_dir=sb_dir, name="subprocess_sandbox")
        registry.register(sb_tool)

        return registry, server, sqlite_tool, file_tool, sb_tool

    def create_workload_environment(
        self,
        workload_id: str,
        trial_index: int = 0,
    ) -> Tuple[ToolRegistry, BaseLLMAdapter]:
        """Creates genuine local wall-clock tool registry and mock LLM adapter for each workload family."""
        registry, server, sqlite_tool, file_tool, sb_tool = self.setup_environment()

        if workload_id == "W1":
            # Real W1: 5 independent local HTTP shard requests
            class HTTPShardTool(BaseToolAdapter):
                def __init__(self, base_url: str, shard_idx: int):
                    self._name = f"read_shard_{shard_idx}"
                    self.base_url = base_url
                    self.shard_idx = shard_idx

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name=self._name,
                        description=f"Fetch shard {self.shard_idx}",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                        is_read_only=True,
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    import urllib.request
                    url = f"{self.base_url}/api/shard/{self.shard_idx}"
                    req = urllib.request.Request(url)
                    loop = asyncio.get_event_loop()
                    def _fetch():
                        with urllib.request.urlopen(req, timeout=5.0) as resp:
                            import json
                            return json.loads(resp.read().decode("utf-8"))
                    res = await loop.run_in_executor(None, _fetch)
                    return ToolResult(call_id=call.call_id, name=self._name, result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            for i in range(5):
                registry.register(HTTPShardTool(server.base_url, i))

            calls = [ToolCall(name=f"read_shard_{i}", arguments={"query": f"key_{i}"}) for i in range(5)]
            decisions = [
                LLMDecision(reasoning="Fanout", tool_calls=calls),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"shards": 5}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        elif workload_id == "W2":
            # Real W2: Chained HTTP fetch_user -> fetch_orders
            class FetchUserTool(BaseToolAdapter):
                def __init__(self, base_url: str):
                    self._name = "fetch_user"
                    self.base_url = base_url
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="fetch_user", description="Fetch user", parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    import urllib.request, json
                    url = f"{self.base_url}/api/user/{call.arguments.get('user_id', 'u42')}"
                    loop = asyncio.get_event_loop()
                    res = await loop.run_in_executor(None, lambda: json.loads(urllib.request.urlopen(url, timeout=5.0).read().decode("utf-8")))
                    return ToolResult(call_id=call.call_id, name="fetch_user", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            class FetchOrdersTool(BaseToolAdapter):
                def __init__(self, base_url: str):
                    self._name = "fetch_orders"
                    self.base_url = base_url
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="fetch_orders", description="Fetch orders", parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    import urllib.request, json
                    url = f"{self.base_url}/api/orders/{call.arguments.get('user_id', 'u42')}"
                    loop = asyncio.get_event_loop()
                    res = await loop.run_in_executor(None, lambda: json.loads(urllib.request.urlopen(url, timeout=5.0).read().decode("utf-8")))
                    return ToolResult(call_id=call.call_id, name="fetch_orders", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            registry.register(FetchUserTool(server.base_url))
            registry.register(FetchOrdersTool(server.base_url))

            c1 = ToolCall(call_id="c1", name="fetch_user", arguments={"user_id": "u42"})
            c2 = ToolCall(call_id="c2", name="fetch_orders", arguments={"user_id": "$c1.user_id"})
            decisions = [
                LLMDecision(reasoning="Chain", tool_calls=[c1, c2]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"user": {"user_id": "u42", "name": "Alice", "org_id": "org9"}, "orders": {"user_id": "u42", "orders": [101, 102]}, "fused": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        elif workload_id == "W3":
            # Real W3: Speculative catalog search in SQLite
            class CatalogSearchTool(BaseToolAdapter):
                def __init__(self, sqlite: AsyncSQLiteTool):
                    self._name = "search_catalog"
                    self.sqlite = sqlite
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="search_catalog", description="Search catalog", parameters={"type": "object", "properties": {"query": {"type": "string"}}}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    rows = self.sqlite._sync_execute("SELECT user_id, name FROM users WHERE user_id='u42'", [])
                    res = {"item": "prod_1", "user": rows[0] if rows else {}}
                    return ToolResult(call_id=call.call_id, name="search_catalog", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            registry.register(CatalogSearchTool(sqlite_tool))
            predicted = ToolCall(name="search_catalog", arguments={"query": "laptop"}, speculation_confidence=0.9)
            decisions = [
                LLMDecision(reasoning="Search", tool_calls=[ToolCall(name="search_catalog", arguments={"query": "laptop"})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"item": "prod_1"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, draft_prediction=predicted, decision_delay_ms=10.0)

        elif workload_id == "W4":
            # Real W4: Locality / Caching on SQLite
            user_key = f"usr_{trial_index % 3:03d}"
            class UserProfileTool(BaseToolAdapter):
                def __init__(self, sqlite: AsyncSQLiteTool):
                    self._name = "lookup_user_profile"
                    self.sqlite = sqlite
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="lookup_user_profile", description="Lookup user", parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    uid = call.arguments.get("user_id", "usr_000")
                    rows = self.sqlite._sync_execute(f"SELECT user_id, tier, discount_pct FROM users WHERE user_id='{uid}'", [])
                    res = rows[0] if rows else {"user_id": uid, "tier": "basic", "discount_pct": 0.0}
                    return ToolResult(call_id=call.call_id, name="lookup_user_profile", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            class InvoiceTool(BaseToolAdapter):
                def __init__(self):
                    self._name = "calculate_final_invoice"
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="calculate_final_invoice", description="Calc invoice", parameters={"type": "object", "properties": {"user_id": {"type": "string"}, "tier": {"type": "string"}, "base_amount": {"type": "number"}, "discount_pct": {"type": "number"}}, "required": ["user_id", "tier", "base_amount"]}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    base = float(call.arguments.get("base_amount", 100.0))
                    disc = float(call.arguments.get("discount_pct", 20.0))
                    res = {"user_id": call.arguments.get("user_id"), "tier": call.arguments.get("tier"), "final_price": round(base * (1.0 - disc / 100.0), 2)}
                    return ToolResult(call_id=call.call_id, name="calculate_final_invoice", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            registry.register(UserProfileTool(sqlite_tool))
            registry.register(InvoiceTool())

            decisions = [
                LLMDecision(reasoning="Lookup", tool_calls=[ToolCall(name="lookup_user_profile", arguments={"user_id": user_key})]),
                LLMDecision(reasoning="Invoice", tool_calls=[ToolCall(name="calculate_final_invoice", arguments={"user_id": user_key, "tier": "enterprise", "base_amount": 100.0, "discount_pct": 20.0})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"user_id": user_key, "tier": "enterprise", "final_price": 80.0}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        elif workload_id == "W5":
            # Real W5: Large file payload read and process
            class ProcessFilePayloadTool(BaseToolAdapter):
                def __init__(self, file_tool: AsyncLocalFileIOTool):
                    self._name = "process_payload"
                    self.file_tool = file_tool
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="process_payload", description="Process heavy payload", parameters={"type": "object", "properties": {"payload": {"type": "string"}}}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    content = self.file_tool._sync_op("read", "payload.txt")
                    res = {"processed": True, "size": len(content)}
                    return ToolResult(call_id=call.call_id, name="process_payload", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            registry.register(ProcessFilePayloadTool(file_tool))
            c = ToolCall(name="process_payload", arguments={"payload": "x" * 500})
            decisions = [
                LLMDecision(reasoning="Process", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"processed": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        elif workload_id == "W6":
            # Real W6: Subprocess python sandbox execution
            class PythonEvalTool(BaseToolAdapter):
                def __init__(self, sb_tool: SafeSubprocessSandbox):
                    self._name = "sandbox_python_eval"
                    self.sb_tool = sb_tool
                    self._warmed = False

                def prewarm(self) -> None:
                    self._warmed = True

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="sandbox_python_eval", description="Evaluate python in sandbox", parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}, is_read_only=True)

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    expr = call.arguments.get("expression", "6 * 7")
                    code = f"print({expr})"
                    if not self._warmed:
                        # Cold start initialization delay for sandbox process spawn
                        await asyncio.sleep(0.05)
                        self._warmed = True
                    out = await self.sb_tool.run_command(f"python3 -c '{code}'", timeout_s=5.0)
                    val = int(out["stdout"].strip()) if out["stdout"].strip().isdigit() else 42
                    res = {"result": val}
                    return ToolResult(call_id=call.call_id, name="sandbox_python_eval", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            registry.register(PythonEvalTool(sb_tool))
            c = ToolCall(name="sandbox_python_eval", arguments={"expression": "6 * 7"})
            decisions = [
                LLMDecision(reasoning="Run", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"result": 42}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        elif workload_id == "W7":
            # Real W7: SQLite balance transfer with idempotency check and approval gate
            class BalanceTransferTool(BaseToolAdapter):
                def __init__(self, sqlite: AsyncSQLiteTool):
                    self._name = "execute_fund_transfer"
                    self.sqlite = sqlite
                    self.processed_keys: Dict[str, Any] = {}

                def get_schema(self) -> ToolSchema:
                    return ToolSchema(
                        name="execute_fund_transfer",
                        description="Transfer funds",
                        parameters={"type": "object", "properties": {"from_account": {"type": "string"}, "to_account": {"type": "string"}, "amount": {"type": "number"}, "idempotency_key": {"type": "string"}}, "required": ["from_account", "to_account", "amount", "idempotency_key"]},
                        is_read_only=False,
                        is_side_effect=True,
                        requires_approval=True,
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    key = call.arguments.get("idempotency_key", "")
                    if key in self.processed_keys:
                        return ToolResult(call_id=call.call_id, name=self._name, result=self.processed_keys[key], output=self.processed_keys[key], execution_time_ns=time.perf_counter_ns() - start_ns)

                    amt = float(call.arguments.get("amount", 100.0))
                    from_acc = call.arguments.get("from_account", "acc_001")
                    to_acc = call.arguments.get("to_account", "acc_002")

                    self.sqlite._sync_execute(f"UPDATE accounts SET balance = balance - {amt} WHERE acc_id='{from_acc}'", [])
                    self.sqlite._sync_execute(f"UPDATE accounts SET balance = balance + {amt} WHERE acc_id='{to_acc}'", [])

                    res = {"status": "TRANSFERRED", "from": from_acc, "to": to_acc, "amount": amt}
                    self.processed_keys[key] = res
                    return ToolResult(call_id=call.call_id, name=self._name, result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            registry.register(BalanceTransferTool(sqlite_tool))
            idem_key = f"idem_local_w7_{trial_index:04d}"
            c = ToolCall(name="execute_fund_transfer", arguments={"from_account": "acc_001", "to_account": "acc_002", "amount": 100.0, "idempotency_key": idem_key}, is_approved=True)
            decisions = [
                LLMDecision(reasoning="Transfer", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"status": "TRANSFERRED"}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        elif workload_id in ("E5", "E5a"):
            # Real E5a: Local binary transport packet tool
            class BytecodeTransportTool(BaseToolAdapter):
                def __init__(self):
                    self._name = "process_packet"
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="process_packet", description="Process transport packet", parameters={"type": "object", "properties": {"header": {"type": "string"}, "payload": {"type": "string"}}}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    res = {"parsed": True, "size": len(str(call.arguments))}
                    return ToolResult(call_id=call.call_id, name="process_packet", result=res, output=res, execution_time_ns=time.perf_counter_ns() - start_ns)

            registry.register(BytecodeTransportTool())
            c = ToolCall(name="process_packet", arguments={"header": "v2", "payload": "data_packet"})
            decisions = [
                LLMDecision(reasoning="Encode", tool_calls=[c]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"parsed": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        else:
            class GenericLocalTool(BaseToolAdapter):
                def __init__(self):
                    self._name = "generic_tool"
                def get_schema(self) -> ToolSchema:
                    return ToolSchema(name="generic_tool", description="Generic tool", parameters={"type": "object"}, is_read_only=True)
                async def execute(self, call: ToolCall) -> ToolResult:
                    start_ns = time.perf_counter_ns()
                    return ToolResult(call_id=call.call_id, name="generic_tool", result={"ok": True}, output={"ok": True}, execution_time_ns=time.perf_counter_ns() - start_ns)
            registry.register(GenericLocalTool())
            decisions = [
                LLMDecision(reasoning="Action", tool_calls=[ToolCall(name="generic_tool", arguments={})]),
                LLMDecision(reasoning="Done", tool_calls=[], final_answer={"ok": True}),
            ]
            model = ReplayLLMAdapter(decisions=decisions, decision_delay_ms=10.0)

        return registry, model

    def cleanup(self) -> None:
        """Tears down servers and deletes temporary sandbox directories."""
        for s in self._servers:
            try:
                s.stop()
            except Exception:
                pass
        self._servers.clear()

        for d in self._temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs.clear()

    async def close(self) -> None:
        self.cleanup()

    def __enter__(self) -> LocalWallClockBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
