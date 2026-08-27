"""Real Local Wall-Clock Backend executing on local OS primitives."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import os
import tempfile
import time

from toolspeed.adapters.base import (
    BaseLLMAdapter,
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
from toolspeed.core.types import (
    EvidenceLevel,
    Task,
    ToolCall,
    ToolSpec,
)


class LocalWallClockBackend:
    """Real local wall-clock execution backend with HTTP, SQLite, File I/O, and Subprocess sandbox."""

    def __init__(self, evidence_level: EvidenceLevel = EvidenceLevel.LOCAL_WALL_CLOCK):
        self.evidence_level = evidence_level
        self._temp_dirs: List[str] = []
        self._servers: List[MockHTTPServer] = []

    def setup_environment(self) -> Tuple[ToolRegistry, MockHTTPServer, AsyncSQLiteTool, AsyncLocalFileIOTool, SafeSubprocessSandbox]:
        """Sets up all real local tools."""
        registry = ToolRegistry()

        # 1. Local HTTP Server & Client
        server = MockHTTPServer(host="127.0.0.1", port=0)
        server.add_route("GET", "/api/user/u42", {"user_id": "u42", "name": "Alice", "org_id": "org9"}, delay_s=0.01)
        server.add_route("GET", "/api/orders/u42", {"user_id": "u42", "orders": [101, 102]}, delay_s=0.01)
        server.add_route("POST", "/api/payment", {"status": "paid", "tx": "tx99"}, delay_s=0.015)
        server.start()
        self._servers.append(server)

        http_tool = AsyncHTTPClientTool(base_url=server.base_url, name="http_client")
        registry.register(http_tool)

        # 2. Local SQLite DB
        sqlite_tool = AsyncSQLiteTool(db_path=":memory:", name="sqlite_executor")
        # Initialize schema and seed data
        sqlite_tool._sync_execute(
            "CREATE TABLE users (user_id TEXT PRIMARY KEY, name TEXT, balance REAL)",
            [],
        )
        sqlite_tool._sync_execute(
            "INSERT INTO users VALUES ('u42', 'Alice', 150.0)",
            [],
        )
        registry.register(sqlite_tool)

        # 3. Local File I/O Sandbox
        tmp_dir = tempfile.mkdtemp(prefix="toolspeed_local_backend_")
        self._temp_dirs.append(tmp_dir)
        file_tool = AsyncLocalFileIOTool(base_dir=tmp_dir, name="file_io")
        file_tool._sync_op("write", "data.txt", "Local benchmark content payload")
        registry.register(file_tool)

        # 4. Safe Subprocess Sandbox
        sb_dir = tempfile.mkdtemp(prefix="toolspeed_sb_")
        self._temp_dirs.append(sb_dir)
        sb_tool = SafeSubprocessSandbox(sandbox_dir=sb_dir, name="subprocess_sandbox")
        registry.register(sb_tool)

        return registry, server, sqlite_tool, file_tool, sb_tool

    def cleanup(self) -> None:
        """Tears down servers and deletes temporary sandbox directories."""
        for s in self._servers:
            try:
                s.stop()
            except Exception:
                pass
        self._servers.clear()

        import shutil
        for d in self._temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs.clear()

    def __enter__(self) -> LocalWallClockBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
