"""Tests for Phase 32: Fail-closed process-tree termination, SIGTERM/SIGKILL escalation, and zero orphan guarantee."""

from __future__ import annotations

import asyncio
import signal
import unittest

from toolspeed.adapters.live_tools import SafeSubprocessSandbox
from toolspeed.core.types import ToolCall


class TestSubprocessSecurity(unittest.IsolatedAsyncioTestCase):
    """Verifies fail-closed process group cleanup, SIGTERM/SIGKILL escalation, and zero orphans."""

    async def test_01_timeout_kills_process_group_no_orphans(self) -> None:
        """Process group spawning child jobs is killed on timeout without orphans."""
        sandbox = SafeSubprocessSandbox(default_timeout_s=0.2)
        call = ToolCall("c_timeout", "subprocess_sandbox", {"command": "sh -c 'sleep 30 & wait'", "timeout_s": 0.2})

        res = await sandbox.execute(call)
        self.assertTrue(res.is_error)
        self.assertIn("timed out", str(res.error))
        self.assertTrue(sandbox.is_process_tree_terminated(call.call_id))

    async def test_02_cancellation_kills_process_group(self) -> None:
        """Cancelled coroutine terminates entire process tree before returning."""
        sandbox = SafeSubprocessSandbox(default_timeout_s=10.0)
        call = ToolCall("c_cancel", "subprocess_sandbox", {"command": "sleep 30"})

        task = asyncio.create_task(sandbox.execute(call))
        await asyncio.sleep(0.05)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(sandbox.is_process_tree_terminated(call.call_id))

    async def test_03_sigterm_grace_period_escalation_to_sigkill(self) -> None:
        """A process that traps and ignores SIGTERM is forcibly terminated by SIGKILL after grace period."""
        sandbox = SafeSubprocessSandbox(default_timeout_s=10.0)

        # Spawn a process that ignores SIGTERM
        proc = await asyncio.create_subprocess_shell(
            "trap '' TERM; sleep 30",
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await asyncio.sleep(0.05)
        self.assertIsNone(proc.returncode)

        # Call terminate with short grace period
        await sandbox._terminate_process_group(proc, grace_period_s=0.2)

        self.assertIsNotNone(proc.returncode)
        # On Unix, SIGKILL exit code is -9 (or 137 / -signal.SIGKILL)
        self.assertIn(proc.returncode, (-signal.SIGKILL, -9, 137))

    async def test_04_cooperative_process_exits_on_sigterm(self) -> None:
        """A normal process terminates cleanly on SIGTERM within the grace period."""
        sandbox = SafeSubprocessSandbox(default_timeout_s=10.0)

        proc = await asyncio.create_subprocess_shell(
            "sleep 30",
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await asyncio.sleep(0.05)
        self.assertIsNone(proc.returncode)

        await sandbox._terminate_process_group(proc, grace_period_s=0.5)

        self.assertIsNotNone(proc.returncode)
        # Exited on SIGTERM
        self.assertIn(proc.returncode, (-signal.SIGTERM, -15, 143))


if __name__ == "__main__":
    unittest.main()
