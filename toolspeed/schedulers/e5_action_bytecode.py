"""Experiment E5: Action Bytecode Transport Codec & Execution Scheduler."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import json
import struct
import time

from toolspeed.adapters.base import BaseLLMAdapter, ToolRegistry
from toolspeed.core.types import EventType, ToolCall, ToolResult, ToolSpec
from toolspeed.schedulers.base import BaseScheduler, ExecutionContext


class ActionBytecodeCodec:
    """E5a: Compact binary transport codec for tool calls.
    
    Format:
      [Version (1B)] [Opcode (2B >H)] [ArgCount (2B >H)]
      Repeated per argument:
        [KeyLen (2B >H)] [ValLen (4B >I)] [KeyBytes] [ValBytes (JSON)]
    """
    PROTOCOL_VERSION = 0x02

    def __init__(self, tool_specs: Optional[List[ToolSpec]] = None) -> None:
        self.tool_to_opcode: Dict[str, int] = {}
        self.opcode_to_tool: Dict[int, str] = {}
        self.tool_arg_order: Dict[str, List[str]] = {}

        if tool_specs:
            for idx, spec in enumerate(tool_specs, start=1):
                self.register_tool(spec.name, list(spec.parameters.get("properties", {}).keys()) or spec.required_args, opcode=idx)

    def register_tool(self, name: str, arg_order: Optional[List[str]] = None, opcode: Optional[int] = None) -> int:
        if name in self.tool_to_opcode:
            return self.tool_to_opcode[name]

        op = opcode if opcode is not None else (len(self.tool_to_opcode) + 1)
        if op in self.opcode_to_tool and self.opcode_to_tool[op] != name:
            raise ValueError(f"Opcode collision: opcode {op} is already registered to tool '{self.opcode_to_tool[op]}'")
        if op > 65535 or op <= 0:
            raise ValueError(f"Opcode out of range: must be 1..65535, got {op}")

        self.tool_to_opcode[name] = op
        self.opcode_to_tool[op] = name
        self.tool_arg_order[name] = list(arg_order or [])
        return op

    def encode(self, call: ToolCall) -> bytes:
        """Encodes a ToolCall into a compact binary packet with 16-bit opcode support."""
        tool_name = call.name or call.tool_name or "default_tool"
        op = self.tool_to_opcode.get(tool_name)
        if op is None:
            op = self.register_tool(tool_name, list(call.arguments.keys()))

        payload_parts = []
        for k, v in call.arguments.items():
            k_bytes = k.encode("utf-8")
            v_bytes = json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            if len(k_bytes) > 65535:
                raise ValueError(f"Argument key '{k}' exceeds maximum length of 65535 bytes")
            if len(v_bytes) > 100_000_000:
                raise ValueError(f"Argument value for '{k}' exceeds maximum payload limit")
            payload_parts.append(
                struct.pack(">HI", len(k_bytes), len(v_bytes)) + k_bytes + v_bytes
            )

        body = b"".join(payload_parts)
        # Header: Version (1B), Opcode (2B >H), ArgCount (2B >H) = 5 bytes
        header = struct.pack(">BHH", self.PROTOCOL_VERSION, op, len(call.arguments))
        return header + body

    def decode(self, data: bytes) -> ToolCall:
        """Decodes binary bytecode back into a structured ToolCall, validating length and bounds."""
        if len(data) < 5:
            raise ValueError(f"Bytecode packet too short: expected at least 5 bytes header, got {len(data)}")

        version, op, arg_count = struct.unpack(">BHH", data[:5])
        if version != self.PROTOCOL_VERSION:
            raise ValueError(f"Unsupported bytecode protocol version: {version}")
        if op == 0 or op not in self.opcode_to_tool:
            raise ValueError(f"Unknown or invalid opcode: {op}")

        tool_name = self.opcode_to_tool[op]
        offset = 5
        args: Dict[str, Any] = {}

        for i in range(arg_count):
            if offset + 6 > len(data):
                raise ValueError(f"Bytecode packet truncated: missing header for argument {i}")
            k_len, val_len = struct.unpack(">HI", data[offset : offset + 6])
            offset += 6
            if offset + k_len + val_len > len(data):
                raise ValueError(f"Bytecode packet truncated: expected {k_len + val_len} bytes for argument {i}, got {len(data) - offset}")

            key = data[offset : offset + k_len].decode("utf-8")
            offset += k_len
            val_bytes = data[offset : offset + val_len]
            offset += val_len

            try:
                parsed_val = json.loads(val_bytes.decode("utf-8"))
            except Exception as ex:
                raise ValueError(f"Malformed JSON argument value for key '{key}': {ex}")

            args[key] = parsed_val

        if offset != len(data):
            raise ValueError(f"Bytecode packet contains {len(data) - offset} unexpected trailing bytes")

        return ToolCall(name=tool_name, tool_name=tool_name, arguments=args)

    def calculate_compression_ratio(self, call: ToolCall) -> Tuple[int, int, float]:
        """Compares JSON character count to compact binary bytecode size."""
        json_len = len(json.dumps({"name": call.name or call.tool_name, "arguments": call.arguments}, separators=(",", ":")))
        bc_len = len(self.encode(call))
        ratio = json_len / max(1, bc_len)
        return json_len, bc_len, ratio


class ActionBytecodeScheduler(BaseScheduler):
    """Experiment E5: Action Bytecode Engine.
    
    Evaluates binary transport codec compression and decoding throughput.
    Note: Direct model action-token generation is scoped as E5b and marked unimplemented for live models.
    """

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.codec = ActionBytecodeCodec()

    async def _execute_internal(
        self,
        ctx: ExecutionContext,
        model: BaseLLMAdapter,
        tools: ToolRegistry,
    ) -> Any:
        for spec in tools.list_specs():
            self.codec.register_tool(spec.name, spec.required_args or list(spec.parameters.get("properties", {}).keys()))

        for turn in range(ctx.config.max_turns):
            ctx.step_count = turn + 1

            ctx.profiler.start_span(f"bytecode_model_turn_{turn}")
            decision = await model.decide(
                ctx.task,
                ctx.history,
                tools.list_specs(),
            )
            ctx.profiler.end_span(
                f"bytecode_model_turn_{turn}",
                EventType.MODEL_END,
                details={"turn": turn, "calls": len(decision.tool_calls)},
            )
            ctx.record_model_decision(decision)

            if decision.final_answer is not None or not decision.tool_calls:
                return decision.final_answer

            for raw_call in decision.tool_calls:
                # 1. Transport encode
                t_enc_start = time.perf_counter()
                encoded_bytes = self.codec.encode(raw_call)
                t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0
                json_size, bc_size, comp_ratio = self.codec.calculate_compression_ratio(raw_call)

                ctx.profiler.record_event(
                    EventType.BYTECODE_ENCODE,
                    duration_ms=t_enc_ms,
                    details={
                        "tool": raw_call.name,
                        "json_bytes": json_size,
                        "bytecode_bytes": bc_size,
                        "compression_ratio": comp_ratio,
                    },
                )

                # 2. Transport decode
                t_dec_start = time.perf_counter()
                decoded_call = self.codec.decode(encoded_bytes)
                t_dec_ms = (time.perf_counter() - t_dec_start) * 1000.0
                decoded_call.call_id = raw_call.call_id
                ctx.tool_calls.append(decoded_call)

                ctx.profiler.record_event(
                    EventType.BYTECODE_DECODE,
                    duration_ms=t_dec_ms,
                    details={"tool": decoded_call.name, "call_id": decoded_call.call_id},
                )

                # 3. Tool execution via ToolExecutor
                res = await ctx.executor.execute(decoded_call)
                ctx.record_tool_result(res)

        return "Max turns reached without final answer."
