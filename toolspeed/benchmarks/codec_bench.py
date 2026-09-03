"""Codec comparison benchmarks under symmetric serialization policies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from toolspeed.core.types import ToolCall
from toolspeed.schedulers.e5_action_bytecode import ActionBytecodeCodec


@dataclass
class CodecConfig:
    name: str
    float_precision_policy: str = "ieee754_double"
    key_sort_policy: str = "canonical_lexicographic"


class CanonicalJSONCodec:
    """Canonical JSON codec with strict key ordering and IEEE 754 serialization."""

    def __init__(self, config: CodecConfig | None = None) -> None:
        self.config = config or get_json_codec()

    def encode(self, call: ToolCall, schema_hash: str = "") -> bytes:
        data: dict[str, Any] = {
            "call_id": call.call_id,
            "tool_name": call.name or call.tool_name,
            "arguments": dict(call.arguments),
            "schema_hash": schema_hash,
        }
        return json.dumps(data, sort_keys=True, allow_nan=False).encode("utf-8")

    def decode(self, payload: bytes) -> ToolCall:
        data = json.loads(payload.decode("utf-8"))
        return ToolCall(
            call_id=data["call_id"],
            tool_name=data["tool_name"],
            name=data["tool_name"],
            arguments=data["arguments"],
        )


def get_json_codec() -> CodecConfig:
    return CodecConfig(name="json")


def get_bytecode_codec() -> CodecConfig:
    return CodecConfig(name="action_bytecode")


@dataclass
class CodecBenchmarkResult:
    json_bytes: int
    bytecode_bytes: int
    compression_ratio: float
    json_encode_ns: int
    json_decode_ns: int
    bytecode_encode_ns: int
    bytecode_decode_ns: int
    round_trip_equal: bool


def benchmark_codecs_symmetric(call: ToolCall, schema_hash: str = "s_123") -> CodecBenchmarkResult:
    """Benchmarks JSONCodec vs ActionBytecodeCodec under symmetric policies."""
    json_codec = CanonicalJSONCodec()
    byte_codec = ActionBytecodeCodec()

    # JSON encode/decode
    t0 = time.perf_counter_ns()
    json_bytes = json_codec.encode(call, schema_hash=schema_hash)
    t1 = time.perf_counter_ns()
    decoded_json = json_codec.decode(json_bytes)
    t2 = time.perf_counter_ns()

    # Bytecode encode/decode
    t3 = time.perf_counter_ns()
    bytecode_bytes = byte_codec.encode(call, schema_hash=schema_hash)
    t4 = time.perf_counter_ns()
    decoded_byte = byte_codec.decode(bytecode_bytes)
    t5 = time.perf_counter_ns()

    equal = decoded_json.arguments == decoded_byte.arguments and (decoded_json.name or decoded_json.tool_name) == (
        decoded_byte.name or decoded_byte.tool_name
    )

    ratio = len(bytecode_bytes) / len(json_bytes) if len(json_bytes) > 0 else 1.0

    return CodecBenchmarkResult(
        json_bytes=len(json_bytes),
        bytecode_bytes=len(bytecode_bytes),
        compression_ratio=ratio,
        json_encode_ns=t1 - t0,
        json_decode_ns=t2 - t1,
        bytecode_encode_ns=t4 - t3,
        bytecode_decode_ns=t5 - t4,
        round_trip_equal=equal,
    )
