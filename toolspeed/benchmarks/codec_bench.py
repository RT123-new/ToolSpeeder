"""Codec comparison benchmarks under symmetric serialization policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodecConfig:
    name: str
    float_precision_policy: str = "ieee754_double"
    key_sort_policy: str = "canonical_lexicographic"


def get_json_codec() -> CodecConfig:
    return CodecConfig(name="json")


def get_bytecode_codec() -> CodecConfig:
    return CodecConfig(name="action_bytecode")
