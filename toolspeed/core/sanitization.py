"""External Egress Sanitization and Boundary Enforcement.

Strictly protects private credentials, secrets, tokens, authorization headers,
and benchmark oracle/evaluation data from leaking to external inference providers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Forbidden oracle and ground-truth identifiers that must never cross the egress boundary
FORBIDDEN_ORACLE_KEYS: frozenset[str] = frozenset(
    {
        "expected_output",
        "expected_outcome",
        "ground_truth",
        "oracle_canary",
        "validator",
        "oracle",
        "evaluation_label",
        "hidden_ground_truth",
    }
)

# Common credential and sensitive header regex patterns
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("BEARER_TOKEN", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{12,}\b")),
    ("JWT_TOKEN", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    (
        "API_KEY_EXPLICIT",
        re.compile(
            r"(?i)(?:api[_-]?key|secret[_-]?key|token|auth[_-]?token|passwd|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{8,})['\"]?"
        ),
    ),
    ("GENERIC_SECRET_PREFIX", re.compile(r"\b(?:sk|ts|key|akid|ghp|glpat|xoxb|xoxp|live)[_\-][A-Za-z0-9_\-]{12,}\b")),
    ("URL_WITH_CREDENTIALS", re.compile(r"https?://[^:\s/]+:[^@\s/]+@[^\s/]+")),
    ("URL_QUERY_TOKEN", re.compile(r"(?i)[?&](?:token|key|api_key|secret|password)=([A-Za-z0-9_\-\.]{8,})")),
    ("AUTH_HEADER", re.compile(r"(?i)authorization\s*:\s*[^\r\n,]+")),
]

MAX_PROMPT_CHARS: int = 2048


class EgressSecurityError(ValueError):
    """Raised when an unredacted secret or prohibited oracle leak is detected at the egress boundary."""


def sanitize_text(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Sanitizes text by redacting credentials and truncating to bounded size."""
    if not isinstance(text, str):
        text = str(text)

    sanitized = text

    # Redact secret patterns
    for name, pattern in _SECRET_PATTERNS:
        if name == "URL_QUERY_TOKEN":
            sanitized = pattern.sub(r"?\g<0>=[REDACTED]", sanitized)
        elif name == "API_KEY_EXPLICIT":
            sanitized = pattern.sub(r"key=[REDACTED]", sanitized)
        else:
            sanitized = pattern.sub(f"[REDACTED_{name}]", sanitized)

    # Redact literal mentions of known sensitive env vars if followed by values
    sanitized = re.sub(
        r"(TYPESAFE_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)\s*=\s*\S+",
        r"\1=[REDACTED]",
        sanitized,
    )

    if len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars] + "...[TRUNCATED]"

    return sanitized


def sanitize_value(value: Any, depth: int = 0) -> Any:
    """Recursively sanitizes values, redacting strings and pruning forbidden keys."""
    if depth > 8:
        return "[MAX_DEPTH_REACHED]"

    if isinstance(value, str):
        return sanitize_text(value)
    elif isinstance(value, (int, float, bool)) or value is None:
        return value
    elif isinstance(value, Mapping):
        clean_map: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k).lower()
            # Drop forbidden oracle and evaluation keys completely
            if key_str in FORBIDDEN_ORACLE_KEYS:
                continue
            # Drop explicit credential keys
            if any(s in key_str for s in ("password", "secret", "token", "auth_header", "private_key", "api_key")):
                clean_map[str(k)] = "[REDACTED_CREDENTIAL]"
            else:
                clean_map[str(k)] = sanitize_value(v, depth + 1)
        return clean_map
    elif isinstance(value, (list, tuple)):
        return [sanitize_value(elem, depth + 1) for elem in value]
    else:
        return sanitize_text(str(value))


def derive_argument_shapes(arguments: Mapping[str, Any] | None) -> dict[str, str]:
    """Extracts structural type metadata rather than raw argument values.

    Example:
        {"query": "SELECT * FROM users", "limit": 10} -> {"query": "str", "limit": "int"}
    """
    if not arguments:
        return {}
    shapes: dict[str, str] = {}
    for k, v in arguments.items():
        k_clean = str(k)
        if str(k).lower() in FORBIDDEN_ORACLE_KEYS:
            continue
        shapes[k_clean] = type(v).__name__
    return shapes


def assert_no_egress_violations(data: Any) -> None:
    """Verifies that data contains zero unredacted secrets or prohibited oracle keys.

    Raises EgressSecurityError if any violations are discovered.
    """

    def _inspect(node: Any, path: str = "") -> None:
        if isinstance(node, str):
            for name, pattern in _SECRET_PATTERNS:
                if pattern.search(node):
                    raise EgressSecurityError(f"Egress boundary violation at '{path}': detected unredacted {name}")
        elif isinstance(node, Mapping):
            for k, v in node.items():
                k_lower = str(k).lower()
                if k_lower in FORBIDDEN_ORACLE_KEYS:
                    raise EgressSecurityError(
                        f"Egress boundary violation: prohibited oracle key '{k}' found at '{path}'"
                    )
                _inspect(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for i, elem in enumerate(node):
                _inspect(elem, f"{path}[{i}]")

    _inspect(data)
