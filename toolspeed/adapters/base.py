"""Base classes and interfaces for Tool and LLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union
import json

from toolspeed.core.types import TokenUsage, ToolCall, ToolResult, ToolSpec, Task


@dataclass
class StreamingChunk:
    """A streaming token chunk emitted by an LLM adapter."""
    text: str = ""
    delta_text: str = ""
    tool_call_delta: Optional[dict[str, Any]] = None
    is_final: bool = False
    token_index: int = 0
    parsed_tool_calls: List[ToolCall] = field(default_factory=list)
    commit_horizon_ready: List[ToolCall] = field(default_factory=list)
    raw_json_fragment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.delta_text and self.text:
            self.delta_text = self.text
        elif not self.text and self.delta_text:
            self.text = self.delta_text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolSchema:
    """JSON Schema definition and execution constraints for a tool."""
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    is_side_effect: bool = False
    is_read_only: bool = True
    requires_approval: bool = False
    cost_usd: float = 0.0
    cache_ttl_s: Optional[float] = None
    required_args: List[str] = field(default_factory=list)
    commit_horizon_args: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.is_read_only:
            self.is_side_effect = True
        elif self.is_side_effect:
            self.is_read_only = False

    @property
    def side_effects(self) -> bool:
        return self.is_side_effect

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolSchema:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=dict(data.get("parameters", {})),
            is_side_effect=bool(data.get("is_side_effect", False)),
            requires_approval=bool(data.get("requires_approval", False)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            cache_ttl_s=data.get("cache_ttl_s"),
            required_args=list(data.get("required_args", [])),
            commit_horizon_args=list(data.get("commit_horizon_args", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class LLMDecision:
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[Any] = None
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    raw_chunks: List[StreamingChunk] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None or len(self.tool_calls) == 0


class BaseToolAdapter(ABC):
    """Abstract interface for tool executors."""

    def __init__(self, spec: Optional[Union[ToolSpec, ToolSchema]] = None) -> None:
        if spec is not None:
            if isinstance(spec, ToolSpec):
                self._spec = spec
                self._schema = ToolSchema(
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.parameters,
                    is_side_effect=spec.side_effects or not spec.is_read_only,
                    requires_approval=not spec.is_read_only,
                    required_args=spec.required_args,
                    commit_horizon_args=spec.commit_horizon_args,
                )
            else:
                self._schema = spec
                self._spec = ToolSpec(
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.parameters,
                    required_args=spec.required_args,
                    commit_horizon_args=spec.commit_horizon_args,
                    is_read_only=not spec.is_side_effect,
                    side_effects=spec.is_side_effect,
                )
        else:
            self._spec = None
            self._schema = None

    @property
    def name(self) -> str:
        if hasattr(self, "config") and hasattr(self.config, "name"):
            return self.config.name
        spec = getattr(self, "_spec", None)
        if spec:
            return spec.name
        schema = getattr(self, "_schema", None)
        if schema:
            return schema.name
        try:
            return self.get_schema().name
        except Exception:
            return getattr(self, "_name", self.__class__.__name__)

    @property
    def spec(self) -> ToolSpec:
        spec = getattr(self, "_spec", None)
        if spec is not None:
            return spec
        schema = self.get_schema()
        return ToolSpec(
            name=schema.name,
            description=schema.description,
            parameters=schema.parameters,
            required_args=schema.required_args or list(schema.parameters.get("required", [])),
            commit_horizon_args=getattr(schema, "commit_horizon_args", []),
            is_read_only=not schema.is_side_effect,
            side_effects=schema.is_side_effect,
            requires_approval=getattr(schema, "requires_approval", False),
            is_idempotent=getattr(schema, "is_idempotent", True),
        )

    def get_schema(self) -> ToolSchema:
        """Return schema and metadata for this tool."""
        schema = getattr(self, "_schema", None)
        if schema is not None:
            return schema
        if hasattr(self, "config"):
            c = getattr(self, "config")
            return ToolSchema(
                name=getattr(c, "name", "tool"),
                description=getattr(c, "description", ""),
                parameters=getattr(c, "parameters", {}),
                is_side_effect=getattr(c, "is_side_effect", False),
                requires_approval=getattr(c, "requires_approval", False),
                cost_usd=getattr(c, "cost_usd", 0.0),
                cache_ttl_s=getattr(c, "cache_ttl_s", None),
            )
        return ToolSchema(name=self.name)

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call asynchronously."""
        ...

    async def validate_args(self, args: dict[str, Any]) -> bool:
        """Validate input arguments against schema."""
        schema = self.get_schema()
        required = schema.parameters.get("required", []) or schema.required_args
        for req in required:
            if req not in args:
                return False
        return True

    async def cancel(self, call_id: str) -> bool:
        """Attempt to cancel an in-flight tool call."""
        return False


class ToolRegistry:
    """Manages registered tool adapters."""

    def __init__(self, tools: Optional[List[BaseToolAdapter]] = None) -> None:
        self._tools: Dict[str, BaseToolAdapter] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool: BaseToolAdapter) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseToolAdapter]:
        return self._tools.get(name)

    def list_specs(self) -> List[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def list_schemas(self) -> List[ToolSchema]:
        return [tool.get_schema() for tool in self._tools.values()]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())


class BaseLLMAdapter(ABC):
    """Abstract interface for model providers and simulators."""

    async def generate(
        self,
        prompt: str,
        tools: Optional[list[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Tuple[str, list[ToolCall], TokenUsage]:
        """Generate response and tool calls."""
        task = Task(prompt=prompt)
        specs = [ToolSpec(name=t.name, description=t.description, parameters=t.parameters) for t in (tools or [])]
        dec = await self.decide(task, [], specs)
        usage = TokenUsage(prompt_tokens=dec.input_tokens, completion_tokens=dec.output_tokens, total_tokens=dec.input_tokens + dec.output_tokens)
        return dec.reasoning or str(dec.final_answer or ""), dec.tool_calls, usage

    async def stream_generate(
        self,
        prompt: str,
        tools: Optional[list[ToolSchema]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        """Stream token chunks and partial tool calls."""
        task = Task(prompt=prompt)
        specs = [ToolSpec(name=t.name, description=t.description, parameters=t.parameters) for t in (tools or [])]
        async for chunk in self.stream_decision(task, [], specs):
            yield chunk

    async def decide(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        tools: List[ToolSpec],
    ) -> LLMDecision:
        """Generates a complete decision."""
        schemas = [ToolSchema(name=t.name, description=t.description, parameters=t.parameters) for t in tools]
        text, calls, usage = await self.generate(task.prompt, schemas)
        return LLMDecision(
            reasoning=text,
            tool_calls=calls,
            final_answer=text if not calls else None,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    async def stream_decision(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        tools: List[ToolSpec],
    ) -> AsyncIterator[StreamingChunk]:
        """Streams token chunks."""
        decision = await self.decide(task, history, tools)
        yield StreamingChunk(
            text=decision.reasoning,
            delta_text=decision.reasoning,
            is_final=True,
            token_index=decision.output_tokens,
            parsed_tool_calls=decision.tool_calls,
            commit_horizon_ready=decision.tool_calls,
        )

    async def predict_draft(
        self,
        task: Task,
        history: List[Dict[str, Any]],
        tools: List[ToolSpec],
    ) -> Optional[ToolCall]:
        """Draft speculative predictor."""
        return await self.predict_speculative_call(task.prompt, history)

    async def predict_speculative_call(
        self,
        prompt: str,
        history: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> Optional[ToolCall]:
        """Draft model prediction for speculative execution."""
        return None
