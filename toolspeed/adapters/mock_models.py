"""Simulated async LLM with token streaming, reasoning pauses, bytecode codecs, and draft predictors."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from toolspeed.adapters.base import BaseLLMAdapter, LLMDecision, StreamingChunk, ToolSchema
from toolspeed.core.types import AgentTask, LatencyProfile, TokenUsage, ToolCall, ToolSpec


class ActionBytecodeCodec:
    """Compact action bytecode encoder/decoder for tool calls (Experiment E5).

    Replaces verbose JSON formatting with compact typed binary action tokens,
    accelerating tool-call decode bandwidth.
    """

    OP_TOOL_CALL = 0x01
    OP_FINAL_RETURN = 0x02

    @classmethod
    def encode(cls, call: ToolCall) -> bytes:
        """Encode a ToolCall into compact action bytecode."""
        t_name = call.tool_name or call.name
        tool_name_bytes = t_name.encode("utf-8")
        args_json = json.dumps(call.arguments, separators=(",", ":")).encode("utf-8")

        # Format: [OP(1B)][name_len(2B)][name_bytes][args_len(4B)][args_bytes]
        header = struct.pack("!BH", cls.OP_TOOL_CALL, len(tool_name_bytes))
        body = struct.pack("!I", len(args_json)) + args_json
        return header + tool_name_bytes + body

    @classmethod
    def decode(cls, data: bytes) -> ToolCall:
        """Decode compact action bytecode into a ToolCall."""
        if len(data) < 3:
            raise ValueError("Bytecode too short: minimum header is 3 bytes.")
        op, name_len = struct.unpack("!BH", data[:3])
        if op != cls.OP_TOOL_CALL:
            raise ValueError(f"Unknown opcode: {op}")

        offset = 3
        if len(data) < offset + name_len + 4:
            raise ValueError("Bytecode packet truncated: missing tool name or args length header.")

        try:
            tool_name = data[offset : offset + name_len].decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Malformed UTF-8 in tool name: {e}") from e
        offset += name_len

        args_len = struct.unpack("!I", data[offset : offset + 4])[0]
        offset += 4

        if len(data) < offset + args_len:
            raise ValueError(
                f"Bytecode packet truncated: expected {args_len} bytes for arguments, got {len(data) - offset}."
            )

        args_json_bytes = data[offset : offset + args_len]
        try:
            arguments = json.loads(args_json_bytes.decode("utf-8")) if args_json_bytes else {}
        except Exception as e:
            raise ValueError(f"Malformed JSON argument payload in bytecode: {e}") from e

        if not isinstance(arguments, dict):
            arguments = {"value": arguments}

        return ToolCall(
            name=tool_name,
            tool_name=tool_name,
            arguments=arguments,
            bytecode=data,
        )

    @classmethod
    def estimate_token_count(cls, text_or_bytes: str | bytes) -> int:
        """Estimate token count (approx 4 chars per token for text, 4 bytes per token for bytecode)."""
        length = len(text_or_bytes)
        return max(1, (length + 3) // 4)


@dataclass
class ModelCostConfig:
    """Pricing configuration for simulated token consumption ($ per 1,000,000 tokens)."""

    prompt_cost_per_million: float = 2.50
    completion_cost_per_million: float = 10.00

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.prompt_cost_per_million / 1_000_000.0) + (
            completion_tokens * self.completion_cost_per_million / 1_000_000.0
        )


class DraftPredictorModel:
    """Fast auxiliary draft model for speculative tool dispatch (Experiment E3)."""

    def __init__(
        self,
        latency_ms: float = 70.0,
        accuracy: float = 0.85,
        confidence_threshold: float = 0.70,
        seed: int | None = None,
        clock: Any = None,
    ):
        self.latency_ms = latency_ms
        self.accuracy = accuracy
        self.confidence_threshold = confidence_threshold
        self.clock = clock
        self._rng = np.random.default_rng(seed)

    async def _sleep_ms(self, delay_ms: float) -> None:
        if self.clock is not None and hasattr(self.clock, "sleep_ms"):
            await self.clock.sleep_ms(delay_ms)
        elif delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

    async def predict(
        self,
        prompt: str,
        candidate_tools: list[str],
        ground_truth_tool: str | None = None,
        ground_truth_args: dict[str, Any] | None = None,
    ) -> tuple[ToolCall, float] | None:
        """Simulate fast draft prediction with confidence score."""
        await self._sleep_ms(max(0.0001, self.latency_ms))

        confidence = float(self._rng.uniform(0.5, 1.0))
        if confidence < self.confidence_threshold:
            return None

        is_correct = self._rng.random() < self.accuracy
        if is_correct and ground_truth_tool is not None:
            tool_name = ground_truth_tool
            args = dict(ground_truth_args or {})
        else:
            other_tools = [t for t in candidate_tools if t != ground_truth_tool]
            tool_name = (
                str(self._rng.choice(other_tools))
                if other_tools
                else (candidate_tools[0] if candidate_tools else "default_tool")
            )
            args = {"predicted_arg": "fallback_speculation"}

        call = ToolCall(
            name=tool_name,
            tool_name=tool_name,
            arguments=args,
            is_speculative=True,
            speculation_confidence=confidence,
            metadata={"draft_confidence": confidence, "is_prediction_correct": is_correct},
        )
        return call, confidence


class SimulatedLLM(BaseLLMAdapter):
    """Simulated async LLM with token streaming, reasoning pauses, and bytecode dispatch."""

    def __init__(
        self,
        profile: LatencyProfile | None = None,
        tokens_per_second: float = 100.0,
        cost_config: ModelCostConfig | None = None,
        draft_accuracy: float = 0.85,
        draft_confidence_threshold: float = 0.70,
        use_bytecode: bool = False,
        commit_fraction: float = 0.5,
        seed: int | None = None,
        clock: Any = None,
    ):
        self.profile = profile or LatencyProfile()
        self.tokens_per_second = max(1.0, tokens_per_second)
        self.cost_config = cost_config or ModelCostConfig()
        self.use_bytecode = use_bytecode
        self.commit_fraction = commit_fraction
        self.clock = clock
        self._rng = np.random.default_rng(seed)
        self.draft_predictor = DraftPredictorModel(
            latency_ms=self.profile.draft_model_ms,
            accuracy=draft_accuracy,
            confidence_threshold=draft_confidence_threshold,
            seed=seed,
            clock=clock,
        )

    def _sample_reasoning_ms(self, is_final: bool = False) -> float:
        median = self.profile.model_final_ms if is_final else self.profile.model_decision_ms
        return float(self._rng.lognormal(np.log(max(1.0, median)), self.profile.sigma))

    async def _sleep_ms(self, delay_ms: float) -> None:
        if self.clock is not None and hasattr(self.clock, "sleep_ms"):
            await self.clock.sleep_ms(delay_ms)
        elif delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

    async def generate(
        self,
        prompt: str,
        tools: list[ToolSchema] | None = None,
        expected_calls: list[ToolCall] | None = None,
        final_answer: str = "Completed successfully.",
        is_final: bool = False,
        **kwargs: Any,
    ) -> tuple[str, list[ToolCall], TokenUsage]:
        reasoning_ms = self._sample_reasoning_ms(is_final=is_final)
        await self._sleep_ms(max(0.0001, reasoning_ms))

        calls = list(expected_calls or [])
        if self.use_bytecode:
            for call in calls:
                call.bytecode = ActionBytecodeCodec.encode(call)

        prompt_tokens = max(10, len(prompt.split()) * 2)
        if is_final:
            completion_tokens = max(5, len(final_answer.split()) * 2)
        else:
            calls_repr = json.dumps([c.to_dict() for c in calls])
            completion_tokens = max(5, len(calls) * 12) if self.use_bytecode else max(10, len(calls_repr.split()) * 2)

        cost = self.cost_config.calculate_cost(prompt_tokens, completion_tokens)
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
        )

        return final_answer if is_final else "", calls, token_usage

    async def decide(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> LLMDecision:
        is_final = len(history) >= 2 or not tools
        reasoning_ms = self._sample_reasoning_ms(is_final=is_final)
        await self._sleep_ms(max(0.0001, reasoning_ms))

        if is_final:
            # Derive answer strictly from tool results in history or prompt context
            last_tool_res = None
            for h in reversed(history):
                if h.get("role") == "tool":
                    last_tool_res = h.get("output")
                    break

            return LLMDecision(
                reasoning="Task completed.",
                final_answer=last_tool_res if last_tool_res is not None else {"status": "done", "prompt": task.prompt},
                duration_ms=reasoning_ms,
                input_tokens=100,
                output_tokens=20,
            )
        else:
            tool_calls = []
            if tools:
                tool_calls.append(ToolCall(name=tools[0].name, arguments={}))
            return LLMDecision(
                reasoning="Executing tool.",
                tool_calls=tool_calls,
                duration_ms=reasoning_ms,
                input_tokens=150,
                output_tokens=30,
            )

    async def stream_generate(
        self,
        prompt: str,
        tools: list[ToolSchema] | None = None,
        expected_calls: list[ToolCall] | None = None,
        final_answer: str = "Completed successfully.",
        is_final: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        reasoning_ms = self._sample_reasoning_ms(is_final=is_final)
        await self._sleep_ms(max(0.0001, reasoning_ms))

        target_text = final_answer if is_final else ""
        calls = list(expected_calls or [])
        token_delay_ms = 1000.0 / self.tokens_per_second

        if is_final:
            words = target_text.split()
            for idx, word in enumerate(words):
                await self._sleep_ms(token_delay_ms)
                is_last = idx == len(words) - 1
                yield StreamingChunk(
                    text=word + " ",
                    delta_text=word + " ",
                    is_final=is_last,
                    token_index=idx,
                )
        else:
            total_call_tokens = max(10, len(calls) * 20)
            commit_token_idx = int(total_call_tokens * self.commit_fraction)

            for idx in range(total_call_tokens):
                await self._sleep_ms(token_delay_ms)
                chunk_delta = None
                meta: dict[str, Any] = {}

                if idx == commit_token_idx:
                    meta["commit_horizon_reached"] = True
                    meta["committed_calls"] = [c.to_dict() for c in calls]

                is_last = idx == total_call_tokens - 1
                if is_last:
                    meta["completed_calls"] = [c.to_dict() for c in calls]

                fragment = json.dumps(calls[0].arguments) if (calls and idx >= commit_token_idx) else ""

                yield StreamingChunk(
                    text="",
                    delta_text="",
                    tool_call_delta=chunk_delta,
                    is_final=is_last,
                    token_index=idx,
                    parsed_tool_calls=calls if is_last else [],
                    commit_horizon_ready=calls if (idx >= commit_token_idx and idx == commit_token_idx) else [],
                    raw_json_fragment=fragment,
                    metadata=meta,
                )

    async def predict_speculative_call(
        self,
        prompt: str,
        history: list[Any] | None = None,
        ground_truth_tool: str | None = None,
        ground_truth_args: dict[str, Any] | None = None,
        candidate_tools: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolCall | None:
        res = await self.draft_predictor.predict(
            prompt=prompt,
            candidate_tools=candidate_tools or ["read_db", "fetch_url", "compute_stats"],
            ground_truth_tool=ground_truth_tool,
            ground_truth_args=ground_truth_args,
        )
        if res is None:
            return None
        call, _ = res
        return call


class MockScriptedLLM(BaseLLMAdapter):
    """Scripted LLM simulator with token streaming, draft speculation, and commit horizons."""

    def __init__(
        self,
        decision_steps: list[LLMDecision] | None = None,
        decision_fn: Callable[[AgentTask, list[dict[str, Any]]], LLMDecision] | None = None,
        draft_predictor_fn: Callable[[AgentTask, list[dict[str, Any]]], ToolCall | None] | None = None,
        simulated_decision_ms: float = 5.0,
        simulated_draft_ms: float = 1.0,
        commit_horizon_fraction: float = 0.4,
        token_chunk_count: int = 4,
        clock: Any = None,
    ) -> None:
        self.decision_steps = list(decision_steps or [])
        self.decision_fn = decision_fn
        self.draft_predictor_fn = draft_predictor_fn
        self.simulated_decision_ms = simulated_decision_ms
        self.simulated_draft_ms = simulated_draft_ms
        self.commit_horizon_fraction = commit_horizon_fraction
        self.token_chunk_count = token_chunk_count
        self.clock = clock
        self._current_step = 0

    def reset(self) -> None:
        self._current_step = 0

    def _now_s(self) -> float:
        if self.clock is not None and hasattr(self.clock, "now_s"):
            return self.clock.now_s()
        return time.perf_counter()

    async def _sleep_ms(self, delay_ms: float) -> None:
        if self.clock is not None and hasattr(self.clock, "sleep_ms"):
            await self.clock.sleep_ms(delay_ms)
        elif delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

    async def decide(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> LLMDecision:
        start = self._now_s()

        if self.simulated_decision_ms > 0:
            await self._sleep_ms(self.simulated_decision_ms)

        decision: LLMDecision
        if self.decision_fn:
            decision = self.decision_fn(task, history)
        elif self._current_step < len(self.decision_steps):
            decision = self.decision_steps[self._current_step]
            self._current_step += 1
        else:
            # Derive answer strictly from observed tool history or prompt context
            last_tool_res = None
            for h in reversed(history):
                if h.get("role") == "tool":
                    last_tool_res = h.get("output")
                    break

            decision = LLMDecision(
                reasoning="Completed task.",
                final_answer=last_tool_res if last_tool_res is not None else {"status": "done", "prompt": task.prompt},
                input_tokens=150,
                output_tokens=30,
            )

        duration_ms = (self._now_s() - start) * 1000.0
        decision.duration_ms = duration_ms
        return decision

    async def stream_decision(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> AsyncIterator[StreamingChunk]:
        if self.decision_fn:
            target = self.decision_fn(task, history)
        elif self._current_step < len(self.decision_steps):
            target = self.decision_steps[self._current_step]
            self._current_step += 1
        else:
            last_tool_res = None
            for h in reversed(history):
                if h.get("role") == "tool":
                    last_tool_res = h.get("output")
                    break

            target = LLMDecision(
                reasoning="Finished.",
                final_answer=last_tool_res if last_tool_res is not None else {"status": "done", "prompt": task.prompt},
                input_tokens=100,
                output_tokens=20,
            )

        total_chunks = max(2, self.token_chunk_count)
        per_chunk_sleep_ms = (self.simulated_decision_ms / total_chunks) if self.simulated_decision_ms > 0 else 0.0
        commit_chunk_index = int(total_chunks * self.commit_horizon_fraction)

        full_text = target.reasoning or (str(target.final_answer) if target.final_answer is not None else "")
        words = full_text.split() if full_text else [f"token_{i}" for i in range(total_chunks)]

        commit_emitted = False
        for chunk_idx in range(total_chunks):
            if per_chunk_sleep_ms > 0:
                await self._sleep_ms(per_chunk_sleep_ms)

            is_last = chunk_idx == (total_chunks - 1)
            is_commit_time = (chunk_idx >= commit_chunk_index) and not commit_emitted and bool(target.tool_calls)

            commit_ready = target.tool_calls if is_commit_time else []
            if is_commit_time:
                commit_emitted = True

            chunk_word = words[chunk_idx] if chunk_idx < len(words) else f"token_{chunk_idx}"
            chunk_text = (chunk_word + " ") if not is_last else chunk_word

            meta: dict[str, Any] = {}
            if is_last and target.final_answer is not None:
                meta["final_answer"] = target.final_answer

            fragment = ""
            if commit_ready:
                fragment = json.dumps(commit_ready[0].arguments)
            elif is_last and target.tool_calls:
                fragment = json.dumps(target.tool_calls[0].arguments)
            elif is_last:
                fragment = f'{{"step": {chunk_idx}}}'

            yield StreamingChunk(
                text=chunk_text,
                delta_text=chunk_text,
                is_final=is_last,
                token_index=chunk_idx,
                parsed_tool_calls=target.tool_calls if is_last else [],
                commit_horizon_ready=commit_ready,
                raw_json_fragment=fragment,
                metadata=meta,
            )

    async def predict_draft(
        self,
        task: AgentTask,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> ToolCall | None:
        if self.simulated_draft_ms > 0:
            await self._sleep_ms(self.simulated_draft_ms)

        if self.draft_predictor_fn is not None:
            return self.draft_predictor_fn(task, history)

        prompt_lower = task.prompt.lower()
        for t in tools:
            if t.is_read_only and not t.side_effects and not t.requires_approval and t.name.lower() in prompt_lower:
                return ToolCall(
                    name=t.name,
                    tool_name=t.name,
                    arguments={"query": task.prompt},
                    is_speculative=True,
                    speculation_confidence=0.85,
                )
        return None
