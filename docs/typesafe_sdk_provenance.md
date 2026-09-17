# TypeSafe AI SDK & System One Adapter Provenance

## 1. Overview
This document records the exact first-party interfaces, versions, schemas, and runtime behaviors of the TypeSafe SDK and System One Adapter inspected for the ToolSpeeder speculative routing experiment.

* **Date Inspected:** 2026-09-17
* **Official Docs:** `https://docs.typesafe.ai/`
* **Official SDK Repository:** `https://github.com/typesafe-ai/typesafe-sdk-python`
* **System One Adapter Repository:** `https://github.com/typesafe-ai/system-one-adapter-python`

---

## 2. TypeSafe SDK Specifications

* **PyPI Package:** `typesafe-sdk`
* **Pinned Version:** `0.6.0`
* **Repository Git Commit:** `bda0433b4cc515d1cfbafb110c7e6388534a5641` (branch `main`)
* **Python Compatibility:** `>=3.10`
* **Core Dependencies:** `httpx2`, `msgspec`, `tenacity`, `anyio`, `truststore`

### Public Interfaces & Primitives
The SDK exposes structured System One decision primitives rather than conversational text generation:

1. **`Choice(instructions=..., criteria={label: description})`**:
   - Discrete label selection across predefined criteria.
   - Evaluates probability for each candidate label.
2. **`Noul(instructions=..., criteria={"true": ..., "false": ...})`**:
   - Probabilistic boolean primitive ("Yes/No").
   - Returns float probability in `[0.0, 1.0]`.
3. **`Score(instructions=..., criteria=[...])`**:
   - Numerical score evaluated against an ordered rubric list.

### Client Interfaces (Sync & Async)
- Sync Client: `TypeSafeClient(api_key=..., base_url=..., timeout=..., retry=...)`
- Async Client: `AsyncTypeSafeClient(api_key=..., base_url=..., timeout=..., retry=...)`
- Context manager support: `with TypeSafeClient() as client:` and `async with AsyncTypeSafeClient() as client:`

### Invocation Signature
```python
response = await client.system_one(
    state: JSONContent,  # str, dict, or list
    questions: Mapping[str, Question],
    *,
    model: str | None = None,  # defaults to TYPESAFE_DEFAULT_MODEL or server default
    retry: RetryPolicy | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, JSONValue | None] | None = None,
) -> SystemOneResponse
```

### Response Structure
`SystemOneResponse` is a frozen `msgspec.Struct` providing:
- `model: str`: Identifier of the model that evaluated the request.
- `usage: Usage`: Contains `input_tokens: int | None` and `output_tokens: int | None`.
- `answers: dict[str, Answer]`: Map of question key to typed answer object.
- `choices: dict[str, ChoiceAnswer]`:
  - `choice: str`: Selected label.
  - `confidence: float`: Model confidence score in `[0.0, 1.0]`.
  - `probabilities: dict[str, float]`: Normalized probability distribution across all labels.
- `nouls: dict[str, NoulAnswer]`:
  - `noul: float`: Probability of `true` outcome in `[0.0, 1.0]`.
- `scores: dict[str, ScoreAnswer]`:
  - `score: float`, `confidence: float`, `probabilities: dict[int, float]`, `legend: dict[int, Any]`.

### Error Hierarchy
All exceptions inherit from `TypeSafeError`:
- `TypeSafeAPIError`: Base API error response.
- `TypeSafeAuthenticationError` (401 / 403)
- `TypeSafeBadRequestError` (400)
- `TypeSafeNotFoundError` (404)
- `TypeSafeRateLimitError` (429)
- `TypeSafeInternalServerError` (500)
- `TypeSafeAPITimeoutError`: HTTP/socket timeout.
- `TypeSafeAPIConnectionError`: Network reachability failure.
- `TypeSafeAPIResponseValidationError`: Schema validation mismatch.

### Timeout and Retry Policy
- Client-level and call-level `timeout` in seconds.
- `RetryPolicy(max_retries=..., initial_interval=..., max_interval=..., backoff_multiplier=...)`.

---

## 3. System One Adapter Specifications

* **PyPI Package:** `system-one-adapter`
* **Pinned Version:** `0.1.4`
* **Repository Git Commit:** `typesafe-ai/system-one-adapter-python@main`
* **Drop-in Role:** Replaces `TypeSafeClient` / `AsyncTypeSafeClient` with standard LLM endpoints (e.g. OpenAI, Anthropic) executing the identical `SystemOneResponse` schema.
* **Constructor:**
  ```python
  client = AsyncSystemOneAdapterClient(
      structured_outputs=True,
      llm_answer_mode="probabilities",
      normalize_probabilities=True,
      provider="openai",
      model="gpt-4o-mini",
  )
  ```
* **Use in ToolSpeeder:** Serves as the experimental comparator (`B4 — LLM System One comparator`) to isolate the effect of System One question formulation from the Jev model itself.

---

## 4. Key Constraints for ToolSpeeder Integration

1. **Jev makes typed decisions over predefined alternatives; it does not generate arbitrary tool calls or mutate arguments.**
2. **External Egress Boundary:** No API keys, credentials, secret URLs, authorization tokens, or benchmark ground truth labels may ever be transmitted to TypeSafe.
3. **Read-Only Invariant:** ToolSpeeder's safety classifier remains authoritative; Jev cannot authorize mutative tools.
4. **Latency Accounting:** Provider execution overhead (P50/P95 latency) must be counted in end-to-end Correct Completion Latency (CCL).
