# ToolSpeed — research and implementation plan

## Original hypothesis

An AI agent can reduce the time to a **correct completed task** by shortening or
removing the serial critical path between model reasoning and tool execution.

This is deliberately narrower than “make APIs faster.” The model usually cannot
change a remote service's internal runtime. It can avoid calls, start them
earlier, execute independent work together, replace repeated reasoning with
compiled control logic, and consume results more efficiently.

## Primary metric

`Correct Completion Latency (CCL)`

Measure from receipt of the user request to the first final answer or completed
action that passes an exact task validator.

Report P50, P95, and P99. Never report latency without task success.

## Guardrails

Also report:

- exact task success;
- tool-selection and argument accuracy;
- unnecessary and duplicated calls;
- speculative calls cancelled or wasted;
- cost per successful task;
- cache freshness violations;
- unsafe or duplicated side effects;
- peak concurrency and rate-limit failures.

## Baselines

1. Synchronous ReAct-style loop.
2. Native parallel calls where dependencies allow.
3. Same model, tools, prompts, seeds, and concurrency limits.
4. An oracle DAG baseline to expose how much parallelism is theoretically
   available.
5. A deterministic handwritten workflow for repeated tasks.

## Workload families

1. Independent fan-out reads.
2. Deterministic dependent chains.
3. Branching workflows where each result changes the next decision.
4. Repeated workflows with high plan locality.
5. Large tool arguments and large tool results.
6. Cold-start code/browser sandboxes.
7. Side-effecting actions requiring approval and idempotency.

## First five experiments

### E1 — DAG parallelism

One variable: scheduler only.

Success: at least 20% lower P95 CCL with no loss in success and no rate-limit
increase above 0.5 percentage points.

Failure: the planner invents false independence, or P95 improvement is below 10%.

### E2 — Programmatic/JIT workflow fusion

One variable: replace repeated model round-trips with deterministic code for a
bounded workflow.

Success: at least 25% lower P95 CCL and at least 20% fewer model input tokens,
with identical task outcomes.

Failure: runtime code needs to ask the model for intermediate decisions on more
than 15% of cases; the workflow is not deterministic enough to compile.

### E3 — Confidence-gated speculative reads

One variable: a small predictor launches a read-only call while the main model
reasons.

Success: at least 15% lower P95 CCL, under 20% wasted calls, under 5% added tool
cost, and zero correctness loss after verification.

Failure: tail latency regresses, predictor confidence is poorly calibrated, or
wrong calls contend with correct calls.

### E4 — Commit-horizon dispatch

One variable: dispatch after the tool name and all semantics-changing required
arguments are fixed, rather than after the complete JSON object.

Success: at least 10% lower P95 tool-start time and zero semantic mismatches in
one million generated calls.

Failure: any call launches with arguments later changed by the model.

### E5 — Action bytecode

One variable: replace verbose JSON scaffolding with compact typed action tokens,
then deterministically expand to the ordinary tool schema.

Success: at least 2x faster tool-call generation, equal or better exact argument
accuracy, and at least 15% lower end-to-end CCL on decode-dominated workloads.

Failure: end-to-end gain is below 5%, or compatibility/repair overhead erases the
decode gain.

## Stop rule

Do not begin model training until E1–E4 are instrumented against real tools.
System-level gains are cheaper, easier to falsify, and may make a new model
unnecessary.

## Phase 2 candidates

- semantic and exact tool-result caches with explicit freshness contracts;
- predictive sandbox prewarming;
- dynamic tool retrieval and lazy schema loading;
- stateful connections, persistent model sessions, and prefix/KV caching;
- streaming result sketches;
- latency-aware planning trained against critical-path cost.

## Phase 3 research ideas

- local state twins that answer from a continuously reconciled copy of remote
  state;
- optimistic read execution with cancellation and verification;
- a small action model running beside a larger reasoning model;
- learned tool surrogates used only for speculative continuation while the real
  result verifies;
- JIT compilation of repeated agent traces with automatic deoptimisation when
  assumptions fail;
- direct latent or fixed-width action heads that bypass text/JSON generation.

## Falsification

The central hypothesis is wrong for a workload when, after equalising success,
cost, concurrency, and safety, no tested mechanism improves P95 correct
completion latency by at least 10%.

## Evidence log template

| Experiment | Tested | Succeeded | Failed | Still unproven | Next action |
|---|---|---|---|---|---|
| Phase 0 simulator | Yes | Mechanisms behave as expected under declared assumptions | No real-world claim established | Model/tool integration and correctness under live failures | Instrument synchronous baseline |
