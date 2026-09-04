# Audit & Quarantine Index: September 2, 2026 Replay Diagnostics

**Audit Classification:** `NONCANONICAL EXPLORATORY DIAGNOSTIC OUTPUTS`  
**Quarantine Location:** `artifacts/noncanonical/2026-09-02-replay-diagnostics/`  
**Audit Date:** 2026-09-03  
**Auditor:** Gemini Flash (Primary Orchestrator / Evidence Integrity Context)

---

## 1. Executive Summary & Quarantine Justification

On September 2, 2026, three separate replay benchmark runs were produced and reported as a 3-seed confirmatory empirical evaluation under protocol v1.1/v1.2. 

Following independent verification, these outputs have been formally quarantined and reclassified as **noncanonical exploratory diagnostics**. They do not constitute prospective confirmatory evidence for the following authoritative reasons:

1. **Synthetic Seed Matrix**: The benchmark CLI did not execute a true 3-seed matrix across trials. Instead, three standalone single-seed runs were invoked, and `cases.jsonl` has identical SHA-256 digests (`f957de79...`) across all three seeds—proving the random seed did not materially vary the benchmark fixtures.
2. **Retrospective Protocol**: The embedded protocol is `tool-speed-v1.1`, which is explicitly classified as `retrospective_repair`. Protocol `v1.2-draft` was unfrozen and in draft status during execution.
3. **Hard-Coded Negative and Positive Controls**: `controls-traces.jsonl` has the identical hash (`3b9240af...`) across all three runs because controls returned hardcoded literal values (`1.0x` and `2.0x`) rather than running genuine comparison arms.
4. **Oracle Contamination in E2**: Code execution at SHA `1fc9ebc` included direct access to `ctx.task.expected_output` in `JITFusionScheduler`, polluting candidate performance.
5. **Hard-Coded Comparison Baselines**: The harness did not execute the protocol's primary attribution baselines (e.g. W1 ran against SyncReAct instead of serial ablation; W2 ran against SyncReAct instead of fusion-disabled).
6. **Incomplete Bundle Sealing**: `raw_trace_hash` in the manifest computed only the candidate trace hash; `result_hash` was written to external `manifest.json` after `result.json` was already saved, leaving manifests mismatched.

---

## 2. Quarantined Bundles Audit Ledger

### Bundle 1: `confirmatory_seed42`
- **Original Path:** `artifacts/confirmatory_seed42/`
- **Quarantined Path:** `artifacts/noncanonical/2026-09-02-replay-diagnostics/confirmatory_seed42/`
- **Total Directory Size:** 97.4 MB (15 files)
- **Embedded Code Git SHA:** `1fc9ebc4a38ba659ee9be3d3cd3277336f36643c`
- **Git Tree Dirty:** `False`
- **Embedded Seed:** `42`
- **Trial Count:** 1,000 paired trials across 9 workloads (9,000 baseline traces, 9,000 candidate traces, 9,000 cases)
- **External Manifest Hash:** `37a6ce3eb3c63fa82ea77e48df98a35ffcc105cc3592df2bfa6859c92268127f`
- **Result Hash (`result.json`):** `b9e6648e02ee9024bc3016e32e388b01d8de7f1eddc7a65b6bad10bdb8794d57`
- **File Checksums (`bundle.sha256`):**
  ```text
  7f0fdbb120a2571675836abbd384d219a47598050297506f5c1d7f522701e5fe  baseline-traces.jsonl
  5df51bf09bd41a3ee5c3851e9af4dba8fbb6bd6966783abe07750f31e8b1bd04  benchmark-plan.json
  b9e6648e02ee9024bc3016e32e388b01d8de7f1eddc7a65b6bad10bdb8794d57  benchmark_result.json
  9cbd028e4bb880f7dd4432f146095e896602b5f1f8ece7a3cc95aef461da7874  candidate-traces.jsonl
  f957de79995b80c80c0ee26ee7f5bdf2bb26f2d1cfe2e9221f696caaf509e256  cases.jsonl
  3b9240af4c9fc2ee4d20bea2fd3d5f992eff7ecdf2f63f857ec42e87bc06d0b7  controls-traces.jsonl
  c723443e999d5a00eb16f80df5bff1d1b5966f567b902742bda1667d8f9aebdc  falsification.json
  37a6ce3eb3c63fa82ea77e48df98a35ffcc105cc3592df2bfa6859c92268127f  manifest.json
  5df51bf09bd41a3ee5c3851e9af4dba8fbb6bd6966783abe07750f31e8b1bd04  protocol.json
  8c3fc980c78678eebc12497cb90d4ed7d0931b0de37d0f7db232ac1594a2c0f5  report.html
  0ecd9b2a87ce79a3580506d0c143965545cd73898aa9c12a09da69f4ab782ba9  report.md
  b9e6648e02ee9024bc3016e32e388b01d8de7f1eddc7a65b6bad10bdb8794d57  result.json
  ```

---

### Bundle 2: `confirmatory_seed137`
- **Original Path:** `artifacts/confirmatory_seed137/`
- **Quarantined Path:** `artifacts/noncanonical/2026-09-02-replay-diagnostics/confirmatory_seed137/`
- **Total Directory Size:** 97.4 MB (15 files)
- **Embedded Code Git SHA:** `1fc9ebc4a38ba659ee9be3d3cd3277336f36643c`
- **Git Tree Dirty:** `False`
- **Embedded Seed:** `137`
- **Trial Count:** 1,000 paired trials across 9 workloads (9,000 baseline traces, 9,000 candidate traces, 9,000 cases)
- **External Manifest Hash:** `f179cdc1725b746120d8396bdc5caa242e57e3e83fa40b02304793aebeceb5e8`
- **Result Hash (`result.json`):** `12bf0489c4287af8bdf1db5b723ad28eb420effc672ea9dff0b0e695c3849864`
- **File Checksums (`bundle.sha256`):**
  ```text
  a8b8bed1a6ad870f26800eccb75452cfe1f93810f37a2ad2f03500f67f156b39  baseline-traces.jsonl
  5df51bf09bd41a3ee5c3851e9af4dba8fbb6bd6966783abe07750f31e8b1bd04  benchmark-plan.json
  12bf0489c4287af8bdf1db5b723ad28eb420effc672ea9dff0b0e695c3849864  benchmark_result.json
  956937dd6d2b5126b1d0d6804d3327e5c1c34822561da336f1c95449bfba6fae  candidate-traces.jsonl
  f957de79995b80c80c0ee26ee7f5bdf2bb26f2d1cfe2e9221f696caaf509e256  cases.jsonl
  3b9240af4c9fc2ee4d20bea2fd3d5f992eff7ecdf2f63f857ec42e87bc06d0b7  controls-traces.jsonl
  6d31cb1a9cd0548097f4f88939440de5cc07c0ea661b138d395e2dc22b2da763  falsification.json
  f179cdc1725b746120d8396bdc5caa242e57e3e83fa40b02304793aebeceb5e8  manifest.json
  5df51bf09bd41a3ee5c3851e9af4dba8fbb6bd6966783abe07750f31e8b1bd04  protocol.json
  88c7eacb325fcf46f03a7a9a89ad07a4aed35e576d10e4d146cb1fff1ed6220e  report.html
  b1df304848892d4ac2ca70c94a06b3e05a3ff7cd5dfcb921f856017fef128a17  report.md
  12bf0489c4287af8bdf1db5b723ad28eb420effc672ea9dff0b0e695c3849864  result.json
  ```

---

### Bundle 3: `confirmatory_seed2026`
- **Original Path:** `artifacts/confirmatory_seed2026/`
- **Quarantined Path:** `artifacts/noncanonical/2026-09-02-replay-diagnostics/confirmatory_seed2026/`
- **Total Directory Size:** 97.4 MB (15 files)
- **Embedded Code Git SHA:** `1fc9ebc4a38ba659ee9be3d3cd3277336f36643c`
- **Git Tree Dirty:** `False`
- **Embedded Seed:** `2026`
- **Trial Count:** 1,000 paired trials across 9 workloads (9,000 baseline traces, 9,000 candidate traces, 9,000 cases)
- **External Manifest Hash:** `e008550c03c422eb15e0a000a064a14aa7c0babbc9938f97105652bbfbe86109`
- **Result Hash (`result.json`):** `2ae655e14b80fbd9aa0860d0e9bc003b61ee30df7ac9335eb28575d56a83a6a8`
- **File Checksums (`bundle.sha256`):**
  ```text
  b65a8faaefd96d582fe63da1d1854e15a7ed768944ca8aa2a23a91dd9ba4cfa1  baseline-traces.jsonl
  5df51bf09bd41a3ee5c3851e9af4dba8fbb6bd6966783abe07750f31e8b1bd04  benchmark-plan.json
  2ae655e14b80fbd9aa0860d0e9bc003b61ee30df7ac9335eb28575d56a83a6a8  benchmark_result.json
  26823cb91983a9f674324d50ff3d229ad7c3d3e4b7afde49296c1eb20c75303e  candidate-traces.jsonl
  f957de79995b80c80c0ee26ee7f5bdf2bb26f2d1cfe2e9221f696caaf509e256  cases.jsonl
  3b9240af4c9fc2ee4d20bea2fd3d5f992eff7ecdf2f63f857ec42e87bc06d0b7  controls-traces.jsonl
  1dc618b226e521022195da0bca851a35315d4be626aa2e9290a9346f5c59c689  falsification.json
  e008550c03c422eb15e0a000a064a14aa7c0babbc9938f97105652bbfbe86109  manifest.json
  5df51bf09bd41a3ee5c3851e9af4dba8fbb6bd6966783abe07750f31e8b1bd04  protocol.json
  7909ae016733f124cd4bc4a39e490f49d0a82c386105ebdc802cb0494e0fa8fb  report.html
  13ff4b01d3d37f9efb05b7bf010ed1179b1225dd3bba062da8c446266622bc28  report.md
  2ae655e14b80fbd9aa0860d0e9bc003b61ee30df7ac9335eb28575d56a83a6a8  result.json
  ```
