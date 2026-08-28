# Legacy Untrusted Benchmark Bundles (`pr1-head-050176b`)

These outputs are archived legacy development artifacts and are **not canonical evidence**.

### Reasons for Quarantine
1. **Stale Commit SHA:** Manifests declare earlier commit `14c19751c218b8498712060e2aee81f62283390c` rather than the PR head.
2. **Dirty Tree:** Manifests declare `git_dirty: true`.
3. **No Hosted Full Evidence Run:** Generated outside of a hosted, clean-tree GitHub Actions workflow.
4. **Incomplete Raw Traces:** Traces contain only candidate summaries and omit baseline paired arms.
5. **Missing Execution Evidence:** Traces omit actual tool calls, arguments, results, timeline events, and initial/final state snapshots.
6. **Non-Reconstructable Statistics:** Statistics, percentiles, speedups, and confidence intervals cannot be independently calculated from the raw traces.
7. **Stored Verdict Trust:** `falsify` reported stored boolean flags instead of recomputing from raw evidence and a pre-registered plan.
8. **Unenforced Scientific Validation:** Validator accepted placeholder hashes and did not verify Git tree SHA or paired state isolation.
