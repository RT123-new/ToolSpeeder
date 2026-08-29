# Legacy Untrusted Artifacts Audit Index

> [!WARNING]
> Historical benchmark bundles and outputs generated prior to the complete scientific-integrity repair are **quarantined and noncanonical**. They must not be cited or used as evidence for any empirical speedup or safety claims.

---

## 1. Audit Ledger

| Historical Commit | Historical Path | Invalidation Reason | Payload Hash Summary |
|---|---|---|---|
| `2066ae80e40905e7ffca50bb148d74700094101a` | `artifacts/legacy-untrusted/pr1-head-2066ae80/` | Synthetic Monte Carlo simulation math was relabelled with `replay_integration` and `local_wall_clock` tags; unhandled `asyncio.CancelledError` on Python 3.10/3.11; arbitrary callable execution in E2 fusion; unverified formulas used instead of real schedulers. | Manifest: `none`, Result: `b6b85f9b...` (replay), `9a5265e3...` (local) |
| `050176b98a40273491df4ad55526016a5d11c4ae` | `artifacts/legacy-untrusted/pr1-head-050176b/` | Manifest generated with `git_dirty: true`; missing required protocol and fixture hashes; W7 null latency and 0% success; single-threaded local HTTP server; unisolated database states; missing canonical raw-trace recomputation. | Manifest: `9f9c7b90...` (replay), `736146ec...` (local), Result: `7b8eee8c...` (replay), `c065eefb...` (local) |

---

## 2. Quarantined File Hash Index

### Head `050176b98a40273491df4ad55526016a5d11c4ae`
- `replay/benchmark_result.json`: `7b8eee8c640efa2f6592c8178899f576a658e69c022aa458d0c00c638bbd5458`
- `replay/manifest.json`: `9f9c7b90670ed99d69b395bc1c8e43f977f66214c43ae92fdf60eb12447cf2d1`
- `replay/raw-traces.jsonl`: `26f8e8cc41288a35c115d7ae980dab809ccccbc516e6acc53f484b75ad361bbe`
- `replay/report.html`: `6be25a50861e125cc86d564c7cc796a7a08438032b626e9d69430b41e6c3fb6f`
- `replay/report.md`: `16eba77971d76c39b90705a7e8e5b7521ffe0fc194455cab17905762d8bd9510`
- `replay/summary.json`: `87152d40d24b302d770b43a0f1f22b02d2be4cae7eb00935427543c8b6c8c37f`
- `local/benchmark_result.json`: `c065eefb1d6be5b72b16c91620e7434bcfb8bf8da9a82335691557adf878e302`
- `local/manifest.json`: `736146ec4e69e3785f6b0097cfb40b11c69afb0a6c758b10bfd7f5d0638bd713`
- `local/raw-traces.jsonl`: `c7fa099d6436743b469f5993cdbf8bf433066a7d26c2bc42c0abde3001c4ef60`
- `local/report.html`: `ae059dc2beedc2f6443b1f0f9c8443e8cf3f074941906e5a6a6b643626b0081c`
- `local/report.md`: `0e7dee3570ea39accc724d6332f93fbfd174487ead91e3dd4f31f46528eedf1c`
- `local/summary.json`: `0f55b1be9890d7bb6bbca1de8e664e663ad8dd881992d062ee6f221e1bf60752`

### Head `2066ae80e40905e7ffca50bb148d74700094101a`
- `replay/benchmark_result.json`: `b6b85f9b132c672a27eda5a4bf57581c398323b3d1f5c484beb74be798c2a2c9`
- `replay/summary_report.json`: `cc7fa41b5526be35a0f0b4e30ecbe9ba891c57d2a00f1fbe107a85d7675dfd4a`
- `replay/EVIDENCE_LOG.md`: `1ce725fea8e63c19933e60d55d2c7afbed8332e035a21b3f63440d7b9b5f2728`
- `replay/dashboard.html`: `64642f99a0d7decaef0cd8762f668d5a12f7b9962c0f7fede8d394b8f552f3ba`
- `replay/workload_summary.csv`: `4a882e3214a85bd53224d87f96d9262f862741439fcc27f66530d3d1a9a9ddda`
- `local/benchmark_result.json`: `9a5265e3139205f8ae91ab71ff581f247f04fdfac44c0eea28b1bee6dd94f6e0`
- `local/summary_report.json`: `fa8c50cfc55eb612d8f639585a1602b23936ea95307d186d3c6542759c3872fe`
- `local/EVIDENCE_LOG.md`: `3ac83adbabb20e6f5573cd381dbc090936b347c26321b9774ae412de994fa24a`
- `local/dashboard.html`: `9abe83c20fcd32e4570e6e0aa78e06f7a67456fa36b8bd20da823fdd1664d879`
- `local/workload_summary.csv`: `de5dc0adee2f81d3f70c5b154022776b1a35be3f0551b75e37c1ce2337a3ce18`

---

## 3. Governance Policy
1. Canonical evidence must be produced using the frozen `tool-speed-v1.1.json` protocol.
2. Generated evidence bundles are transient test artifacts during CI runs and are preserved exclusively via GitHub Actions artifact uploads.
