# Contributing to ToolSpeed

Thank you for contributing to ToolSpeed!

## Scientific Integrity Standards

ToolSpeed enforces strict scientific integrity and evidence taxonomy:

1. **Evidence Taxonomy Separation**:
   - `SYNTHETIC`: Deterministic mathematical simulation models.
   - `REPLAY_INTEGRATION`: Real schedulers executing deterministic virtual-delay adapters.
   - `LOCAL_WALL_CLOCK`: Real schedulers executing real local I/O, SQLite, sandboxes, and mock HTTP servers.
   - `LIVE`: Real schedulers executing live production LLMs and cloud APIs.
   *Synthetic evidence must never be presented as empirical proof of scheduler implementation speedups.*

2. **Negative Controls & Null Baselines**:
   Every new optimization feature must include a negative control proving that disabling the feature yields a speedup of ~1.0x with no artificial gains.

3. **Mandatory Test Verification**:
   All PRs must pass the 22 adversarial integrity unit tests in `tests/test_adversarial_integrity.py` and the complete test discovery suite across Python 3.10–3.13.

## Local Development Workflow

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run full unit tests
python -m unittest discover -s tests -p "test_*.py"

# Run paired benchmark suites
toolspeed benchmark --backend replay --trials 50
toolspeed benchmark --backend local --trials 10
```
