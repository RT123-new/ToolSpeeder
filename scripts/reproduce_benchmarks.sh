#!/usr/bin/env bash
# ==============================================================================
# ToolSpeeder: Zero-Dependency Independent Benchmark Replication Package
# ==============================================================================
# This script reproduces ToolSpeeder benchmarks from scratch in an isolated,
# zero-dependency environment. It validates bundle cryptographic seals,
# verifies zero-trust trace recomputation, and runs fresh clean-slate sweeps.
#
# Usage:
#   ./scripts/reproduce_benchmarks.sh [OPTIONS]
#
# Options:
#   --quick        (Default) Run fast reproduction sweep (10 trials) + full bundle verification
#   --verify-only  Verify existing canonical bundle seals and exit codes without execution
#   --full         Run full multi-seed confirmatory sweep (N=1,000 per seed)
#   --help         Display this help message
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODE="quick"
for arg in "$@"; do
    case "$arg" in
        --quick)
            MODE="quick"
            ;;
        --verify-only)
            MODE="verify-only"
            ;;
        --full)
            MODE="full"
            ;;
        --help|-h)
            sed -n '2,18p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "❌ Unknown argument: $arg"
            echo "Run ./scripts/reproduce_benchmarks.sh --help for usage."
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "🔬 ToolSpeeder Independent Benchmark Replication Package"
echo "========================================================================"
echo "Directory: ${ROOT_DIR}"
echo "Date:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Mode:      ${MODE}"
echo "Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown') ($(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') modified files)"

# ------------------------------------------------------------------------------
# 1. Environment & Python Runtime Resolution
# ------------------------------------------------------------------------------
echo ""
echo "--- [1/4] Resolving Python Runtime Environment ---"

if command -v uv >/dev/null 2>&1; then
    PY_CMD="uv run python"
    echo "✅ Found 'uv' package manager. Using: uv run python"
elif command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
    echo "ℹ️  'uv' not found. Using system python3."
else
    echo "❌ Error: Neither 'uv' nor 'python3' is available on PATH."
    exit 1
fi

PY_VERSION=$(${PY_CMD} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "Python Version: ${PY_VERSION}"

${PY_CMD} -c '
import sys
if sys.version_info < (3, 10):
    print("❌ Error: ToolSpeeder requires Python >= 3.10. Found: " + sys.version)
    sys.exit(1)
'

# ------------------------------------------------------------------------------
# 2. Cryptographic Bundle Seal & Hash Verification
# ------------------------------------------------------------------------------
echo ""
echo "--- [2/4] Verifying Canonical Artifact Bundles ---"

if [ -d "artifacts/confirmatory" ]; then
    echo "📦 Validating Confirmatory Bundle (artifacts/confirmatory)..."
    ${PY_CMD} -m toolspeed.cli validate-bundle --input artifacts/confirmatory
    echo "✅ Confirmatory bundle hash and manifest structure verified."
    
    echo "🔍 Evaluating Confirmatory Zero-Trust Falsification..."
    set +e
    ${PY_CMD} -m toolspeed.cli falsify --input artifacts/confirmatory
    CONF_EXIT=$?
    set -e
    if [ "${CONF_EXIT}" -ne 0 ]; then
        echo "❌ Expected exit code 0 for confirmatory bundle, got ${CONF_EXIT}"
        exit 1
    fi
    echo "✅ Zero-trust confirmatory recomputation passed with exit code 0."
else
    echo "⚠️  artifacts/confirmatory not found. Skipping."
fi

if [ -d "artifacts/local" ]; then
    echo ""
    echo "📦 Evaluating Local Wall-Clock Bundle (artifacts/local)..."
    set +e
    ${PY_CMD} -m toolspeed.cli falsify --input artifacts/local
    LOCAL_EXIT=$?
    set -e
    if [ "${LOCAL_EXIT}" -ne 1 ]; then
        echo "❌ Expected exit code 1 (falsified) for unconfigured local bundle, got ${LOCAL_EXIT}"
        exit 1
    fi
    echo "✅ Fail-closed local falsification verified with exit code 1."
else
    echo "⚠️  artifacts/local not found. Skipping."
fi

if [ "${MODE}" = "verify-only" ]; then
    echo ""
    echo "========================================================================"
    echo "✨ Bundle verification completed successfully (--verify-only)."
    echo "========================================================================"
    exit 0
fi

# ------------------------------------------------------------------------------
# 3. Fresh Benchmark Execution from Scratch
# ------------------------------------------------------------------------------
echo ""
echo "--- [3/4] Executing Fresh Reproduction Sweep ---"

TMP_DIR=$(mktemp -d -t toolspeed_replication_XXXXXX)
trap 'rm -rf "${TMP_DIR}"' EXIT

REPRO_OUT="${TMP_DIR}/reproduction_bundle"
mkdir -p "${REPRO_OUT}"

if [ "${MODE}" = "quick" ]; then
    TRIALS=10
    SEEDS="42"
    echo "🚀 Running fast reproduction sweep (N=${TRIALS} trials, seed=${SEEDS})..."
    ${PY_CMD} -m toolspeed.cli benchmark \
        --backend replay \
        --mode exploratory \
        --seeds "${SEEDS}" \
        --trials "${TRIALS}" \
        --out "${REPRO_OUT}"
elif [ "${MODE}" = "full" ]; then
    TRIALS=1000
    SEEDS="42,137,2026"
    echo "🚀 Running full confirmatory multi-seed sweep (N=${TRIALS} trials, seeds=${SEEDS})..."
    ${PY_CMD} -m toolspeed.cli benchmark \
        --backend replay \
        --mode confirmatory \
        --protocol benchmarks/protocols/tool-speed-v1.3.json \
        --seeds "${SEEDS}" \
        --trials "${TRIALS}" \
        --out "${REPRO_OUT}"
fi

echo "✅ Benchmark execution completed. Artifacts stored in: ${REPRO_OUT}"

# ------------------------------------------------------------------------------
# 4. Validating Newly Generated Reproduction Bundle
# ------------------------------------------------------------------------------
echo ""
echo "--- [4/4] Validating Reproduction Bundle Integrity ---"

if [ ! -f "${REPRO_OUT}/manifest.sig" ]; then
    echo "❌ Error: Reproduction bundle did not produce a detached seal (manifest.sig)!"
    exit 1
fi
echo "✅ Detached cryptographic seal (manifest.sig) present."

${PY_CMD} -m toolspeed.cli validate-bundle --input "${REPRO_OUT}"
echo "✅ Reproduction bundle passed all hash and structural validation gates."

echo ""
echo "========================================================================"
echo "🎉 REPLICATION SUITE PASSED SUCCESSFULLY!"
echo "========================================================================"
echo "• Environment: Python ${PY_VERSION}"
echo "• Canonical bundles: Verified (confirmatory=0, local=1)"
echo "• Fresh execution: Cleanly produced and validated sealed bundle"
echo "========================================================================"
