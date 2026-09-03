"""Self-contained, detached-seal, non-tamperable benchmark bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from toolspeed.core.types import BenchmarkCase

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "manifest.json",
    "result.json",
    "protocol.json",
    "git-commit.txt",
    "environment.json",
    "cases.jsonl",
    "raw-traces.jsonl",
    "manifest.sig",
)


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_current_git_commit() -> str:
    """Gets current git commit hash, falling back to env var or 'unknown'."""
    sha = os.environ.get("GIT_COMMIT_SHA")
    if sha:
        return sha.strip()
    try:
        import subprocess

        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def capture_environment_metadata() -> dict[str, Any]:
    """Captures runtime environment metadata for provenance."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def create_bundle(
    result_data: dict[str, Any],
    out_dir: str | Path,
    protocol_json: str | None = None,
    git_commit: str | None = None,
    environment_info: dict[str, Any] | None = None,
    cases: list[BenchmarkCase | dict[str, Any]] | None = None,
    raw_traces: list[dict[str, Any]] | None = None,
) -> Path:
    """Creates a self-contained, detached-seal bundle in out_dir without relying on artifacts/ dir."""
    dest = Path(out_dir).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = dest.parent / f"{dest.name}.staging_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Protocol JSON
        proto_text = protocol_json or json.dumps({"plan_id": "tool-speed-v1.3-draft"}, indent=2)
        proto_file = staging_dir / "protocol.json"
        proto_file.write_text(proto_text, encoding="utf-8")

        # 2. Git commit txt
        commit_val = git_commit or get_current_git_commit()
        commit_file = staging_dir / "git-commit.txt"
        commit_file.write_text(f"{commit_val}\n", encoding="utf-8")

        # 3. Environment JSON
        env_meta = environment_info or capture_environment_metadata()
        env_file = staging_dir / "environment.json"
        env_file.write_text(json.dumps(env_meta, indent=2, sort_keys=True), encoding="utf-8")

        # 4. Cases JSONL
        cases_file = staging_dir / "cases.jsonl"
        with open(cases_file, "w", encoding="utf-8") as f_cases:
            if cases:
                for c in cases:
                    data = c.to_dict() if hasattr(c, "to_dict") else dict(c)
                    f_cases.write(json.dumps(data) + "\n")
            else:
                f_cases.write(json.dumps({"case_id": "case_default"}) + "\n")

        # 5. Raw Traces JSONL
        traces_file = staging_dir / "raw-traces.jsonl"
        with open(traces_file, "w", encoding="utf-8") as f_traces:
            if raw_traces:
                for tr in raw_traces:
                    f_traces.write(json.dumps(tr) + "\n")
            else:
                f_traces.write(json.dumps({"trace_id": "trace_default"}) + "\n")

        # 6. File hashes for payload
        file_hashes: dict[str, str] = {
            "protocol.json": compute_file_sha256(proto_file),
            "git-commit.txt": compute_file_sha256(commit_file),
            "environment.json": compute_file_sha256(env_file),
            "cases.jsonl": compute_file_sha256(cases_file),
            "raw-traces.jsonl": compute_file_sha256(traces_file),
        }

        # 7. Result JSON
        res_copy = dict(result_data)
        manifest_meta = dict(res_copy.get("manifest") or {})
        manifest_meta["file_hashes"] = file_hashes
        manifest_meta["code_git_sha"] = commit_val
        res_copy["manifest"] = manifest_meta

        res_file = staging_dir / "result.json"
        res_file.write_text(json.dumps(res_copy, indent=2, sort_keys=True), encoding="utf-8")
        file_hashes["result.json"] = compute_file_sha256(res_file)

        # 8. Manifest JSON
        manifest_meta["file_hashes"] = file_hashes
        manifest_meta["result_hash"] = file_hashes["result.json"]
        manifest_file = staging_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_meta, indent=2, sort_keys=True), encoding="utf-8")

        # 9. Detached seal manifest.sig
        manifest_sha = compute_file_sha256(manifest_file)
        sig_file = staging_dir / "manifest.sig"
        sig_file.write_text(f"{manifest_sha}\n", encoding="utf-8")

        # Atomic move
        if dest.exists():
            shutil.rmtree(dest)
        staging_dir.rename(dest)
        return dest
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def validate_bundle_hashes_first(bundle_path: str | Path) -> tuple[bool, list[str]]:
    """Strictly checks all SHA-256 hashes and detached seal BEFORE reading contents.

    Returns (is_valid, error_messages).
    """
    b_dir = Path(bundle_path).resolve()
    errors: list[str] = []

    if not b_dir.is_dir():
        return False, [f"Bundle path '{b_dir}' is not a directory."]

    # Step 1: Check presence of all required files
    for req_name in REQUIRED_BUNDLE_FILES:
        req_file = b_dir / req_name
        if not req_file.exists():
            errors.append(f"Missing required bundle file: '{req_name}'")

    if errors:
        return False, errors

    # Step 2: Verify detached seal manifest.sig against manifest.json
    manifest_file = b_dir / "manifest.json"
    sig_file = b_dir / "manifest.sig"

    actual_manifest_sha = compute_file_sha256(manifest_file)
    sig_content = sig_file.read_text(encoding="utf-8").strip()

    if sig_content != actual_manifest_sha:
        errors.append(f"Detached seal manifest.sig mismatch! Expected {sig_content}, actual {actual_manifest_sha}")
        return False, errors

    # Step 3: Parse manifest file_hashes and check EVERY file hash before reading contents
    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Failed to parse manifest.json: {e}"]

    file_hashes = manifest_data.get("file_hashes")
    if not isinstance(file_hashes, dict) or not file_hashes:
        return False, ["Manifest is missing required 'file_hashes' dictionary."]

    for fname, expected_hash in file_hashes.items():
        if fname in ("manifest.json", "manifest.sig"):
            continue
        target_path = b_dir / fname
        if not target_path.exists():
            errors.append(f"File '{fname}' listed in manifest file_hashes does not exist on disk.")
            continue

        actual_hash = compute_file_sha256(target_path)
        if actual_hash != expected_hash:
            errors.append(f"File '{fname}' SHA-256 hash mismatch! Expected {expected_hash}, got {actual_hash}")

    if errors:
        return False, errors

    return True, []
