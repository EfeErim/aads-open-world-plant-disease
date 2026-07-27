#!/usr/bin/env python3
"""Build, verify, fetch, and publish the immutable demo adapter release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.adapter_release import (  # noqa: E402
    EXPECTED_BUNDLE_FILES,
    RELEASE_SCHEMA,
    create_adapter_draft,
    fetch_adapter_release,
    preflight_adapter_release,
    promote_adapter_pointer,
    publish_adapter_release,
    record_draft_release,
    sha256_file,
    validate_promotion_approval,
    validate_release_manifest,
    verify_adapter_draft,
    verify_release_files,
    write_release_receipt,
)
from src.shared.json_utils import read_json_dict, write_json  # noqa: E402


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _repo_path_sha256(relative_path: str) -> str:
    path = ROOT / relative_path
    if path.is_file():
        return sha256_file(path)
    content = subprocess.check_output(["git", "show", f"HEAD:{relative_path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def build_manifest(spec_path: Path, output_path: Path) -> dict[str, Any]:
    spec = read_json_dict(spec_path)
    release_id = str(spec["release_id"])
    files: list[dict[str, Any]] = []
    targets: list[str] = []
    for adapter in spec["adapters"]:
        target_id = str(adapter["target_id"])
        crop, part = target_id.split("__", 1)
        targets.append(target_id)
        bundle = ROOT / str(adapter["source_bundle"])
        actual_names = {path.name for path in bundle.iterdir() if path.is_file()}
        expected_source = EXPECTED_BUNDLE_FILES - {"DINOV3_LICENSE.md", "production_readiness.json"}
        if actual_names != expected_source:
            raise ValueError(
                f"Source allowlist mismatch for {target_id}; missing={sorted(expected_source-actual_names)}, "
                f"unexpected={sorted(actual_names-expected_source)}."
            )
        source_files = sorted(bundle.iterdir()) + [
            ROOT / str(adapter["readiness_file"]),
            ROOT / "docs/evidence/current/demo_release/DINOV3_LICENSE.md",
        ]
        for source in source_files:
            name = source.name
            local_path = Path(crop) / part / "continual_sd_lora_adapter" / name
            files.append(
                {
                    "target_id": target_id,
                    "source_path": source.relative_to(ROOT).as_posix(),
                    "local_path": local_path.as_posix(),
                    "asset_name": f"{target_id}--{name}",
                    "asset_id": None,
                    "github_digest": None,
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
    evidence = []
    for value in spec["evidence_paths"]:
        evidence.append({"path": str(value), "sha256": _repo_path_sha256(str(value))})
    payload: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": release_id,
        "release_label": "demo-ready",
        "production_ready": False,
        "created_at": str(spec.get("created_at") or datetime.now(timezone.utc).isoformat()),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "targets": targets,
        "github_release": {
            "repository": spec.get("github_repository") or os.environ.get("GITHUB_RELEASE_REPOSITORY"),
            "tag": spec["release_tag"],
            "name": spec["release_name"],
            "body": spec.get("release_body"),
            "release_id": None,
            "tag_commit_sha": _git_output("rev-parse", "HEAD"),
            "draft": True,
            "immutable": False,
            "html_url": None,
        },
        "files": files,
        "accepted_evidence": evidence,
        "taxonomy_path": spec["taxonomy_path"],
        "prototype_bank_path": spec["prototype_bank_path"],
        "calibration_path": spec["calibration_path"],
        "rollback": spec["rollback"],
        "environment": {
            "python_target": "3.11",
            "generator_python": platform.python_version(),
            "platform": platform.platform(),
            "requirements_sha256": sha256_file(ROOT / "requirements.txt"),
            "requirements_colab_sha256": sha256_file(ROOT / "requirements_colab.txt"),
        },
        "license_review": spec["license_review"],
    }
    validate_release_manifest(payload)
    write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--root", type=Path)
    stage = subparsers.add_parser("stage-local")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--root", type=Path, required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--cache-root", type=Path, default=Path("models/adapters"))
    preflight.add_argument("--write", action="store_true")
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--cache-root", type=Path, default=Path("models/adapters"))
    fetch.add_argument("--receipt", type=Path)
    verify_draft = subparsers.add_parser("verify-draft")
    verify_draft.add_argument("--manifest", type=Path, required=True)
    draft = subparsers.add_parser("create-draft")
    draft.add_argument("--manifest", type=Path, required=True)
    draft.add_argument("--receipt", type=Path, required=True)
    record = subparsers.add_parser("record-draft")
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--receipt", type=Path, required=True)
    publish = subparsers.add_parser("publish-immutable")
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument("--approval", type=Path, required=True)
    publish.add_argument("--receipt", type=Path)
    approval = subparsers.add_parser("verify-approval")
    approval.add_argument("--manifest", type=Path, required=True)
    approval.add_argument("--approval", type=Path, required=True)
    promote = subparsers.add_parser("promote-local")
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--approval", type=Path, required=True)
    promote.add_argument("--cache-root", type=Path, default=Path("models/adapters"))
    promote.add_argument("--config", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.command == "build":
        result = build_manifest(args.spec, args.output)
        print(json.dumps({"status": "pass", "files": len(result["files"]), "output": str(args.output)}))
    elif args.command == "verify":
        manifest = read_json_dict(args.manifest)
        validate_release_manifest(manifest)
        if args.root:
            verify_release_files(args.root, manifest)
        else:
            for record in manifest["files"]:
                source = ROOT / str(record.get("source_path") or "")
                if not source.is_file() or sha256_file(source) != record["sha256"]:
                    raise ValueError(f"Source checksum mismatch: {source}")
        print(json.dumps({"status": "pass", "files": len(manifest["files"])}))
    elif args.command == "stage-local":
        manifest = read_json_dict(args.manifest)
        validate_release_manifest(manifest)
        if args.root.exists():
            shutil.rmtree(args.root)
        for record in manifest["files"]:
            source = ROOT / str(record["source_path"])
            destination = args.root / str(record["local_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        verify_release_files(args.root, manifest)
        print(json.dumps({"status": "pass", "files": len(manifest["files"]), "root": str(args.root)}))
    elif args.command == "preflight":
        print(json.dumps(preflight_adapter_release(args.manifest, args.cache_root, write=args.write)))
    elif args.command == "verify-draft":
        print(json.dumps(verify_adapter_draft(args.manifest)))
    elif args.command == "fetch":
        result = fetch_adapter_release(args.manifest, args.cache_root)
        if args.receipt:
            write_release_receipt(args.receipt, result)
        print(json.dumps(result))
    elif args.command == "create-draft":
        result = create_adapter_draft(args.manifest)
        write_release_receipt(args.receipt, result)
        print(json.dumps(result))
    elif args.command == "record-draft":
        print(json.dumps(record_draft_release(args.manifest, args.receipt)))
    elif args.command == "publish-immutable":
        result = publish_adapter_release(args.manifest, args.approval)
        if args.receipt:
            write_release_receipt(args.receipt, result)
        print(json.dumps(result))
    elif args.command == "verify-approval":
        result = validate_promotion_approval(args.approval, args.manifest)
        print(json.dumps({"status": "pass", "approver_identity": result["approver_identity"]}))
    else:
        config_paths = args.config or [Path("config/base.json"), Path("config/colab.json")]
        print(json.dumps(promote_adapter_pointer(args.manifest, args.approval, args.cache_root, config_paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
