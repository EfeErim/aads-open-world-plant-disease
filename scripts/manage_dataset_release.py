#!/usr/bin/env python3
"""Audit, package, publish, fetch, materialize, and verify dataset releases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_release import (  # noqa: E402
    DatasetResourceLimits,
    bind_snapshot_operation,
    build_dataset_snapshot,
    build_runtime_parity_candidate,
    build_shard_plan,
    build_snapshot_reports,
    diff_snapshot_inventory,
    public_contract_summary,
    read_json_dict,
    validate_archive,
    verify_snapshot_files,
    write_json,
)
from src.data.dataset_release_github import (  # noqa: E402
    dry_run_dataset_publish,
    fetch_dataset_release,
    materialize_dataset_release,
    package_dataset_shards,
    preflight_dataset_release,
    promote_dataset_pointer,
    public_github_contract_summary,
    publish_dataset_release,
    upload_dataset_draft,
    verify_dataset_release,
)


def _metadata(path: Path | None) -> dict:
    return read_json_dict(path) if path else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("contract")
    contract.add_argument("--output", type=Path)

    audit = commands.add_parser("audit")
    audit.add_argument("--root", action="append", type=Path, required=True)
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--metadata", type=Path)
    audit.add_argument("--staging-root", type=Path, required=True)
    audit.add_argument("--dataset-version", required=True)
    audit.add_argument("--dataset-tag", required=True)
    audit.add_argument("--inventory-cutoff", required=True)
    audit.add_argument("--snapshot-output", type=Path, required=True)
    audit.add_argument("--audit-output", type=Path, required=True)
    audit.add_argument("--quarantine-output", type=Path, required=True)
    audit.add_argument("--release-output", type=Path, required=True)
    audit.add_argument("--shard-plan-output", type=Path, required=True)

    parity = commands.add_parser("parity-candidate")
    parity.add_argument("--root", type=Path, required=True)
    parity.add_argument("--dataset-version", required=True)
    parity.add_argument("--dataset-tag", required=True)
    parity.add_argument("--inventory-cutoff", required=True)
    parity.add_argument("--release-output", type=Path, required=True)
    parity.add_argument("--shard-plan-output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=Path("."))
    verify.add_argument("--staging-root", type=Path, required=True)

    diff = commands.add_parser("diff")
    diff.add_argument("--snapshot", type=Path, required=True)
    diff.add_argument("--repo-root", type=Path, default=Path("."))

    binding = commands.add_parser("bind-operation")
    binding.add_argument("--snapshot", type=Path, required=True)
    binding.add_argument("--operation", choices=("audit", "diff", "publish", "verify"), required=True)

    archive = commands.add_parser("validate-archive")
    archive.add_argument("--archive", type=Path, required=True)

    package = commands.add_parser("package")
    package.add_argument("--candidate", type=Path, required=True)
    package.add_argument("--shard-plan", type=Path, required=True)
    package.add_argument("--staging-root", type=Path, required=True)
    package.add_argument("--package-root", type=Path, required=True)
    package.add_argument("--repository", required=True)
    package.add_argument("--tag-commit-sha", required=True)
    package.add_argument("--previous-release-tag")
    package.add_argument("--publisher")
    package.add_argument("--manifest-output", type=Path, required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--workspace-root", type=Path, required=True)
    preflight.add_argument("--write", action="store_true")
    preflight.add_argument("--max-upload-bytes", type=int)

    publish = commands.add_parser("publish")
    publish_mode = publish.add_mutually_exclusive_group(required=True)
    publish_mode.add_argument("--dry-run", action="store_true")
    publish_mode.add_argument("--execute", action="store_true")
    publish.add_argument("--candidate", type=Path)
    publish.add_argument("--shard-plan", type=Path)
    publish.add_argument("--repository")
    publish.add_argument("--tag-commit-sha")
    publish.add_argument("--manifest", type=Path)
    publish.add_argument("--approval", type=Path)
    publish.add_argument("--max-upload-bytes", type=int)

    upload = commands.add_parser("upload-draft")
    upload.add_argument("--manifest", type=Path, required=True)
    upload.add_argument("--max-upload-bytes", type=int)

    remote_verify = commands.add_parser("verify-release")
    remote_verify.add_argument("--manifest", type=Path, required=True)
    remote_verify.add_argument("--record", action="store_true")

    fetch = commands.add_parser("fetch")
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--cache-root", type=Path, required=True)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--cache-root", type=Path, required=True)
    materialize.add_argument("--destination", type=Path, required=True)
    materialize.add_argument("--allow-replace", action="store_true")

    promote = commands.add_parser("promote")
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--cache-root", type=Path, required=True)
    promote.add_argument("--materialized-root", type=Path, required=True)
    promote.add_argument("--pointer", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "contract":
        result = public_contract_summary()
        result["github_release_management"] = public_github_contract_summary()
        if args.output:
            write_json(args.output, result)
    elif args.command == "audit":
        snapshot = build_dataset_snapshot(
            args.root,
            repo_root=args.repo_root,
            metadata=_metadata(args.metadata),
            staging_root=args.staging_root,
            dataset_version=args.dataset_version,
            inventory_cutoff=args.inventory_cutoff,
            limits=DatasetResourceLimits(),
        )
        reports = build_snapshot_reports(snapshot, dataset_tag=args.dataset_tag)
        shard_plan = build_shard_plan(reports["release"])
        write_json(args.snapshot_output, snapshot)
        write_json(args.audit_output, reports["audit"])
        write_json(args.quarantine_output, reports["quarantine"])
        write_json(args.release_output, reports["release"])
        write_json(args.shard_plan_output, shard_plan)
        result = {
            "status": "pass",
            "staging_snapshot_id": snapshot["staging_snapshot_id"],
            "manifest_sha256": snapshot["manifest_sha256"],
            "record_count": snapshot["record_count"],
            "uploadable_count": reports["audit"]["uploadable_count"],
            "quarantined_count": reports["audit"]["quarantined_count"],
            "shard_count": len(shard_plan["shards"]),
        }
    elif args.command == "parity-candidate":
        candidate, shard_plan = build_runtime_parity_candidate(
            args.root,
            dataset_version=args.dataset_version,
            dataset_tag=args.dataset_tag,
            inventory_cutoff=args.inventory_cutoff,
        )
        write_json(args.release_output, candidate)
        write_json(args.shard_plan_output, shard_plan)
        result = {
            "status": "pass",
            "staging_snapshot_id": candidate["staging_snapshot_id"],
            "manifest_sha256": candidate["snapshot_manifest_sha256"],
            "record_count": candidate["file_count"],
            "distributed_bytes": candidate["distributed_bytes"],
            "shard_count": len(shard_plan["shards"]),
        }
    elif args.command == "verify":
        result = verify_snapshot_files(
            read_json_dict(args.snapshot), repo_root=args.repo_root.resolve(), staging_root=args.staging_root.resolve()
        )
    elif args.command == "diff":
        result = diff_snapshot_inventory(read_json_dict(args.snapshot), repo_root=args.repo_root.resolve())
    elif args.command == "bind-operation":
        result = bind_snapshot_operation(args.operation, read_json_dict(args.snapshot))
    elif args.command == "validate-archive":
        result = validate_archive(args.archive)
    elif args.command == "package":
        result = package_dataset_shards(
            args.candidate,
            args.shard_plan,
            args.staging_root,
            args.package_root,
            repository=args.repository,
            tag_commit_sha=args.tag_commit_sha,
            previous_release_tag=args.previous_release_tag,
            publisher=args.publisher,
        )
        write_json(args.manifest_output, result)
        result = {
            "status": "pass",
            "release_tag": result["release_tag"],
            "release_manifest_sha256": result["release_manifest_sha256"],
            "file_count": result["file_count"],
            "shard_count": result["shard_count"],
        }
    elif args.command == "preflight":
        result = preflight_dataset_release(
            args.manifest,
            args.workspace_root,
            write=args.write,
            max_upload_bytes=args.max_upload_bytes,
        )
    elif args.command == "publish":
        if args.dry_run:
            if not all((args.candidate, args.shard_plan, args.repository, args.tag_commit_sha)):
                parser.error("publish --dry-run requires --candidate, --shard-plan, --repository, and --tag-commit-sha")
            result = dry_run_dataset_publish(
                args.candidate,
                args.shard_plan,
                repository=args.repository,
                tag_commit_sha=args.tag_commit_sha,
            )
        else:
            if not args.manifest or not args.approval:
                parser.error("publish --execute requires --manifest and --approval")
            result = publish_dataset_release(
                args.manifest,
                args.approval,
                max_upload_bytes=args.max_upload_bytes,
            )
    elif args.command == "upload-draft":
        result = upload_dataset_draft(args.manifest, max_upload_bytes=args.max_upload_bytes)
    elif args.command == "verify-release":
        result = verify_dataset_release(args.manifest, record=args.record)
    elif args.command == "fetch":
        result = fetch_dataset_release(args.manifest, args.cache_root)
    elif args.command == "materialize":
        result = materialize_dataset_release(
            args.manifest, args.cache_root, args.destination, allow_replace=args.allow_replace
        )
    else:
        result = promote_dataset_pointer(args.manifest, args.cache_root, args.materialized_root, args.pointer)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
