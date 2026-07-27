from pathlib import Path

import scripts.build_open_world_router_manifest as builder
from scripts.enrich_m2_demo_image_set import AcceptedImage, Candidate


def _accepted(path: Path, *, sha256: str, photo_id: str) -> AcceptedImage:
    path.write_bytes(b"image")
    return AcceptedImage(
        path=path,
        sha256=sha256,
        width=400,
        height=300,
        contrast=22.0,
        candidate=Candidate(
            source_url=f"http://example.test/{photo_id}.jpg",
            observation_url=f"http://example.test/observations/{photo_id}",
            photo_id=photo_id,
            query="test query",
            taxon_name="Test taxon",
            license_code="cc-by",
            attribution="tester",
        ),
    )


def test_build_open_world_manifest_writes_required_fields_and_summary(tmp_path: Path, monkeypatch):
    plans = (
        builder.OpenWorldPlan(
            "unsupported_crop",
            "unknown_crop",
            "unknown",
            "unknown",
            "Malus domestica",
            ("apple leaf",),
            2,
            "unsupported apple",
        ),
        builder.OpenWorldPlan(
            "same_crop_wrong_part",
            "tomato__unknown_part",
            "tomato",
            "unknown",
            "Solanum lycopersicum",
            ("tomato flower",),
            1,
            "tomato wrong part",
        ),
    )
    monkeypatch.setattr(builder, "OPEN_WORLD_PLAN", plans)

    def fake_collect(plan, *, staging_dir, used_hashes, **_kwargs):
        staging_dir.mkdir(parents=True, exist_ok=True)
        accepted = []
        for index in range(plan.count):
            digest = f"{plan.expected_target}_{index}"
            assert digest not in used_hashes
            used_hashes.add(digest)
            accepted.append(_accepted(staging_dir / f"{digest}.jpg", sha256=digest, photo_id=digest))
        return accepted

    monkeypatch.setattr(builder, "_collect_external_images", fake_collect)

    manifest = tmp_path / "manifests" / "m2_open_world_router_manifest.csv"
    summary = builder.build_open_world_manifest(
        output_manifest=manifest,
        output_image_dir=tmp_path / "images",
        staging_dir=tmp_path / "staging",
        disjoint_roots=[],
        reject_ids_path=None,
        start_id=1,
        per_page=10,
        max_pages=1,
        timeout=5,
        sleep_seconds=0.0,
        min_side=1,
        min_bytes=1,
        min_contrast=0.0,
        max_aspect_ratio=10.0,
        min_rows=3,
    )

    text = manifest.read_text(encoding="utf-8")
    assert "ood_slice" in text
    assert "provenance_notes" in text
    assert "ow_0001" in text
    assert summary["row_count"] == 3
    assert summary["slice_counts"] == {"same_crop_wrong_part": 1, "unsupported_crop": 2}
    assert summary["duplicate_hash_count"] == 0


def test_build_open_world_manifest_fails_when_total_rows_below_minimum(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        builder,
        "OPEN_WORLD_PLAN",
        (
            builder.OpenWorldPlan(
                "unsupported_crop",
                "unknown_crop",
                "unknown",
                "unknown",
                "Malus domestica",
                ("apple leaf",),
                2,
                "unsupported apple",
            ),
        ),
    )
    monkeypatch.setattr(builder, "_collect_external_images", lambda *_args, **_kwargs: [])

    try:
        builder.build_open_world_manifest(
            output_manifest=tmp_path / "manifest.csv",
            output_image_dir=tmp_path / "images",
            staging_dir=tmp_path / "staging",
            disjoint_roots=[],
            reject_ids_path=None,
            start_id=1,
            per_page=10,
            max_pages=1,
            timeout=5,
            sleep_seconds=0.0,
            min_side=1,
            min_bytes=1,
            min_contrast=0.0,
            max_aspect_ratio=10.0,
            min_rows=1,
        )
    except RuntimeError as exc:
        assert "open-world manifest selected 0/1 required rows" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_build_open_world_manifest_resumes_existing_manifest_and_stops_at_minimum(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifests" / "m2_open_world_router_manifest.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    existing_image = image_dir / "ow_0001_existing.jpg"
    existing_image.write_bytes(b"existing")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "\n".join(
            [
                (
                    "image_id,source,expected_target,expected_crop,expected_part,expected_class,"
                    "expected_behavior,ood_slice,origin_url,notes,provenance_notes,resolved_source_path,"
                    "photo_url,taxon_name,query,photo_id,license_code,attribution,sha256,width,height,"
                    "contrast,source_backend"
                ),
                (
                    "ow_0001,staged_external:images/ow_0001_existing.jpg,unknown_crop,unknown,unknown,,"
                    "open-world negative; abstain or review expected,unsupported_crop,http://example.test/o1,"
                    "existing,source=iNaturalist; photo_id=existing,images/ow_0001_existing.jpg,"
                    "http://example.test/p1,Test,test,existing,cc-by,tester,existing-sha,400,300,22.00,"
                    "inaturalist"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder,
        "OPEN_WORLD_PLAN",
        (
            builder.OpenWorldPlan(
                "unsupported_crop",
                "unknown_crop",
                "unknown",
                "unknown",
                "Malus domestica",
                ("apple leaf",),
                2,
                "unsupported apple",
            ),
        ),
    )

    def fake_collect(plan, *, staging_dir, used_hashes, **_kwargs):
        assert "existing-sha" in used_hashes
        staging_dir.mkdir(parents=True, exist_ok=True)
        return [_accepted(staging_dir / "new.jpg", sha256="new-sha", photo_id="new-photo")]

    monkeypatch.setattr(builder, "_collect_external_images", fake_collect)

    summary = builder.build_open_world_manifest(
        output_manifest=manifest,
        output_image_dir=image_dir,
        staging_dir=tmp_path / "staging",
        disjoint_roots=[],
        reject_ids_path=None,
        start_id=1,
        per_page=10,
        max_pages=1,
        timeout=5,
        sleep_seconds=0.0,
        min_side=1,
        min_bytes=1,
        min_contrast=0.0,
        max_aspect_ratio=10.0,
        min_rows=2,
    )

    text = manifest.read_text(encoding="utf-8")
    assert "ow_0001" in text
    assert "ow_0002" in text
    assert summary["row_count"] == 2
