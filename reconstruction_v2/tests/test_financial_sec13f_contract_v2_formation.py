from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from replication_runtime.financial_semantic_v2 import pack as period_pack
from replication_runtime.financial_sec13f_contract_v2 import formation


ROOT = Path(__file__).resolve().parents[1]
MANAGER_COUNT = 32
ISSUER_COUNT = 20


def _write_tsv(
    path: Path, header: list[str], rows: list[list[object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_period(
    root: Path,
    *,
    report_date: str,
    accession_prefix: str,
    entity_prefix: str,
    phase: int,
) -> None:
    cover_rows: list[list[object]] = []
    info_rows: list[list[object]] = []
    for manager_index in range(MANAGER_COUNT):
        accession = f"{accession_prefix}{manager_index:05d}"
        manager = f"{entity_prefix} Period Fund {manager_index:02d} LLC"
        cover_rows.append(
            [accession, report_date, "13F HOLDINGS REPORT", manager]
        )
        for issuer_index in range(ISSUER_COUNT):
            cusip = f"{100_000_000 + issuer_index:09d}"
            base = 10_000 + manager_index * 100 + issuer_index
            value = (
                base
                + phase * (issuer_index + 1) * 1_000
                + phase * manager_index
            )
            info_rows.append(
                [
                    accession,
                    f"{entity_prefix} Issuer Corporation {issuer_index:02d}",
                    "COM",
                    cusip,
                    value,
                ]
            )
        info_rows.append(
            [
                accession,
                f"{entity_prefix} Private Note {manager_index:02d}",
                "PUT",
                f"{900_000_000 + manager_index:09d}",
                777 + manager_index,
            ]
        )
    cover_rows.extend(
        [
            ["OLD000", "30-JUN-2024", "13F HOLDINGS REPORT", "Old Fund"],
            ["NOTICE0", report_date, "13F NOTICE", "Notice Fund"],
        ]
    )
    info_rows.extend(
        [
            ["OLD000", "Ignored Old Issuer", "COM", "800000001", 999],
            ["NOTICE0", "Ignored Notice Issuer", "COM", "800000002", 999],
        ]
    )
    if phase % 2:
        cover_rows.reverse()
        info_rows.reverse()
    _write_tsv(
        root / "COVERPAGE.tsv",
        [
            "ACCESSION_NUMBER",
            "REPORTCALENDARORQUARTER",
            "REPORTTYPE",
            "FILINGMANAGER_NAME",
        ],
        cover_rows,
    )
    _write_tsv(
        root / "INFOTABLE.tsv",
        [
            "ACCESSION_NUMBER",
            "NAMEOFISSUER",
            "TITLEOFCLASS",
            "CUSIP",
            "VALUE",
        ],
        info_rows,
    )


def _zip_period(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(source.iterdir()):
            archive.write(path, arcname=f"official-sec-period/{path.name}")


@pytest.fixture()
def synthetic_periods(tmp_path: Path) -> dict[str, Path]:
    descriptions = {
        "old_previous": ("30-SEP-2025", "OP", "Legacy", 0),
        "old_current": ("31-DEC-2025", "OC", "Legacy", 1),
        "new_previous": ("31-DEC-2025", "NP", "Fresh", 0),
        "new_current": ("31-MAR-2026", "NC", "Fresh", 1),
    }
    result: dict[str, Path] = {}
    for name, (date, accession, entity, phase) in descriptions.items():
        directory = tmp_path / name
        _write_period(
            directory,
            report_date=date,
            accession_prefix=accession,
            entity_prefix=entity,
            phase=phase,
        )
        archive = tmp_path / f"{name}.zip"
        _zip_period(directory, archive)
        result[name] = archive
    return result


def _rehash(payload: dict, field: str) -> None:
    body = dict(payload)
    body.pop(field, None)
    payload[field] = period_pack.payload_hash(body)


def _old_view(sources: dict[str, Path]) -> dict:
    old_pack = period_pack.build_public_pack(
        previous_source=sources["old_previous"],
        current_source=sources["old_current"],
        previous_period_label="2025 Q3",
        current_period_label="2025 Q4",
        preregistration_seed="synthetic-old-period-seed",
        previous_container_root="/root/old-previous",
        current_container_root="/root/old-current",
    )
    return period_pack.build_measurement_view(old_pack)


def _valid_structural_acquisition(preregistration: dict) -> dict:
    inherited = preregistration["inherited_previous"]["archive_receipt"]
    previous = {
        "role": "previous",
        **{
            key: inherited[key]
            for key in (
                "source_url",
                "archive_sha256",
                "size_bytes",
                "coverpage_sha256",
                "infotable_sha256",
                "source_fingerprint",
                "source_path_persisted",
            )
        },
    }
    cover = "a" * 64
    info = "b" * 64
    current = {
        "role": "current",
        "source_url": formation.CURRENT_URL,
        "archive_sha256": "c" * 64,
        "size_bytes": 123,
        "coverpage_sha256": cover,
        "infotable_sha256": info,
        "source_fingerprint": period_pack.payload_hash(
            {
                "source_policy": (
                    "official_sec_form_13f_quarterly_flattened_v1"
                ),
                "coverpage_sha256": cover,
                "infotable_sha256": info,
            }
        ),
        "source_path_persisted": False,
        "calendar_window": formation.CURRENT_CALENDAR_WINDOW,
        "observed_last_modified": None,
    }
    rows = [previous, current]
    body = {
        "receipt_version": formation.ACQUISITION_RECEIPT_VERSION,
        "study_id": formation.STUDY_ID,
        "preregistration": {
            "relative_path": "manifests/fresh-preregistration.json",
            "file_sha256": "d" * 64,
            "manifest_hash": preregistration["manifest_hash"],
            "committed_at_git_commit": formation.CANDIDATE_COMMIT,
        },
        "acquisition_order": {
            "policy": "current_archive_ctime_not_before_preregistration_v1",
            "preregistration_file_ctime_ns": 10,
            "current_archive_file_ctime_ns": 11,
            "current_archive_not_older_than_preregistration": True,
            "previous_archive_ctime_observed": False,
            "previous_archive_ctime_constrained": False,
        },
        "archives": rows,
        "archive_set_hash": period_pack.payload_hash(rows),
        "previous_inherited_from_receipt_hash": (
            formation.PRIOR_ACQUISITION_RECEIPT_HASH
        ),
        "resampling_used": False,
        "model_calls": 0,
        "online_judge_calls": 0,
        "secret_value_persisted": False,
    }
    return {**body, "receipt_hash": period_pack.payload_hash(body)}


def test_preregistration_inherits_prior_current_and_binds_only_new_url() -> None:
    preregistration = formation.build_preregistration_v1(ROOT)
    assert (
        formation.validate_preregistration_v1(
            preregistration, project_root=ROOT
        )
        == preregistration["manifest_hash"]
    )
    assert preregistration == formation.build_preregistration_v1(ROOT)
    assert preregistration["candidate_freeze"]["commit"] == (
        formation.CANDIDATE_COMMIT
    )
    closure = preregistration["formation_source_closure"]
    assert [row["relative_path"] for row in closure["files"]] == list(
        formation._FORMATION_SOURCE_PATHS
    )
    assert closure["file_count"] == len(formation._FORMATION_SOURCE_PATHS)
    assert closure["file_set_hash"] == period_pack.payload_hash(
        closure["files"]
    )
    inherited = preregistration["inherited_previous"]
    assert inherited["prior_acquisition_receipt_hash"] == (
        formation.PRIOR_ACQUISITION_RECEIPT_HASH
    )
    assert inherited["prior_archive_role"] == "current"
    assert inherited["live_archive_hash_required_at_acquisition"] is True
    assert inherited["archive_ctime_constrained"] is False
    current = preregistration["period_data"]["current_archive"]
    assert current["url"] == formation.CURRENT_URL
    assert current["download_authorized_after_preregistration_only"] is True
    assert current["content_length_bound_at_preregistration"] is False
    assert current["last_modified_bound_at_preregistration"] is False
    assert preregistration["period_data"]["acquisition_order"] == {
        "policy": "current_archive_ctime_not_before_preregistration_v1",
        "previous_archive_ctime_constrained": False,
        "current_archive_ctime_constrained": True,
    }
    assert preregistration["prior_commitment_view"][
        "prior_private_pack_accessed"
    ] is False
    assert preregistration["prior_commitment_view"][
        "prior_sealed_content_accessed"
    ] is False
    assert preregistration["prior_commitment_view"][
        "authoritative_git_commit"
    ] == formation.PRIOR_MEASUREMENT_VIEW_COMMIT
    assert preregistration["prior_commitment_view"][
        "authoritative_blob_sha256"
    ] == formation.PRIOR_MEASUREMENT_VIEW_FILE_SHA256
    assert preregistration["oracles"]["oracle_ids"] == [
        "sec13f_pandas_chunked_v1",
        "sec13f_stdlib_streaming_v1",
    ]
    assert preregistration["oracles"]["calls_during_pack_formation"] == 0
    assert preregistration["evidence_boundary"]["gold_formed"] is False


def test_preregistration_fails_closed_on_url_commit_and_live_hash_drift() -> None:
    preregistration = formation.build_preregistration_v1(ROOT)

    changed_url = copy.deepcopy(preregistration)
    changed_url["period_data"]["current_archive"]["url"] += "?retry=1"
    _rehash(changed_url, "manifest_hash")
    with pytest.raises(formation.FreshFormationError, match="drifted"):
        formation.validate_preregistration_v1(changed_url)

    changed_commit = copy.deepcopy(preregistration)
    changed_commit["candidate_freeze"]["commit"] = "e" * 40
    _rehash(changed_commit, "manifest_hash")
    with pytest.raises(formation.FreshFormationError, match="drifted"):
        formation.validate_preregistration_v1(changed_commit)

    changed_live_receipt = copy.deepcopy(preregistration)
    changed_live_receipt["inherited_previous"][
        "prior_acquisition_file_sha256"
    ] = "f" * 64
    _rehash(changed_live_receipt, "manifest_hash")
    # The shape is valid, but live validation must bind the inherited receipt.
    formation.validate_preregistration_v1(changed_live_receipt)
    with pytest.raises(formation.FreshFormationError, match="live frozen"):
        formation.validate_preregistration_v1(
            changed_live_receipt, project_root=ROOT
        )

    changed_closure = copy.deepcopy(preregistration)
    changed_closure["formation_source_closure"]["files"][0][
        "file_sha256"
    ] = "f" * 64
    changed_closure["formation_source_closure"][
        "file_set_hash"
    ] = period_pack.payload_hash(
        changed_closure["formation_source_closure"]["files"]
    )
    closure_body = dict(changed_closure["formation_source_closure"])
    closure_body.pop("closure_hash")
    changed_closure["formation_source_closure"][
        "closure_hash"
    ] = period_pack.payload_hash(closure_body)
    _rehash(changed_closure, "manifest_hash")
    formation.validate_preregistration_v1(changed_closure)
    with pytest.raises(formation.FreshFormationError, match="live frozen"):
        formation.validate_preregistration_v1(
            changed_closure, project_root=ROOT
        )


def test_acquisition_schema_constrains_current_ctime_not_previous() -> None:
    preregistration = formation.build_preregistration_v1(ROOT)
    receipt = _valid_structural_acquisition(preregistration)
    assert (
        formation.validate_acquisition_receipt_v1(
            receipt, preregistration=preregistration
        )
        == receipt["receipt_hash"]
    )
    order = receipt["acquisition_order"]
    assert "previous_archive_file_ctime_ns" not in order
    assert order["previous_archive_ctime_observed"] is False
    assert order["previous_archive_ctime_constrained"] is False

    older_current = copy.deepcopy(receipt)
    older_current["acquisition_order"]["current_archive_file_ctime_ns"] = 9
    _rehash(older_current, "receipt_hash")
    with pytest.raises(formation.FreshFormationError, match="drifted"):
        formation.validate_acquisition_receipt_v1(
            older_current, preregistration=preregistration
        )

    smuggled_previous_ctime = copy.deepcopy(receipt)
    smuggled_previous_ctime["acquisition_order"][
        "previous_archive_file_ctime_ns"
    ] = 0
    _rehash(smuggled_previous_ctime, "receipt_hash")
    with pytest.raises(formation.FreshFormationError, match="drifted"):
        formation.validate_acquisition_receipt_v1(
            smuggled_previous_ctime, preregistration=preregistration
        )

    changed_inherited_bytes = copy.deepcopy(receipt)
    changed_inherited_bytes["archives"][0]["archive_sha256"] = "0" * 64
    changed_inherited_bytes["archive_set_hash"] = period_pack.payload_hash(
        changed_inherited_bytes["archives"]
    )
    _rehash(changed_inherited_bytes, "receipt_hash")
    with pytest.raises(formation.FreshFormationError, match="drifted"):
        formation.validate_acquisition_receipt_v1(
            changed_inherited_bytes, preregistration=preregistration
        )


def test_live_acquisition_binds_both_claimed_ctimes_to_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preregistration = formation.build_preregistration_v1(ROOT)
    receipt = _valid_structural_acquisition(preregistration)
    project = tmp_path / "project"
    project.mkdir()
    preregistration_path = project / "fresh-preregistration.json"
    formation.write_json(preregistration_path, preregistration)
    receipt["preregistration"]["relative_path"] = (
        preregistration_path.relative_to(project).as_posix()
    )

    monkeypatch.setattr(
        formation,
        "validate_preregistration_v1",
        lambda value, project_root=None: preregistration["manifest_hash"],
    )
    monkeypatch.setattr(
        formation,
        "_committed_preregistration",
        lambda project, path, relative: formation.CANDIDATE_COMMIT,
    )
    original_sha256_file = formation.sha256_file

    def _prereg_hash(path: str | Path) -> str:
        if Path(path).resolve() == preregistration_path.resolve():
            return receipt["preregistration"]["file_sha256"]
        return original_sha256_file(path)

    monkeypatch.setattr(formation, "sha256_file", _prereg_hash)
    _rehash(receipt, "receipt_hash")
    with pytest.raises(formation.FreshFormationError, match="differs"):
        formation.validate_acquisition_receipt_v1(
            receipt,
            preregistration=preregistration,
            project_root=project,
            preregistration_path=preregistration_path,
        )

    previous_path = tmp_path / "previous.zip"
    current_path = tmp_path / "current.zip"
    with previous_path.open("wb") as handle:
        handle.truncate(receipt["archives"][0]["size_bytes"])
    with current_path.open("wb") as handle:
        handle.truncate(receipt["archives"][1]["size_bytes"])
    preregistration_ctime = preregistration_path.stat().st_ctime_ns
    current_ctime = current_path.stat().st_ctime_ns
    receipt["acquisition_order"][
        "preregistration_file_ctime_ns"
    ] = preregistration_ctime
    receipt["acquisition_order"][
        "current_archive_file_ctime_ns"
    ] = max(preregistration_ctime, current_ctime) + 1

    archive_by_path = {
        previous_path.resolve(): receipt["archives"][0],
        current_path.resolve(): receipt["archives"][1],
    }

    def _bound_hash(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == preregistration_path.resolve():
            return receipt["preregistration"]["file_sha256"]
        return archive_by_path[resolved]["archive_sha256"]

    def _open_source(path: str | Path) -> SimpleNamespace:
        row = archive_by_path[Path(path).resolve()]
        return SimpleNamespace(
            coverpage_sha256=row["coverpage_sha256"],
            infotable_sha256=row["infotable_sha256"],
            source_fingerprint=row["source_fingerprint"],
        )

    monkeypatch.setattr(formation, "sha256_file", _bound_hash)
    monkeypatch.setattr(formation.Sec13FSource, "open", _open_source)
    _rehash(receipt, "receipt_hash")
    with pytest.raises(
        formation.FreshFormationError,
        match="violates preregistration order",
    ):
        formation.validate_acquisition_receipt_v1(
            receipt,
            preregistration=preregistration,
            project_root=project,
            preregistration_path=preregistration_path,
            previous_archive=previous_path,
            current_archive=current_path,
        )


def test_collision_checked_pack_reuses_old_builder_without_gold_or_oracles(
    synthetic_periods: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prior_view = _old_view(synthetic_periods)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("pack formation must not call an oracle")

    monkeypatch.setattr(formation.oracle_pandas, "evaluate_partition", _forbidden)
    monkeypatch.setattr(
        formation.oracle_streaming, "evaluate_partition", _forbidden
    )
    artifacts = formation.build_collision_checked_pack_v1(
        previous_source=synthetic_periods["new_previous"],
        current_source=synthetic_periods["new_current"],
        prior_measurement_view=prior_view,
    )
    assert period_pack.verify_public_pack(artifacts.private_pack) == (
        artifacts.private_pack
    )
    assert period_pack.verify_measurement_view(
        artifacts.measurement_view,
        private_pack=artifacts.private_pack,
    ) == artifacts.measurement_view
    assert artifacts.private_pack["snapshot_report_dates"] == {
        "previous": "2025-12-31",
        "current": "2026-03-31",
    }
    receipt = artifacts.formation_receipt
    assert receipt["frozen_pack_implementation"] == (
        "replication_runtime.financial_semantic_v2.pack"
    )
    assert receipt["frozen_oracle_ids_reserved_for_later_measurement_gold"] == [
        "sec13f_pandas_chunked_v1",
        "sec13f_stdlib_streaming_v1",
    ]
    assert receipt["oracle_calls"] == 0
    assert receipt["gold_formed"] is False
    assert receipt["prior_private_pack_accessed"] is False
    assert receipt["prior_sealed_content_accessed"] is False
    assert receipt["preregistration_hash"] is None
    assert receipt["acquisition_receipt_hash"] is None
    assert receipt["input_binding_complete"] is False
    assert "answers" not in str(receipt)
    body = dict(receipt)
    declared = body.pop("receipt_hash")
    assert declared == period_pack.payload_hash(body)


def test_production_formation_receipt_has_one_exact_input_bound_schema(
    synthetic_periods: dict[str, Path]
) -> None:
    prior_view = _old_view(synthetic_periods)
    artifacts = formation.build_collision_checked_pack_v1(
        previous_source=synthetic_periods["new_previous"],
        current_source=synthetic_periods["new_current"],
        prior_measurement_view=prior_view,
    )
    receipt_body = dict(artifacts.formation_receipt)
    receipt_body.pop("receipt_hash")
    preregistration_hash = "a" * 64
    acquisition_hash = "b" * 64
    receipt_body.update(
        {
            "preregistration_hash": preregistration_hash,
            "acquisition_receipt_hash": acquisition_hash,
            "input_binding_complete": True,
        }
    )
    receipt = {
        **receipt_body,
        "receipt_hash": period_pack.payload_hash(receipt_body),
    }
    assert formation.validate_pack_formation_receipt_v1(
        receipt,
        private_pack=artifacts.private_pack,
        measurement_view=artifacts.measurement_view,
        prior_measurement_view=prior_view,
        preregistration_hash=preregistration_hash,
        acquisition_receipt_hash=acquisition_hash,
    ) == receipt["receipt_hash"]

    extra = copy.deepcopy(receipt)
    extra["debug"] = "forbidden"
    _rehash(extra, "receipt_hash")
    with pytest.raises(formation.FreshFormationError, match="fields drifted"):
        formation.validate_pack_formation_receipt_v1(
            extra,
            private_pack=artifacts.private_pack,
            measurement_view=artifacts.measurement_view,
            prior_measurement_view=prior_view,
            preregistration_hash=preregistration_hash,
            acquisition_receipt_hash=acquisition_hash,
        )

    wrong_pack = copy.deepcopy(receipt)
    wrong_pack["private_pack_hash"] = "c" * 64
    _rehash(wrong_pack, "receipt_hash")
    with pytest.raises(formation.FreshFormationError, match="receipt drifted"):
        formation.validate_pack_formation_receipt_v1(
            wrong_pack,
            private_pack=artifacts.private_pack,
            measurement_view=artifacts.measurement_view,
            prior_measurement_view=prior_view,
            preregistration_hash=preregistration_hash,
            acquisition_receipt_hash=acquisition_hash,
        )


def test_query_and_instruction_collisions_fail_from_commitments_only(
    synthetic_periods: dict[str, Path]
) -> None:
    prior_view = _old_view(synthetic_periods)
    fresh = formation.build_collision_checked_pack_v1(
        previous_source=synthetic_periods["new_previous"],
        current_source=synthetic_periods["new_current"],
        prior_measurement_view=prior_view,
    ).private_pack

    same_old_pack = period_pack.build_public_pack(
        previous_source=synthetic_periods["old_previous"],
        current_source=synthetic_periods["old_current"],
        previous_period_label="2025 Q3",
        current_period_label="2025 Q4",
        preregistration_seed="synthetic-old-period-seed",
        previous_container_root="/root/old-previous",
        current_container_root="/root/old-current",
    )
    with pytest.raises(formation.FreshFormationError, match="collides"):
        formation.assert_no_prior_commitment_collision_v1(
            new_pack=same_old_pack,
            prior_measurement_view=prior_view,
        )

    query_only = copy.deepcopy(prior_view)
    query_only["sealed_item_commitments"][0][
        "query_commitment_hash"
    ] = period_pack.payload_hash(fresh["items"][0]["query"])
    _rehash(query_only, "measurement_view_hash")
    assert query_only["sealed_content_persisted"] is False
    with pytest.raises(formation.FreshFormationError, match="collides"):
        formation.assert_no_prior_commitment_collision_v1(
            new_pack=fresh,
            prior_measurement_view=query_only,
        )

    instruction_only = copy.deepcopy(prior_view)
    instruction_only["sealed_item_commitments"][0][
        "instruction_sha256"
    ] = fresh["items"][0]["instruction_sha256"]
    _rehash(instruction_only, "measurement_view_hash")
    assert instruction_only["sealed_content_persisted"] is False
    with pytest.raises(formation.FreshFormationError, match="collides"):
        formation.assert_no_prior_commitment_collision_v1(
            new_pack=fresh,
            prior_measurement_view=instruction_only,
        )
