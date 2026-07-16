from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from assumption_agent.benchmarks import hotpot_evaluator_robust_acquisition_v1 as acquisition
from assumption_agent.benchmarks import hotpot_recursive_acquisition_v1 as v2
from assumption_agent.models import stable_hash
from tests.test_hotpot_family_out_v1 import _source_row
from tests.test_hotpot_recursive_acquisition_v1 import (
    _git,
    _prior_pack_and_receipt,
    _project,
)


def _write_hashed(
    path: Path, payload: Mapping[str, Any], hash_field: str, mode: int = 0o644
) -> None:
    acquisition._write_json_exclusive(
        path, payload, hash_field=hash_field, mode=mode
    )


def _final_disposition(v2_receipt_path: Path) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": acquisition.V2_FINAL_DISPOSITION_SCHEMA,
        "status": "L4_narrow_positive_L5_no_promotion_terminal",
        "L4": {},
        "L5": {
            "M_search_authorized": False,
            "M_search_opened": False,
            "challenger_promoted": False,
            "evaluator_coevolution_achieved": False,
        },
        "bindings": {
            "acquisition_file_sha256": acquisition._sha256_file(v2_receipt_path)
        },
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
        "terminal_policy": {
            "future_L5_requires_new_mechanism_and_new_cohort": True,
            "same_anchor_retry_replay_resample": False,
            "same_anchor_challenger_substitution": False,
        },
    }
    return {**body, "disposition_sha256": stable_hash(body)}


def _prepared_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    list[dict[str, Any]],
    frozenset[str],
]:
    project = _project(tmp_path)
    rows = [_source_row(index) for index in range(360)]
    source = project / "artifacts" / "source.parquet"
    source.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), source)
    source_sha = acquisition._sha256_file(source)
    for module in (v2, acquisition):
        monkeypatch.setattr(module, "SOURCE_SIZE", source.stat().st_size)
        monkeypatch.setattr(module, "SOURCE_SHA256", source_sha)
        monkeypatch.setattr(module, "SOURCE_ROW_COUNT", len(rows))
    v2_selection = project / "v2selection.py"
    v2_selection.write_text("V2_SELECTION = 1\n", encoding="utf-8")
    v2_dependency = project / "v2dep.py"
    v2_dependency.write_text("V2_DEPENDENCY = 1\n", encoding="utf-8")
    retained = project / "retained.py"
    retained.write_text("RETAINED_P = 1\n", encoding="utf-8")
    future = project / "future.py"
    future.write_text("PORTFOLIO_VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        v2, "IMPLEMENTATION_RELATIVE_FILES", ("v2selection.py", "v2dep.py")
    )
    monkeypatch.setattr(
        acquisition, "V2_SELECTION_IMPLEMENTATION_RELATIVE", "v2selection.py"
    )
    lineage = (
        ("P_formation_receipt", "retained.py"),
        ("P_frozen_program", "retained.py"),
        ("M1_pre_run_freeze", "retained.py"),
        ("M1_positive_promotion_report", "retained.py"),
    )
    monkeypatch.setattr(v2, "RETAINED_P_LINEAGE_RELATIVE_FILES", lineage)
    monkeypatch.setattr(
        acquisition, "RETAINED_P_LINEAGE_RELATIVE_FILES", lineage
    )
    monkeypatch.setattr(
        acquisition, "IMPLEMENTATION_RELATIVE_FILES", ("impl.py", "future.py")
    )
    design_body: dict[str, Any] = {
        "schema": acquisition.PORTFOLIO_DESIGN_SCHEMA,
        "status": (
            "single_final_same_source_confirmatory_mechanism_fixed_before_new_cohort"
        ),
        "claim_boundary": {},
        "cohort_contract": {},
        "design_evidence": {},
        "execution_contract": {},
        "mechanism": {},
        "promotion_contract": {},
        "terminal_policy": {},
        "raw_content_persisted": False,
    }
    design = {
        **design_body,
        "design_sha256": stable_hash(design_body),
    }
    design_path = project / acquisition.PORTFOLIO_DESIGN_RELATIVE
    design_path.parent.mkdir()
    design_path.write_text(
        json.dumps(design, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        acquisition, "PORTFOLIO_DESIGN_SHA256", design["design_sha256"]
    )
    monkeypatch.setattr(
        acquisition,
        "PORTFOLIO_DESIGN_FILE_SHA256",
        acquisition._sha256_file(design_path),
    )

    _git(
        project,
        "add",
        "v2selection.py",
        "v2dep.py",
        "retained.py",
        "future.py",
        acquisition.PORTFOLIO_DESIGN_RELATIVE,
    )
    _git(
        project, "commit", "-q", "-m", "freeze synthetic dependencies and design"
    )

    original_pack, original_receipt, original_ids = _prior_pack_and_receipt(
        project, rows
    )
    secret = project / "artifacts" / "selection.key"
    v2.generate_selection_secret(project=project, output=secret)
    v2_prereg_path = project / "v2-prereg.json"
    v2_prereg = v2.build_preregistration(
        project=project,
        selection_secret_path=secret,
        prior_acquisition_receipt_path=original_receipt,
    )
    v2._write_json_exclusive(
        v2_prereg_path,
        v2_prereg,
        hash_field="preregistration_sha256",
        mode=0o644,
    )
    _git(project, "add", "v2-prereg.json")
    _git(project, "commit", "-q", "-m", "freeze v2 preregistration")

    v2_root = project / "artifacts" / "v2-pack"
    v2_receipt = v2.acquire_private_blocks(
        project=project,
        preregistration_path=v2_prereg_path,
        selection_secret_path=secret,
        prior_acquisition_receipt_path=original_receipt,
        prior_private_pack_path=original_pack,
        source_parquet_path=source,
        private_root=v2_root,
        private_locator_path=project / "artifacts" / "v2-locator.json",
    )
    v2_receipt_path = project / acquisition.V2_ACQUISITION_RELATIVE
    v2._write_json_exclusive(
        v2_receipt_path,
        v2_receipt,
        hash_field="acquisition_sha256",
        mode=0o644,
    )
    persisted_v2_receipt = json.loads(v2_receipt_path.read_text("utf-8"))
    monkeypatch.setattr(
        acquisition,
        "V2_ACQUISITION_FILE_SHA256",
        acquisition._sha256_file(v2_receipt_path),
    )
    monkeypatch.setattr(
        acquisition,
        "V2_ACQUISITION_SHA256",
        persisted_v2_receipt["acquisition_sha256"],
    )
    disposition_path = project / acquisition.V2_FINAL_DISPOSITION_RELATIVE
    _write_hashed(
        disposition_path,
        _final_disposition(v2_receipt_path),
        "disposition_sha256",
    )
    disposition = json.loads(disposition_path.read_text("utf-8"))
    monkeypatch.setattr(
        acquisition,
        "V2_FINAL_DISPOSITION_FILE_SHA256",
        acquisition._sha256_file(disposition_path),
    )
    monkeypatch.setattr(
        acquisition,
        "V2_FINAL_DISPOSITION_SHA256",
        disposition["disposition_sha256"],
    )
    _git(
        project,
        "add",
        acquisition.V2_ACQUISITION_RELATIVE,
        acquisition.V2_FINAL_DISPOSITION_RELATIVE,
    )
    _git(project, "commit", "-q", "-m", "freeze public v2 terminal record")
    return (
        project,
        source,
        original_pack,
        secret,
        v2_receipt_path,
        disposition_path,
        v2_root,
        rows,
        original_ids,
    )


def _build_and_commit_prereg(
    *,
    project: Path,
    secret: Path,
    v2_receipt: Path,
    disposition: Path,
) -> Path:
    path = project / "robust-prereg.json"
    payload = acquisition.build_preregistration(
        project=project,
        selection_secret_path=secret,
        v2_acquisition_receipt_path=v2_receipt,
        v2_final_disposition_path=disposition,
    )
    _write_hashed(path, payload, "preregistration_sha256")
    _git(project, "add", "robust-prereg.json")
    _git(project, "commit", "-q", "-m", "freeze robust preregistration")
    return path


def test_fixed_rank_window_blocks_future_closure_and_no_v2_pack_parameter() -> None:
    assert acquisition.BLOCK_COUNTS == {
        "A_form_0": 24,
        "A_form_1": 24,
        "F_search_0": 24,
        "F_search_1": 24,
        "A_hold": 48,
        "M_search": 24,
    }
    assert acquisition.SELECTED_COUNT == 168
    assert acquisition.RANK_WINDOW_START == 156
    assert acquisition.RANK_WINDOW_STOP == 324
    assert acquisition.SELECTION_DOMAIN_SEPARATOR == v2.VERSION
    assert (
        "assumption_agent/benchmarks/hotpot_evaluator_portfolio_coevolution_v1.py"
        in acquisition.IMPLEMENTATION_RELATIVE_FILES
    )
    parameters = inspect.signature(acquisition.acquire_private_blocks).parameters
    assert "original_private_pack_path" in parameters
    assert "v2_private_pack_path" not in parameters
    assert "v2_private_root" not in parameters


def test_preregistration_has_zero_row_access_and_binds_committed_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        project,
        _source,
        original_pack,
        secret,
        v2_receipt,
        disposition,
        _v2_root,
        _rows,
        _original_ids,
    ) = _prepared_v2(tmp_path, monkeypatch)

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("no private/source row may open during preregistration")

    monkeypatch.setattr(acquisition, "_read_original_private_ids_after_marker", forbidden)
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    payload = acquisition.build_preregistration(
        project=project,
        selection_secret_path=secret,
        v2_acquisition_receipt_path=v2_receipt,
        v2_final_disposition_path=disposition,
    )
    assert payload["safety"]["source_rows_read"] == 0
    assert payload["safety"]["original_twelve_private_pack_rows_read"] == 0
    assert payload["safety"]["v2_private_block_rows_read"] == 0
    assert payload["selection"]["selection_secret_reused_from_v2"] is True
    assert payload["selection"]["rank_window_start_inclusive"] == 156
    assert payload["selection"]["rank_window_stop_exclusive"] == 324
    assert payload["access_contract"]["v2_private_pack_path_parameter_accepted"] is False
    assert payload["access_contract"]["previous_M_search_content_opened"] is False
    assert payload["access_contract"]["previous_M_search_outcome_opened"] is False
    assert payload["portfolio_design_binding"]["schema"] == (
        acquisition.PORTFOLIO_DESIGN_SCHEMA
    )
    assert payload["portfolio_design_binding"]["design_sha256"] == (
        acquisition.PORTFOLIO_DESIGN_SHA256
    )
    assert payload["portfolio_design_binding"]["committed_custody"][
        "clean_tracked_HEAD_blob"
    ] is True
    public = json.dumps(payload, sort_keys=True)
    assert str(original_pack) not in public
    assert '"item_id"' not in public
    assert '"question"' not in public

    # Even a correctly self-hashed public artifact is rejected after its
    # committed portfolio-design byte identity changes.
    design_path = project / acquisition.PORTFOLIO_DESIGN_RELATIVE
    tampered = json.loads(design_path.read_text("utf-8"))
    tampered["status"] = "substituted_after_commit"
    body = dict(tampered)
    body.pop("design_sha256")
    tampered["design_sha256"] = stable_hash(body)
    design_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", "utf-8")
    with pytest.raises(
        acquisition.HotpotEvaluatorRobustAcquisitionError,
        match="portfolio evaluator design",
    ):
        acquisition.build_preregistration(
            project=project,
            selection_secret_path=secret,
            v2_acquisition_receipt_path=v2_receipt,
            v2_final_disposition_path=disposition,
        )


def test_v2_full_implementation_closure_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        project,
        _source,
        _original_pack,
        secret,
        v2_receipt,
        disposition,
        _v2_root,
        _rows,
        _original_ids,
    ) = _prepared_v2(tmp_path, monkeypatch)

    # This is deliberately not the selection implementation file.  It proves
    # the complete frozen v2 dependency closure, rather than one HMAC module,
    # is checked against live bytes.
    (project / "v2dep.py").write_text(
        "V2_DEPENDENCY = 2\n", encoding="utf-8"
    )
    with pytest.raises(
        acquisition.HotpotEvaluatorRobustAcquisitionError,
        match="v2 ordering|implementation",
    ):
        acquisition.build_preregistration(
            project=project,
            selection_secret_path=secret,
            v2_acquisition_receipt_path=v2_receipt,
            v2_final_disposition_path=disposition,
        )


@pytest.mark.parametrize("dirty_relative", ("future.py", "retained.py"))
def test_dirty_implementation_or_retained_p_closure_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty_relative: str,
) -> None:
    (
        project,
        _source,
        _original_pack,
        secret,
        v2_receipt,
        disposition,
        _v2_root,
        _rows,
        _original_ids,
    ) = _prepared_v2(tmp_path, monkeypatch)
    (project / dirty_relative).write_text("DIRTY = True\n", encoding="utf-8")
    with pytest.raises(
        acquisition.HotpotEvaluatorRobustAcquisitionError,
        match="clean tracked HEAD blob",
    ):
        acquisition.build_preregistration(
            project=project,
            selection_secret_path=secret,
            v2_acquisition_receipt_path=v2_receipt,
            v2_final_disposition_path=disposition,
        )


@pytest.mark.parametrize("substitute_kind", ("acquisition", "disposition"))
def test_committed_noncanonical_v2_substitute_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitute_kind: str,
) -> None:
    (
        project,
        _source,
        _original_pack,
        secret,
        v2_receipt,
        disposition,
        _v2_root,
        _rows,
        _original_ids,
    ) = _prepared_v2(tmp_path, monkeypatch)
    canonical = v2_receipt if substitute_kind == "acquisition" else disposition
    substitute = project / f"committed-{substitute_kind}-substitute.json"
    substitute.write_bytes(canonical.read_bytes())
    _git(project, "add", substitute.name)
    _git(
        project,
        "commit",
        "-q",
        "-m",
        f"commit identical {substitute_kind} substitute",
    )
    acquisition_path = (
        substitute if substitute_kind == "acquisition" else v2_receipt
    )
    disposition_path = (
        substitute if substitute_kind == "disposition" else disposition
    )
    with pytest.raises(
        acquisition.HotpotEvaluatorRobustAcquisitionError,
        match="fixed canonical path",
    ):
        acquisition.build_preregistration(
            project=project,
            selection_secret_path=secret,
            v2_acquisition_receipt_path=acquisition_path,
            v2_final_disposition_path=disposition_path,
        )


def test_next_rank_window_is_exact_disjoint_and_marker_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        project,
        source,
        original_pack,
        secret,
        v2_receipt,
        disposition,
        v2_root,
        rows,
        original_ids,
    ) = _prepared_v2(tmp_path, monkeypatch)
    prereg = _build_and_commit_prereg(
        project=project,
        secret=secret,
        v2_receipt=v2_receipt,
        disposition=disposition,
    )
    marker = project / acquisition.ACQUISITION_CONSUMPTION_RELATIVE
    future = project / "future.py"
    clean_future = future.read_text("utf-8")
    future.write_text("DIRTY_AFTER_PREREG = True\n", encoding="utf-8")
    with pytest.raises(
        acquisition.HotpotEvaluatorRobustAcquisitionError,
        match="clean tracked HEAD blob",
    ):
        acquisition.acquire_private_blocks(
            project=project,
            preregistration_path=prereg,
            selection_secret_path=secret,
            v2_acquisition_receipt_path=v2_receipt,
            v2_final_disposition_path=disposition,
            original_private_pack_path=original_pack,
            source_parquet_path=source,
            private_root=project / "artifacts" / "premarker-rejected-pack",
            private_locator_path=(
                project / "artifacts" / "premarker-rejected-locator.json"
            ),
        )
    assert not marker.exists()
    assert not (
        project / "artifacts" / "premarker-rejected-pack"
    ).exists()
    future.write_text(clean_future, encoding="utf-8")

    events: list[str] = []
    original_reader = acquisition._read_original_private_ids_after_marker
    original_parquet = pq.ParquetFile

    def pack_after_marker(**kwargs: Any):
        assert marker.is_file()
        events.append("original-pack")
        return original_reader(**kwargs)

    def source_after_pack(path: Path):
        assert marker.is_file()
        assert events == ["original-pack"]
        events.append("source")
        return original_parquet(path)

    monkeypatch.setattr(acquisition, "_read_original_private_ids_after_marker", pack_after_marker)
    monkeypatch.setattr(pq, "ParquetFile", source_after_pack)
    new_root = project / "artifacts" / "robust-pack"
    locator = project / "artifacts" / "robust-locator.json"
    receipt = acquisition.acquire_private_blocks(
        project=project,
        preregistration_path=prereg,
        selection_secret_path=secret,
        v2_acquisition_receipt_path=v2_receipt,
        v2_final_disposition_path=disposition,
        original_private_pack_path=original_pack,
        source_parquet_path=source,
        private_root=new_root,
        private_locator_path=locator,
    )
    assert events == ["original-pack", "source"]
    assert receipt["counts"]["selected_rows"] == 168
    assert receipt["counts"]["selected_previous_rank_window_overlap"] == 0
    assert receipt["selection_continuity"]["v2_private_block_files_opened"] == 0
    assert receipt["selection_continuity"]["previous_M_search_content_opened"] is False
    assert receipt["selection_continuity"]["previous_M_search_outcome_opened"] is False

    v2_ids = {
        json.loads(line)["item_id"]
        for block in v2.BLOCK_ORDER
        for line in (v2_root / f"{block}.jsonl").read_text("utf-8").splitlines()
    }
    new_ids: list[str] = []
    public_blocks = tuple(
        acquisition.BlockCommitment(**row)
        for row in receipt["commitments"]["block_files"]
    )
    for commitment in public_blocks:
        block_rows = acquisition.load_private_block(
            new_root / f"{commitment.block}.jsonl",
            commitment=commitment,
        )
        new_ids.extend(row["item_id"] for row in block_rows)
    assert len(v2_ids) == 156
    assert len(new_ids) == len(set(new_ids)) == 168
    assert set(new_ids).isdisjoint(v2_ids)
    assert set(new_ids).isdisjoint(original_ids)

    secret_bytes = secret.read_bytes()
    eligible = [
        acquisition._normalized_source_row(row) for row in rows
    ]
    eligible = [
        row
        for row in eligible
        if row is not None and row["item_id"] not in original_ids
    ]
    eligible.sort(
        key=lambda row: (
            acquisition._selection_key(row["item_id"], secret_bytes),
            row["item_id"],
        )
    )
    assert new_ids == [
        row["item_id"]
        for row in eligible[acquisition.RANK_WINDOW_START : acquisition.RANK_WINDOW_STOP]
    ]

    calls = list(events)
    with pytest.raises(FileExistsError, match="already consumed"):
        acquisition.acquire_private_blocks(
            project=project,
            preregistration_path=prereg,
            selection_secret_path=secret,
            v2_acquisition_receipt_path=v2_receipt,
            v2_final_disposition_path=disposition,
            original_private_pack_path=original_pack,
            source_parquet_path=source,
            private_root=project / "artifacts" / "alternate-pack",
            private_locator_path=project / "artifacts" / "alternate-locator.json",
        )
    assert events == calls


def test_public_and_private_binding_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        project,
        source,
        original_pack,
        secret,
        v2_receipt,
        disposition,
        _v2_root,
        _rows,
        _original_ids,
    ) = _prepared_v2(tmp_path, monkeypatch)
    prereg = _build_and_commit_prereg(
        project=project,
        secret=secret,
        v2_receipt=v2_receipt,
        disposition=disposition,
    )
    private_root = project / "artifacts" / "robust-pack"
    receipt = acquisition.acquire_private_blocks(
        project=project,
        preregistration_path=prereg,
        selection_secret_path=secret,
        v2_acquisition_receipt_path=v2_receipt,
        v2_final_disposition_path=disposition,
        original_private_pack_path=original_pack,
        source_parquet_path=source,
        private_root=private_root,
        private_locator_path=project / "artifacts" / "robust-locator.json",
    )
    public_path = project / "robust-acquisition.json"
    _write_hashed(public_path, receipt, "acquisition_sha256")
    loaded, blocks = acquisition.load_acquisition_binding(public_path)
    assert loaded["counts"]["selected_rows"] == 168
    assert [row.block for row in blocks] == list(acquisition.BLOCK_ORDER)

    first_path = private_root / f"{blocks[0].block}.jsonl"
    with first_path.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(
        acquisition.HotpotEvaluatorRobustAcquisitionError, match="hash"
    ):
        acquisition.load_private_block(first_path, commitment=blocks[0])

    tampered = json.loads(public_path.read_text("utf-8"))
    tampered["commitments"]["block_files"][0]["count"] = 23
    body = dict(tampered)
    body.pop("acquisition_sha256")
    tampered["acquisition_sha256"] = stable_hash(body)
    public_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", "utf-8")
    with pytest.raises(
        acquisition.HotpotEvaluatorRobustAcquisitionError, match="block"
    ):
        acquisition.load_acquisition_binding(public_path)
