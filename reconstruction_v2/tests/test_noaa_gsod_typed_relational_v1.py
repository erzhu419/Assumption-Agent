from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from replication_runtime.noaa_gsod_v1.typed_relational import (
    MAX_CANDIDATES,
    TypedRelationalProgram,
    execute_frozen_operator,
    form_typed_relational_candidate,
    load_formation_receipt,
    load_frozen_program,
)
from replication_runtime.noaa_gsod_v1.contract import (
    ORACLE_IDS,
    STUDY_ID,
    TASK_CONTRACT,
    payload_hash,
    with_self_hash,
)
from replication_runtime.noaa_gsod_v1.train_export import TRAIN_VIEW_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_train_view(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    items = []
    for index in range(12):
        path = inputs / f"noaa_gsod_train_export_{index:02d}.csv"
        token = f"TRAIN_STATION_{index:02d}"
        rows = [
            (token, "2020-01-01", "1.00"),
            (token, "2020-01-02", "1.00"),
            (token, "2020-01-03", "1.00"),
            (token, "2020-01-04", "1.00"),
            (token, "2020-01-05", "1.00"),
            (token, "2020-02-01", "3.00"),
            (token, "2020-02-02", "3.00"),
            (token, "2020-03-01", "3.00"),
            (token, "2020-03-02", "3.00"),
            (token, "2020-04-01", "0.50"),
            (token, "2020-11-01", ""),
            (token, "2020-12-01", "99.99"),
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("STATION", "DATE", "PRCP"))
            writer.writerows(rows)
        oracle = {
            "mean_daily_precip_mm": "76.20",
            "month": "02",
            "valid_day_count": 2,
        }
        item_body = {
                "ordinal": index,
                "train_item_id": f"noaa_gsod_train_export_{index:02d}",
                "anonymized_station_token": token,
                "input_columns": ["STATION", "DATE", "PRCP"],
                "input_relative_path": f"inputs/{path.name}",
                "input_sha256": _sha256(path),
                "oracle_consensus": oracle,
                "oracle_consensus_hash": payload_hash(oracle),
        }
        items.append({**item_body, "train_item_hash": payload_hash(item_body)})
    view_body = {
        "candidate_imports": 0,
        "study_id": STUDY_ID,
        "partition": "train",
        "role": "candidate_formation_input_only",
        "train_item_count": 12,
        "items": items,
        "source_private_pack_hash": "a" * 64,
        "task_contract": TASK_CONTRACT,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
        "train_view_version": TRAIN_VIEW_VERSION,
        "oracle_consensus_ids": list(ORACLE_IDS),
        "model_calls": 0,
        "network_calls": 0,
        "online_judge_calls": 0,
        "scoring_calls": 0,
        "typed_operator_formed": False,
    }
    view = with_self_hash(view_body, "train_view_hash")
    path = root / "train_view.json"
    path.write_text(json.dumps(view), encoding="utf-8")
    return path


def test_finite_train_only_formation_crossfits_and_freezes(tmp_path: Path) -> None:
    train_view = _write_train_view(tmp_path / "train_view")
    output = tmp_path / "formed"

    result = form_typed_relational_candidate(train_view, output_dir=output)

    assert result.status == "formed_unique_exact_crossfit"
    assert result.program is not None
    assert result.program.type_issues() == ()
    assert result.program.missing_tokens == ("", "99.99")
    assert result.program.year == 2020
    assert result.program.aggregation == "mean"
    assert result.program.extreme == "argmax"
    assert result.program.tie_break == "earliest"
    assert result.program.unit_factor == "25.4"
    assert result.program.decimal_places == 2
    assert len(result.program.semantic_nodes) == 7
    assert result.receipt["search_receipt"]["candidate_count"] == 864
    assert result.receipt["search_receipt"]["candidate_count"] <= MAX_CANDIDATES
    assert result.receipt["crossfit_receipt"]["all_station_out_exact"] is True
    assert result.receipt["crossfit_receipt"]["selected_program_stable"] is True
    assert result.receipt["selection_receipt"]["exact_recovery_count"] == 12
    assert result.receipt["selection_receipt"]["contract_deviation_count"] == 0
    assert result.receipt["selection_receipt"]["contract_derived_resolution"] is True
    assert result.receipt["selection_receipt"]["exact_behavior_alias_class_size"] >= 1
    assert result.receipt["offline_contract"]["development_or_sealed_accessed"] is False
    assert result.receipt["claim_boundary"]["performance_claim"] is False

    loaded_receipt = load_formation_receipt(output / "formation.receipt.json")
    loaded_program = load_frozen_program(
        output / "frozen_program.json",
        receipt_path=output / "formation.receipt.json",
    )
    assert loaded_receipt == result.receipt
    assert loaded_program.program_hash == result.program.program_hash
    first_input = train_view.parent / "inputs" / "noaa_gsod_train_export_00.csv"
    assert json.loads(execute_frozen_operator(loaded_program, first_input)) == {
        "mean_daily_precip_mm": "76.20",
        "month": "02",
        "valid_day_count": 2,
    }


def test_receipt_and_input_hash_tampering_fail_closed(tmp_path: Path) -> None:
    train_view = _write_train_view(tmp_path / "train_view")
    output = tmp_path / "formed"
    form_typed_relational_candidate(train_view, output_dir=output)
    receipt_path = output / "formation.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "tampered"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_formation_receipt(receipt_path)

    input_path = train_view.parent / "inputs" / "noaa_gsod_train_export_00.csv"
    input_path.write_text(input_path.read_text() + "S00,2020-04-01,1.00\n")
    with pytest.raises(ValueError, match="input hash mismatch"):
        form_typed_relational_candidate(train_view)


def test_frozen_program_loader_rejects_declared_graph_and_envelope_drift(
    tmp_path: Path,
) -> None:
    train_view = _write_train_view(tmp_path / "train_view")
    output = tmp_path / "formed"
    form_typed_relational_candidate(train_view, output_dir=output)
    frozen_path = output / "frozen_program.json"
    receipt_path = output / "formation.receipt.json"
    original = json.loads(frozen_path.read_text())

    graph_drift = json.loads(json.dumps(original))
    graph_drift["program"]["semantic_nodes"][0]["tokens"] = ["tampered"]
    frozen_path.write_text(json.dumps(graph_drift))
    with pytest.raises(ValueError, match="canonical payload mismatch"):
        load_frozen_program(frozen_path, receipt_path=receipt_path)

    extra_field = json.loads(json.dumps(original))
    extra_field["undeclared"] = True
    frozen_path.write_text(json.dumps(extra_field))
    with pytest.raises(ValueError, match="envelope schema mismatch"):
        load_frozen_program(frozen_path, receipt_path=receipt_path)

    wrong_receipt = json.loads(receipt_path.read_text())
    wrong_receipt["selection_receipt"]["selected_program_hash"] = "0" * 64
    body = {key: value for key, value in wrong_receipt.items() if key != "receipt_hash"}
    wrong_receipt["receipt_hash"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    alternate_receipt_path = output / "alternate.receipt.json"
    alternate_receipt_path.write_text(json.dumps(wrong_receipt))
    frozen_path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="formation receipt binding mismatch"):
        load_frozen_program(
            frozen_path,
            receipt_path=alternate_receipt_path,
        )


def test_operator_rejects_nonexistent_iso_calendar_date(tmp_path: Path) -> None:
    program = TypedRelationalProgram(
        missing_tokens=("", "99.99"),
        year=2020,
        aggregation="mean",
        extreme="argmax",
        tie_break="earliest",
        unit_factor="25.4",
        rounding="ROUND_HALF_UP",
        decimal_places=2,
    )
    path = tmp_path / "invalid-date.csv"
    path.write_text(
        "STATION,DATE,PRCP\nTRAIN_STATION_00,2020-02-99,1.00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid DATE"):
        execute_frozen_operator(program, path)


def test_public_formation_result_is_self_bound_and_non_performance() -> None:
    path = (
        Path(__file__).parents[1]
        / "manifests"
        / "noaa_gsod_typed_relational_formation_result_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = payload.pop("result_hash")
    assert payload_hash(payload) == declared
    assert payload["status"] == "formed_unique_exact_crossfit"
    assert payload["train_exact_count"] == payload["train_item_count"] == 12
    assert payload["exact_behavior_alias_class_size"] == 4
    assert payload["contract_conformant_exact_candidate_count"] == 1
    assert payload["contract_derived_resolution"] is True
    assert payload["development_or_sealed_accessed"] is False
    assert payload["performance_claim"] is False
