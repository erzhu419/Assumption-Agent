from __future__ import annotations

import copy
import csv
from datetime import date, timedelta
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import replication_runtime.noaa_gsod_v1.development_freeze as development_freeze
import replication_runtime.noaa_gsod_v1.development_source as development_source
import replication_runtime.noaa_gsod_v1.pack as pack_module
from replication_runtime.noaa_gsod_v1.contract import (
    STUDY_ID,
    TASK_CONTRACT,
    NoaaGsodError,
    payload_hash,
    with_self_hash,
)
from replication_runtime.noaa_gsod_v1.development_freeze import (
    ARM_IDS,
    DEVELOPMENT_ITEM_COUNT,
    ENDPOINT_IDENTITY_VERSION,
    MODEL_ID,
    MODEL_WORK_UNIT_COUNT,
    WORK_UNIT_COUNT,
    ProviderIdentity,
    endpoint_identity_hash,
    prepare_development_pre_run_freeze,
    verify_controller_plan,
    verify_public_pre_run_freeze,
    verify_worker_plan,
)
from replication_runtime.noaa_gsod_v1.development_implementation import (
    build_development_implementation_set,
)
from replication_runtime.noaa_gsod_v1.development_schemas import (
    PUBLIC_BINDING_HASH_FIELDS,
    SOURCE_VIEW_BINDING_FIELDS,
)
from replication_runtime.noaa_gsod_v1.development_source import (
    PRIVATE_INDEX_NAME,
    export_development_source_view,
)
from replication_runtime.noaa_gsod_v1.pack import (
    build_private_pack,
    build_public_receipt,
    read_json,
    write_json,
)
from replication_runtime.noaa_gsod_v1.train_export import export_train_view
from replication_runtime.noaa_gsod_v1.typed_relational import (
    FORMATION_VERSION,
    OPERATOR_VERSION,
    TypedRelationalProgram,
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_station(path: Path, station_id: str, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = date(2020, 1, 1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["STATION", "DATE", "PRCP", "NAME", "LATITUDE"],
        )
        writer.writeheader()
        while current.year == 2020:
            writer.writerow(
                {
                    "STATION": station_id,
                    "DATE": current.isoformat(),
                    "PRCP": f"{((current.month + offset) % 7) / 10:.2f}",
                    "NAME": f"private synthetic station {offset}",
                    "LATITUDE": str(offset),
                }
            )
            current += timedelta(days=1)


def _source_bundle(tmp_path: Path) -> dict[str, object]:
    selected = []
    station_ids = []
    for index in range(24):
        station_id = f"73{index:09d}"
        station_ids.append(station_id)
        source = tmp_path / "source" / f"station-{index}.csv"
        _write_station(source, station_id, index)
        selected.append(
            {
                "source_path": str(source),
                "station_id": station_id,
                "station_metadata_commitment": f"{index + 200:064x}",
            }
        )
    private_root = tmp_path / "private-pack"
    private = build_private_pack(
        selected=selected,
        private_root=private_root,
        metadata_sha256="a" * 64,
        index_sha256="b" * 64,
        acquisition_statistics={"accepted_station_count": 24},
    )
    acquisition = build_public_receipt(
        private,
        metadata_url="https://official.example/metadata.csv",
        index_url="https://official.example/2020/",
        network_calls=26,
    )
    acquisition_path = tmp_path / "acquisition.json"
    write_json(acquisition_path, acquisition)

    train_root = tmp_path / "train-view"
    train_receipt_path = tmp_path / "train-preparation.json"
    export_train_view(
        private_pack_path=private_root / "private_pack.json",
        private_root=private_root,
        train_view_root=train_root,
        receipt_path=train_receipt_path,
    )
    train_receipt = read_json(train_receipt_path)

    source_view_root = tmp_path / "development-source-view"
    source_receipt_path = tmp_path / "development-source-receipt.json"
    export_development_source_view(
        private_pack_path=private_root / "private_pack.json",
        private_pack_root=private_root,
        acquisition_receipt_path=acquisition_path,
        source_view_root=source_view_root,
        public_receipt_path=source_receipt_path,
    )

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
    formation_body = {
        "formation_version": FORMATION_VERSION,
        "study_id": STUDY_ID,
        "status": "formed_unique_exact_crossfit",
        "offline_contract": {
            "partition": "train",
            "model_calls": 0,
            "network_calls": 0,
            "online_judge_calls": 0,
            "development_or_sealed_accessed": False,
        },
        "source_receipt": {
            "train_view_hash": train_receipt["train_view_hash"],
            "task_contract_hash": payload_hash(TASK_CONTRACT),
        },
        "selection_receipt": {"selected_program_hash": program.program_hash},
        "claim_boundary": {
            "train_only_formation": True,
            "performance_claim": False,
            "development_run_authorized": False,
            "sealed_run_authorized": False,
        },
        "raw_content_persisted": False,
    }
    formation = {
        **formation_body,
        "receipt_hash": _canonical_hash(formation_body),
    }
    formation_path = tmp_path / "formation.receipt.json"
    formation_path.write_text(json.dumps(formation), encoding="utf-8")
    program_envelope = {
        "operator_version": OPERATOR_VERSION,
        "program": program.to_dict(),
        "program_hash": program.program_hash,
        "formation_receipt_hash": formation["receipt_hash"],
        "raw_content_persisted": False,
    }
    program_path = tmp_path / "frozen_program.json"
    program_path.write_text(json.dumps(program_envelope), encoding="utf-8")
    return {
        "acquisition_path": acquisition_path,
        "formation_path": formation_path,
        "private_root": private_root,
        "program": program,
        "program_path": program_path,
        "source_index_path": source_view_root / PRIVATE_INDEX_NAME,
        "source_receipt_path": source_receipt_path,
        "source_view_root": source_view_root,
        "station_ids": station_ids,
        "train_receipt_path": train_receipt_path,
    }


def _provider() -> ProviderIdentity:
    return ProviderIdentity(
        plus_channel_id="ruoli_plus",
        plus_endpoint_origin="https://ruoli.dev",
        pro_channel_id="ruoli_pro",
        pro_endpoint_origin="https://RUOLI.DEV",
    )


def _freeze(
    bundle: dict[str, object],
    *,
    development_root: Path,
    public_path: Path,
) -> dict:
    return prepare_development_pre_run_freeze(
        development_source_view_root=Path(bundle["source_view_root"]),
        development_source_index_path=Path(bundle["source_index_path"]),
        development_source_receipt_path=Path(bundle["source_receipt_path"]),
        acquisition_receipt_path=Path(bundle["acquisition_path"]),
        development_root=development_root,
        public_freeze_path=public_path,
        train_preparation_receipt_path=Path(bundle["train_receipt_path"]),
        formation_receipt_path=Path(bundle["formation_path"]),
        frozen_program_path=Path(bundle["program_path"]),
        provider_identity=_provider(),
    )


def test_pre_run_freeze_consumes_only_gold_free_source_view(
    tmp_path: Path,
) -> None:
    assert MODEL_ID == "gpt-5.4-mini"
    bundle = _source_bundle(tmp_path)
    development_root = tmp_path / "unique-development-root"
    public_path = tmp_path / "public-freeze.json"
    summary = _freeze(
        bundle,
        development_root=development_root,
        public_path=public_path,
    )
    assert summary["development_item_count"] == DEVELOPMENT_ITEM_COUNT
    assert summary["model_work_unit_count"] == MODEL_WORK_UNIT_COUNT
    assert summary["total_work_unit_count"] == WORK_UNIT_COUNT

    worker = read_json(development_root / "worker_plan.json")
    controller = read_json(development_root / "controller_plan.private.json")
    public = read_json(public_path)
    assert verify_worker_plan(worker, development_root=development_root) == worker
    assert verify_controller_plan(controller, worker_plan=worker) == controller
    assert verify_public_pre_run_freeze(public) == public
    assert set(controller["source_view_binding"]) == SOURCE_VIEW_BINDING_FIELDS
    assert set(public["binding_hashes"]) == PUBLIC_BINDING_HASH_FIELDS
    assert public["binding_hashes"]["implementation_set_hash"] == (
        build_development_implementation_set()["implementation_set_hash"]
    )
    acquisition = read_json(Path(bundle["acquisition_path"]))
    assert public["binding_hashes"]["acquisition_receipt_hash"] == (
        acquisition["receipt_hash"]
    )
    assert controller["source_view_binding"]["source_view_input_set_hash"] == (
        controller["source_view_binding"]["staged_input_set_hash"]
    )
    assert public["binding_hashes"]["development_source_input_set_hash"] == (
        public["binding_hashes"]["staged_input_set_hash"]
    )
    expected_pairs = {
        (f"development_item_{ordinal:02d}", arm)
        for ordinal in range(DEVELOPMENT_ITEM_COUNT)
        for arm in ARM_IDS
    }
    assert {
        (unit["anonymous_item_id"], unit["arm"]) for unit in worker["work_units"]
    } == expected_pairs
    assert len({unit["work_unit_id"] for unit in worker["work_units"]}) == 18
    assert all(unit["model_request_hash"] is None for unit in worker["work_units"])
    assert worker["batch_policy"]["maximum_model_concurrency"] == 12

    staged_bytes = b"".join(
        (development_root / item["input_relative_path"]).read_bytes()
        for item in worker["items"]
    )
    for station_id in bundle["station_ids"]:
        assert station_id.encode("ascii") not in staged_bytes
    controller_serialized = json.dumps(controller, sort_keys=True)
    public_serialized = json.dumps(public, sort_keys=True)
    for forbidden in (
        "source_private_pack",
        "private_pack_path",
        "source_item_commitment",
        "station_id",
        "raw_csv_relative_path",
        "oracle_outputs",
    ):
        assert f'"{forbidden}"' not in controller_serialized
        assert f'"{forbidden}"' not in public_serialized
    assert "development_consumed" not in public
    assert public["freeze_state"] == {
        "development_input_accessed": True,
        "development_input_staged": True,
        "generation_joined_count": 0,
        "generation_started": False,
        "gold_released": False,
        "launch_authorized": False,
        "model_request_hashes_precommitted": False,
        "operator_joined_count": 0,
        "scored": False,
        "sealed_runtime_accessed": False,
        "staged_item_count": 6,
        "status": "pre_run_frozen_not_launched",
    }
    assert all(value == 0 for value in public["call_ledger_at_freeze"].values())


def test_freeze_signature_and_execution_never_use_monolithic_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _source_bundle(tmp_path)
    parameters = inspect.signature(prepare_development_pre_run_freeze).parameters
    assert "private_pack_path" not in parameters
    assert "private_pack_root" not in parameters
    assert "acquisition_receipt_path" in parameters
    assert not hasattr(development_freeze, "verify_private_pack")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("freeze reached the acquisition custodian")

    monkeypatch.setattr(pack_module, "verify_private_pack", forbidden)
    monkeypatch.setattr(development_source, "verify_private_pack", forbidden)
    _freeze(
        bundle,
        development_root=tmp_path / "development",
        public_path=tmp_path / "public.json",
    )


def test_source_view_extra_material_and_input_tamper_fail_before_staging(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle(tmp_path)
    source_root = Path(bundle["source_view_root"])
    unexpected = source_root / "unexpected-sealed-map.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NoaaGsodError, match="unexpected material"):
        _freeze(
            bundle,
            development_root=tmp_path / "development-extra",
            public_path=tmp_path / "public-extra.json",
        )
    assert not (tmp_path / "development-extra").exists()
    unexpected.unlink()

    index = read_json(Path(bundle["source_index_path"]))
    first_input = source_root / index["items"][0]["input_relative_path"]
    first_input.write_bytes(first_input.read_bytes() + b"\n")
    with pytest.raises(NoaaGsodError, match="input hash mismatch"):
        _freeze(
            bundle,
            development_root=tmp_path / "development-tampered",
            public_path=tmp_path / "public-tampered.json",
        )
    assert not (tmp_path / "development-tampered").exists()


def test_freeze_requires_the_exact_public_acquisition_receipt(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle(tmp_path)
    acquisition_path = Path(bundle["acquisition_path"])
    acquisition_path.write_bytes(acquisition_path.read_bytes() + b"\n")

    with pytest.raises(NoaaGsodError, match="differs from acquisition"):
        _freeze(
            bundle,
            development_root=tmp_path / "development-acquisition-drift",
            public_path=tmp_path / "public-acquisition-drift.json",
        )
    assert not (tmp_path / "development-acquisition-drift").exists()


def test_worker_verifier_requires_exact_cartesian_units_and_nested_schema(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle(tmp_path)
    root = tmp_path / "development"
    _freeze(bundle, development_root=root, public_path=tmp_path / "public.json")
    worker = read_json(root / "worker_plan.json")

    body = copy.deepcopy(worker)
    body.pop("worker_plan_hash")
    duplicate = body["work_units"][0]
    replaced = body["work_units"][-3]
    replaced.update(
        {
            "anonymous_item_id": duplicate["anonymous_item_id"],
            "arm": duplicate["arm"],
            "input_sha256": duplicate["input_sha256"],
            "work_unit_id": duplicate["work_unit_id"],
        }
    )
    tampered = with_self_hash(body, "worker_plan_hash")
    with pytest.raises(NoaaGsodError, match="Cartesian identity"):
        verify_worker_plan(tampered, development_root=root)

    body = copy.deepcopy(worker)
    body.pop("worker_plan_hash")
    body["batch_policy"]["unexpected"] = False
    tampered = with_self_hash(body, "worker_plan_hash")
    with pytest.raises(NoaaGsodError, match="batch policy mismatch"):
        verify_worker_plan(tampered, development_root=root)

    body = copy.deepcopy(worker)
    body.pop("worker_plan_hash")
    body["shared_context"]["unexpected"] = False
    tampered = with_self_hash(body, "worker_plan_hash")
    with pytest.raises(NoaaGsodError, match="shared context mismatch"):
        verify_worker_plan(tampered, development_root=root)


def test_controller_and_public_nested_objects_are_exact_not_vacuous(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle(tmp_path)
    root = tmp_path / "development"
    public_path = tmp_path / "public.json"
    _freeze(bundle, development_root=root, public_path=public_path)
    worker = read_json(root / "worker_plan.json")
    controller = read_json(root / "controller_plan.private.json")
    public = read_json(public_path)

    controller_body = copy.deepcopy(controller)
    controller_body.pop("controller_plan_hash")
    controller_body["source_view_binding"].pop("source_view_tree_hash")
    tampered_controller = with_self_hash(controller_body, "controller_plan_hash")
    with pytest.raises(NoaaGsodError, match="source-view binding schema"):
        verify_controller_plan(tampered_controller, worker_plan=worker)

    for field, message in (
        ("binding_hashes", "binding schema"),
        ("call_ledger_at_freeze", "call ledger"),
        ("content_boundary", "content boundary"),
    ):
        public_body = copy.deepcopy(public)
        public_body.pop("pre_run_freeze_hash")
        public_body[field] = {}
        tampered_public = with_self_hash(public_body, "pre_run_freeze_hash")
        with pytest.raises(NoaaGsodError, match=message):
            verify_public_pre_run_freeze(tampered_public)


def test_historical_public_verification_does_not_rehash_live_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _source_bundle(tmp_path)
    root = tmp_path / "development"
    public_path = tmp_path / "public.json"
    _freeze(bundle, development_root=root, public_path=public_path)
    public = read_json(public_path)

    def forbidden_live_rehash(*_args, **_kwargs):
        raise AssertionError("historical verifier rehashed current implementation")

    monkeypatch.setattr(
        development_freeze,
        "build_development_implementation_set",
        forbidden_live_rehash,
    )
    assert verify_public_pre_run_freeze(public) == public


def test_provider_identity_uses_one_canonical_origin_hash_policy(
    tmp_path: Path,
) -> None:
    del tmp_path
    provider = _provider()
    provider.validate()
    assert ENDPOINT_IDENTITY_VERSION == "sha256_canonical_origin_payload_v1"
    assert provider.plus_endpoint_identity_hash == provider.pro_endpoint_identity_hash
    assert provider.plus_endpoint_identity_hash == endpoint_identity_hash(
        "https://ruoli.dev"
    )
    assert provider.private_policy()["secret_hmac_precommit_phase"] == (
        "runner_launch_precommit_before_any_model_submission"
    )
    with pytest.raises(NoaaGsodError, match="valid Plus"):
        ProviderIdentity(
            plus_channel_id="primary",
            plus_endpoint_origin="https://ruoli.dev",
            pro_channel_id="ruoli_pro",
            pro_endpoint_origin="https://ruoli.dev",
        ).validate()
    with pytest.raises(NoaaGsodError, match="HTTPS origin"):
        ProviderIdentity(
            plus_channel_id="ruoli_plus",
            plus_endpoint_origin="http://ruoli.dev",
            pro_channel_id="ruoli_pro",
            pro_endpoint_origin="https://ruoli.dev",
        ).validate()
    with pytest.raises(NoaaGsodError, match="canonical HTTPS origin"):
        ProviderIdentity(
            plus_channel_id="ruoli_plus",
            plus_endpoint_origin="https://ruoli.dev/v1",
            pro_channel_id="ruoli_pro",
            pro_endpoint_origin="https://ruoli.dev",
        ).validate()
    with pytest.raises(NoaaGsodError, match="registered Ruoli origin"):
        ProviderIdentity(
            plus_channel_id="ruoli_plus",
            plus_endpoint_origin="https://example.com",
            pro_channel_id="ruoli_pro",
            pro_endpoint_origin="https://ruoli.dev",
        ).validate()


def test_candidate_loader_receives_exact_formation_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _source_bundle(tmp_path)
    original_loader = development_freeze.load_frozen_program
    observed: dict[str, Path] = {}

    def recording_loader(path, *, receipt_path=None):
        observed["receipt_path"] = Path(receipt_path)
        return original_loader(path, receipt_path=receipt_path)

    monkeypatch.setattr(development_freeze, "load_frozen_program", recording_loader)
    _freeze(
        bundle,
        development_root=tmp_path / "development",
        public_path=tmp_path / "public.json",
    )
    assert observed["receipt_path"] == Path(bundle["formation_path"])


@pytest.mark.parametrize("relation", ["inside", "ancestor"])
def test_development_root_must_not_overlap_gold_free_source_view(
    tmp_path: Path,
    relation: str,
) -> None:
    bundle = _source_bundle(tmp_path)
    source_root = Path(bundle["source_view_root"])
    development_root = (
        source_root / "nested-development"
        if relation == "inside"
        else source_root.parent
    )
    with pytest.raises(NoaaGsodError, match="overlaps gold-free source-view"):
        _freeze(
            bundle,
            development_root=development_root,
            public_path=tmp_path / f"public-{relation}.json",
        )


def test_public_freeze_is_atomic_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _source_bundle(tmp_path)
    development_root = tmp_path / "development-existing"
    public_path = tmp_path / "public-existing.json"
    public_path.write_bytes(b"external-owner\n")
    with pytest.raises(NoaaGsodError, match="no-clobber"):
        _freeze(
            bundle,
            development_root=development_root,
            public_path=public_path,
        )
    assert public_path.read_bytes() == b"external-owner\n"
    assert not development_root.exists()

    racing_root = tmp_path / "development-racing"
    racing_public = tmp_path / "public-racing.json"
    original_link = development_freeze.os.link

    def racing_link(source, destination):
        Path(destination).write_bytes(b"racing-owner\n")
        return original_link(source, destination)

    monkeypatch.setattr(development_freeze.os, "link", racing_link)
    with pytest.raises(NoaaGsodError, match="no-clobber"):
        _freeze(
            bundle,
            development_root=racing_root,
            public_path=racing_public,
        )
    assert racing_public.read_bytes() == b"racing-owner\n"
    assert not racing_root.exists()
