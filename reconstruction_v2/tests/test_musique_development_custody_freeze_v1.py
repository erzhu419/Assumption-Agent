from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from assumption_agent.models import stable_hash
import assumption_agent.benchmarks.musique_development_custody_v1 as custody
import assumption_agent.benchmarks.musique_development_freeze_v1 as freeze
from assumption_agent.benchmarks.musique_development_custody_v1 import (
    EVALUATOR_INDEX_NAME,
    FORMAL_PUBLIC_CUSTODY_RECEIPT_RELATIVE,
    FORMAL_PUBLIC_PRIVATE_INDEX_BINDING_RELATIVE,
    GENERATION_DIRECTORY,
    RUNNER_PRIVATE_INDEX_NAME,
    MuSiQueDevelopmentCustodyError,
    export_development_source_view,
    export_synthetic_development_source_view_for_tests,
    load_generation_item,
    load_public_private_index_binding,
    verify_public_custody_receipt,
)
from assumption_agent.benchmarks.musique_development_freeze_v1 import (
    CONTROLLER_PLAN_NAME,
    PRIVATE_INDEX_NAME,
    WORKER_PLAN_NAME,
    MuSiQueDevelopmentFreezeError,
    prepare_synthetic_development_pre_run_freeze_for_tests,
    verify_controller_plan,
    verify_public_pre_run_freeze,
    verify_worker_plan,
)
from assumption_agent.benchmarks.musique_official_core_comparison_v1 import (
    ACQUISITION_SCHEMA,
    PRIVATE_PACK_SCHEMA,
)
from assumption_agent.benchmarks.musique_three_arm_formal_runner_v1 import (
    ARM_IDS,
    ITEM_INPUT_VERSION,
    PRIVATE_INDEX_VERSION,
    WORK_UNIT_COUNT,
)
from assumption_agent.benchmarks.musique_typed_retriever_formation_v1 import (
    CLAIM_SCOPE,
    form_musique_typed_retriever,
)


PROJECT = Path(__file__).resolve().parents[1]
PREREGISTRATION = PROJECT / "manifests/musique_official_core_comparison_v1_preregistration.json"
QUALIFICATION = PROJECT / "manifests/official_hipporag_runtime_adapter_qualification_v1.json"
OFFICIAL_BINDING = PROJECT / "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private_row(split: str, ordinal: int) -> dict[str, object]:
    corpus = [
        {
            "idx": index,
            "title": f"Synthetic title {split} {ordinal} {index}",
            "text": (
                f"Synthetic bridge {ordinal} city evidence {index}."
                if index < 2
                else f"Unrelated synthetic document {index}."
            ),
            "is_supporting": index < 2,
        }
        for index in range(7)
    ]
    return {
        "schema": PRIVATE_PACK_SCHEMA,
        "split": split,
        "item_id": f"private-source-{split}-{ordinal}",
        "question": f"Which city is linked to synthetic bridge {ordinal}?",
        "corpus": corpus,
        "answers": [f"Answer {ordinal}", f"Alias {ordinal}"],
        "normalized_answers": [f"answer {ordinal}", f"alias {ordinal}"],
        "support_indices": [0, 1],
        "source_row_sha256": _sha256_bytes(f"source:{split}:{ordinal}".encode()),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    return _sha256_bytes(raw), stable_hash([stable_hash(row) for row in rows])


def _inputs(tmp_path: Path) -> dict[str, Path]:
    private = tmp_path / "private"
    train = [_private_row("train", ordinal) for ordinal in range(12)]
    development = [_private_row("development", ordinal) for ordinal in range(6)]
    train_path = private / "train.jsonl"
    development_path = private / "development.jsonl"
    train_hash, train_set = _write_jsonl(train_path, train)
    development_hash, development_set = _write_jsonl(development_path, development)
    sealed = private / "residual_sealed.jsonl"
    sealed.write_text("DO-NOT-OPEN\n", encoding="utf-8")
    split_files = [
        {"split": "train", "count": 12, "file_sha256": train_hash, "item_commitment_set_sha256": train_set},
        {"split": "development", "count": 6, "file_sha256": development_hash, "item_commitment_set_sha256": development_set},
        {"split": "residual_sealed", "count": 6, "file_sha256": "c" * 64, "item_commitment_set_sha256": "d" * 64},
    ]
    acquisition_body = {
        "schema": ACQUISITION_SCHEMA,
        "decision": "private_pack_formed_no_model_execution_authorized",
        "source": {"claim_scope": CLAIM_SCOPE},
        "counts": {"selected_rows": 24, "splits": {"train": 12, "development": 6, "residual_sealed": 6}},
        "commitments": {"private_pack_sha256": stable_hash(split_files), "split_files": split_files},
    }
    acquisition = {**acquisition_body, "acquisition_sha256": stable_hash(acquisition_body)}
    public = tmp_path / "public"
    public.mkdir()
    acquisition_path = public / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    formation_root = tmp_path / "formation"
    form_musique_typed_retriever(train_path, acquisition_path, output_dir=formation_root)
    return {
        "private_root": private,
        "development": development_path,
        "sealed": sealed,
        "acquisition": acquisition_path,
        "formation": formation_root / "formation.receipt.json",
        "program": formation_root / "frozen_program.json",
    }


def _custody(tmp_path: Path) -> dict[str, Any]:
    inputs = _inputs(tmp_path)
    source = tmp_path / "source-view"
    receipt = tmp_path / "public" / "custody.json"
    sidecar = tmp_path / "public" / "private_index.binding.json"
    value = export_synthetic_development_source_view_for_tests(
        development_jsonl_path=inputs["development"],
        acquisition_receipt_path=inputs["acquisition"],
        formation_receipt_path=inputs["formation"],
        frozen_program_path=inputs["program"],
        source_view_root=source,
        public_receipt_path=receipt,
        public_private_index_binding_path=sidecar,
    )
    return {**inputs, "source": source, "receipt": receipt, "sidecar": sidecar, "custody": value}


def _freeze(tmp_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    development = tmp_path / "formal-development"
    public = tmp_path / "public" / "freeze.json"
    value = prepare_synthetic_development_pre_run_freeze_for_tests(
        source_view_root=bundle["source"],
        custody_receipt_path=bundle["receipt"],
        private_index_binding_path=bundle["sidecar"],
        preregistration_path=PREREGISTRATION,
        acquisition_receipt_path=bundle["acquisition"],
        formation_receipt_path=bundle["formation"],
        frozen_program_path=bundle["program"],
        qualification_path=QUALIFICATION,
        official_adapter_binding_path=OFFICIAL_BINDING,
        development_root=development,
        public_freeze_path=public,
        plus_channel_id="ruoli_plus",
        pro_channel_id="ruoli_pro",
    )
    return {"development_root": development, "public_path": public, "freeze": value}


def test_custody_gold_separation_public_hash_count_and_runner_native_index(tmp_path: Path) -> None:
    bundle = _custody(tmp_path)
    receipt = bundle["custody"]
    assert set(receipt) == {"schema", "hashes", "counts", "receipt_sha256"}
    assert verify_public_custody_receipt(receipt) == receipt
    public_text = bundle["receipt"].read_text(encoding="utf-8")
    assert "Which city" not in public_text
    assert "private-source" not in public_text

    generation_raw = b""
    for ordinal in range(6):
        item = load_generation_item(bundle["source"], ordinal)
        assert item["schema"] == ITEM_INPUT_VERSION
        assert set(item) == {"schema", "anonymous_item_id", "question", "corpus"}
        assert all(set(row) == {"idx", "title", "paragraph_text"} for row in item["corpus"])
        generation_raw += (bundle["source"] / GENERATION_DIRECTORY / f"development_item_{ordinal:02d}.json").read_bytes()
    for token in (b"answers", b"support_indices", b"is_supporting", b"private-source"):
        assert token not in generation_raw

    audit = json.loads((bundle["source"] / EVALUATOR_INDEX_NAME).read_text())
    assert audit["custody_receipt_sha256"] == receipt["receipt_sha256"]
    assert all(set(row) == {"anonymous_item_id", "answers", "normalized_answers", "support_indices"} for row in audit["items"])
    runner_private = json.loads((bundle["source"] / RUNNER_PRIVATE_INDEX_NAME).read_text())
    assert set(runner_private) == {"private_index_version", "custody_receipt_sha256", "items", "private_index_hash"}
    assert runner_private["private_index_version"] == PRIVATE_INDEX_VERSION
    assert all(set(row) == {"anonymous_item_id", "accepted_aliases", "support_indices"} for row in runner_private["items"])
    sidecar = load_public_private_index_binding(bundle["sidecar"])
    assert sidecar["private_index_file_sha256"] == _sha256_bytes((bundle["source"] / RUNNER_PRIVATE_INDEX_NAME).read_bytes())
    assert os.stat(bundle["source"] / RUNNER_PRIVATE_INDEX_NAME).st_mode & 0o077 == 0


def test_launchable_freeze_matches_runner_schema_and_exact_18_grid(tmp_path: Path) -> None:
    bundle = _custody(tmp_path)
    frozen = _freeze(tmp_path, bundle)
    public = frozen["freeze"]
    root = frozen["development_root"]
    assert verify_public_pre_run_freeze(public) == public
    assert public["authorization"]["launch_authorized"] is True
    assert public["execution_contract"]["maximum_model_concurrency"] == 18
    assert public["execution_contract"]["retries"] == 0
    assert public["gold_release_contract"]["private_index_copied_as_opaque_bytes"] is True
    assert public["protocol_amendment"]["formal_budget_bytes"] == 65536
    assert public["protocol_amendment"]["token_budget_claimed_by_formal_protocol"] is False
    expected_implementation = custody.current_development_implementation_binding()[
        "set_sha256"
    ]
    assert (
        bundle["custody"]["hashes"]["development_implementation_set_sha256"]
        == expected_implementation
        == public["binding_hashes"]["development_implementation_set_sha256"]
    )

    worker = json.loads((root / WORKER_PLAN_NAME).read_text())
    controller = json.loads((root / CONTROLLER_PLAN_NAME).read_text())
    assert verify_worker_plan(worker, development_root=root) == worker
    assert verify_controller_plan(controller, worker_plan=worker) == controller
    assert worker["execution_root_relative_path"] == "formal_execution"
    assert worker["consumption_marker_relative_path"] == "execution.authorization.consumed.json"
    assert worker["shared_contract"]["model_request_body_byte_budget"] == 65536
    assert worker["shared_contract"]["overflow_policy"] == "fail_closed_no_truncation"
    assert len(controller["work_units"]) == WORK_UNIT_COUNT
    assert {(row["anonymous_item_id"], row["arm_id"]) for row in controller["work_units"]} == {
        (f"development_item_{ordinal:02d}", arm) for ordinal in range(6) for arm in ARM_IDS
    }
    assert all(row["top_k"] == 5 and row["attempt_budget"] == 1 for row in controller["work_units"])
    assert all(row["generator_request_sha256"] is None for row in controller["work_units"])
    assert (root / PRIVATE_INDEX_NAME).read_bytes() == (bundle["source"] / RUNNER_PRIVATE_INDEX_NAME).read_bytes()


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("extra_key", True),
        ("work_unit_id", "forged"),
        ("input_relative_path", "inputs/development_item_01.json"),
        ("input_sha256", "0" * 64),
        ("top_k", 999),
        ("attempt_budget", 999),
        ("generator_request_sha256", "1" * 64),
        ("request_hash_state", "forged"),
    ),
)
def test_controller_verifier_rejects_every_formal_work_unit_drift(
    tmp_path: Path, mutation: str, value: object
) -> None:
    bundle = _custody(tmp_path)
    frozen = _freeze(tmp_path, bundle)
    root = frozen["development_root"]
    worker = json.loads((root / WORKER_PLAN_NAME).read_text())
    controller = json.loads((root / CONTROLLER_PLAN_NAME).read_text())
    changed = copy.deepcopy(controller)
    changed["work_units"][0][mutation] = value
    body = dict(changed)
    body.pop("controller_plan_hash")
    changed["controller_plan_hash"] = stable_hash(body)
    with pytest.raises(MuSiQueDevelopmentFreezeError):
        verify_controller_plan(changed, worker_plan=worker)


def test_formal_anchor_rejects_self_consistent_substitute_before_development_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    opened_development = False
    original = custody._secure_read_bytes

    def recording(path: str | Path, **kwargs: Any) -> bytes:
        nonlocal opened_development
        if Path(path).absolute() == inputs["development"].absolute():
            opened_development = True
        return original(path, **kwargs)

    monkeypatch.setattr(custody, "_secure_read_bytes", recording)
    with pytest.raises(MuSiQueDevelopmentCustodyError, match="trust root"):
        export_development_source_view(
            development_jsonl_path=inputs["development"],
            preregistration_path=PREREGISTRATION,
            acquisition_receipt_path=inputs["acquisition"],
            formation_receipt_path=inputs["formation"],
            frozen_program_path=inputs["program"],
            qualification_path=QUALIFICATION,
            official_adapter_binding_path=OFFICIAL_BINDING,
            source_view_root=tmp_path / "formal-source",
            public_receipt_path=PROJECT / FORMAL_PUBLIC_CUSTODY_RECEIPT_RELATIVE,
            public_private_index_binding_path=(
                PROJECT / FORMAL_PUBLIC_PRIVATE_INDEX_BINDING_RELATIVE
            ),
        )
    assert opened_development is False


def test_freeze_rejects_custody_implementation_drift_before_source_view_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _custody(tmp_path)
    receipt = json.loads(bundle["receipt"].read_text(encoding="utf-8"))
    receipt["hashes"]["development_implementation_set_sha256"] = "0" * 64
    body = dict(receipt)
    body.pop("receipt_sha256")
    receipt["receipt_sha256"] = stable_hash(body)
    bundle["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    def forbidden_source_access(_root: str | Path) -> str:
        raise AssertionError("source view opened before implementation binding check")

    monkeypatch.setattr(freeze, "generation_view_set_sha256", forbidden_source_access)
    with pytest.raises(MuSiQueDevelopmentFreezeError, match="implementation binding drifted"):
        _freeze(tmp_path, bundle)


def test_formal_public_outputs_reject_unregistered_paths_before_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> bytes:
        raise AssertionError("formal development opened before output registration check")

    monkeypatch.setattr(custody, "_secure_read_bytes", forbidden)
    with pytest.raises(MuSiQueDevelopmentCustodyError, match="registered manifest paths"):
        export_development_source_view(
            development_jsonl_path=inputs["development"],
            preregistration_path=PREREGISTRATION,
            acquisition_receipt_path=inputs["acquisition"],
            formation_receipt_path=inputs["formation"],
            frozen_program_path=inputs["program"],
            qualification_path=QUALIFICATION,
            official_adapter_binding_path=OFFICIAL_BINDING,
            source_view_root=tmp_path / "formal-source",
            public_receipt_path=tmp_path / "public" / "wrong-custody.json",
            public_private_index_binding_path=tmp_path / "public" / "wrong-binding.json",
        )

    monkeypatch.undo()
    bundle = _custody(tmp_path / "freeze-case")
    with pytest.raises(MuSiQueDevelopmentFreezeError, match="registered manifest path"):
        freeze.prepare_development_pre_run_freeze(
            source_view_root=bundle["source"],
            custody_receipt_path=bundle["receipt"],
            private_index_binding_path=bundle["sidecar"],
            preregistration_path=PREREGISTRATION,
            acquisition_receipt_path=bundle["acquisition"],
            formation_receipt_path=bundle["formation"],
            frozen_program_path=bundle["program"],
            qualification_path=QUALIFICATION,
            official_adapter_binding_path=OFFICIAL_BINDING,
            development_root=tmp_path / "formal-development",
            public_freeze_path=tmp_path / "public" / "wrong-freeze.json",
            plus_channel_id="ruoli_plus",
            pro_channel_id="ruoli_pro",
        )


def test_intermediate_symlink_fails_closed(tmp_path: Path) -> None:
    bundle = _custody(tmp_path)
    alias = tmp_path / "alias-source"
    alias.mkdir()
    (alias / GENERATION_DIRECTORY).symlink_to(bundle["source"] / GENERATION_DIRECTORY, target_is_directory=True)
    with pytest.raises(MuSiQueDevelopmentCustodyError):
        load_generation_item(alias, 0)


def test_no_sealed_open_or_private_parent_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    private_root = inputs["private_root"].resolve()
    sealed = inputs["sealed"].resolve()
    original_secure = custody._secure_read_bytes
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_os_open = os.open
    original_listdir = os.listdir
    original_scandir = os.scandir

    def secure_guard(path: str | Path, **kwargs: Any) -> bytes:
        assert Path(path).resolve(strict=False) != sealed
        return original_secure(path, **kwargs)

    def path_open_guard(self: Path, *args: Any, **kwargs: Any):
        assert self.resolve(strict=False) != sealed
        return original_path_open(self, *args, **kwargs)

    def builtin_open_guard(file: Any, *args: Any, **kwargs: Any):
        if isinstance(file, (str, os.PathLike)):
            assert Path(file).resolve(strict=False) != sealed
        return original_builtin_open(file, *args, **kwargs)

    def os_open_guard(path: Any, *args: Any, **kwargs: Any):
        assert os.fspath(path) != "residual_sealed.jsonl"
        return original_os_open(path, *args, **kwargs)

    def listdir_guard(path: Any = "."):
        assert Path(path).resolve(strict=False) != private_root
        return original_listdir(path)

    def scandir_guard(path: Any = "."):
        assert Path(path).resolve(strict=False) != private_root
        return original_scandir(path)

    monkeypatch.setattr(custody, "_secure_read_bytes", secure_guard)
    monkeypatch.setattr(Path, "open", path_open_guard)
    monkeypatch.setattr(builtins, "open", builtin_open_guard)
    monkeypatch.setattr(os, "open", os_open_guard)
    monkeypatch.setattr(os, "listdir", listdir_guard)
    monkeypatch.setattr(os, "scandir", scandir_guard)
    source = tmp_path / "guarded-source"
    export_synthetic_development_source_view_for_tests(
        development_jsonl_path=inputs["development"],
        acquisition_receipt_path=inputs["acquisition"],
        formation_receipt_path=inputs["formation"],
        frozen_program_path=inputs["program"],
        source_view_root=source,
        public_receipt_path=tmp_path / "public" / "guarded-custody.json",
        public_private_index_binding_path=tmp_path / "public" / "guarded-private-binding.json",
    )


@pytest.mark.parametrize("module", (custody, freeze))
def test_cli_direct_script_help_from_foreign_cwd(module: Any, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(module.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
