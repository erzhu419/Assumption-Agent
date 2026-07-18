from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_formal_acquisition_entrypoint_v2 as module,
)


ROLLOVER_SHA = "1" * 64
TRAIN_LOADER_QUALIFICATION_SHA = "2" * 64
FREEZE_SHA = "3" * 64
IDENTITY_QUALIFICATION_SHA = "4" * 64
ACQUISITION_SHA = "5" * 64
PRIVATE_SECRET_SENTINEL = "PRIVATE-SELECTION-SECRET-MUST-NOT-LOG"
PRIVATE_SECRET_SHA = hashlib.sha256(
    PRIVATE_SECRET_SENTINEL.encode("ascii")
).hexdigest()


def _epoch() -> dict[str, object]:
    return {
        "source_epoch_rollover_sha256": ROLLOVER_SHA,
        "real_train_loader_qualification": {
            "qualification_sha256": TRAIN_LOADER_QUALIFICATION_SHA,
        },
    }


def _freeze() -> dict[str, object]:
    return {
        "implementation_freeze_sha256": FREEZE_SHA,
        "identity_compiler_qualification_sha256": IDENTITY_QUALIFICATION_SHA,
        "source_epoch_rollover_sha256": ROLLOVER_SHA,
        "train_loader_qualification_sha256": TRAIN_LOADER_QUALIFICATION_SHA,
    }


def _receipt() -> dict[str, object]:
    return {
        "status": "all_four_train_blocks_acquired_before_any_action_or_outcome",
        "source_epoch": "feverous_p6_e2_formal_v2",
        "source_epoch_rollover_sha256": ROLLOVER_SHA,
        "train_loader_qualification_sha256": TRAIN_LOADER_QUALIFICATION_SHA,
        "implementation_freeze_sha256": FREEZE_SHA,
        "identity_full_compile_equivalence_qualification_sha256": (
            IDENTITY_QUALIFICATION_SHA
        ),
        "acquisition_receipt_sha256": ACQUISITION_SHA,
        "block_counts": {
            "A_form": 96,
            "F_search": 48,
            "A_hold": 72,
            "M_search": 72,
        },
        "all_blocks_one_acquisition": True,
        "action_retrieval_utility_or_evaluator_calls": 0,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
        # These synthetic private fields must never be copied into the summary.
        "selection_secret_sha256": PRIVATE_SECRET_SHA,
        "selection_secret_private_sentinel": PRIVATE_SECRET_SENTINEL,
    }


def _patch_successful_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    observed: dict[str, object],
) -> None:
    def verify_rollover(project: object) -> dict[str, object]:
        observed["rollover_project"] = project
        return _epoch()

    def verify_freeze(project: object) -> dict[str, object]:
        observed["freeze_project"] = project
        return _freeze()

    def acquire(**kwargs: object) -> dict[str, object]:
        observed["acquisition_kwargs"] = dict(kwargs)
        return _receipt()

    monkeypatch.setattr(module.rollover, "verify_rollover_manifest", verify_rollover)
    monkeypatch.setattr(
        module.implementation_freeze,
        "verify_committed_implementation_freeze",
        verify_freeze,
    )
    monkeypatch.setattr(
        module.formal_acquisition,
        "perform_formal_acquisition_once",
        acquire,
    )


def test_only_project_enters_and_exact_committed_bindings_drive_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}
    _patch_successful_prerequisites(monkeypatch, observed)

    signature = inspect.signature(module.run_from_committed_freeze)
    assert tuple(signature.parameters) == ("project",)
    summary = module.run_from_committed_freeze(tmp_path)

    assert observed == {
        "rollover_project": tmp_path,
        "freeze_project": tmp_path,
        "acquisition_kwargs": {
            "project": tmp_path,
            "source_epoch_rollover_sha256": ROLLOVER_SHA,
            "train_loader_qualification_sha256": TRAIN_LOADER_QUALIFICATION_SHA,
            "implementation_freeze_sha256": FREEZE_SHA,
            "identity_full_compile_equivalence_qualification_sha256": (
                IDENTITY_QUALIFICATION_SHA
            ),
        },
    }
    assert summary["source_epoch_rollover_sha256"] == ROLLOVER_SHA
    assert (
        summary["train_loader_qualification_sha256"]
        == TRAIN_LOADER_QUALIFICATION_SHA
    )
    assert summary["implementation_freeze_sha256"] == FREEZE_SHA
    assert (
        summary["identity_compiler_qualification_sha256"]
        == IDENTITY_QUALIFICATION_SHA
    )
    assert summary["acquisition_receipt_sha256"] == ACQUISITION_SHA
    assert summary["private_selection_secret_logged"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_epoch_rollover_sha256", "9" * 64),
        ("train_loader_qualification_sha256", "8" * 64),
        ("implementation_freeze_sha256", "not-a-sha256"),
        ("identity_compiler_qualification_sha256", "also-not-a-sha256"),
    ),
)
def test_freeze_mismatch_or_invalid_hash_fails_before_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    freeze = _freeze()
    freeze[field] = value
    acquisition_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module.rollover,
        "verify_rollover_manifest",
        lambda _project: _epoch(),
    )
    monkeypatch.setattr(
        module.implementation_freeze,
        "verify_committed_implementation_freeze",
        lambda _project: freeze,
    )
    monkeypatch.setattr(
        module.formal_acquisition,
        "perform_formal_acquisition_once",
        lambda **kwargs: acquisition_calls.append(dict(kwargs)) or _receipt(),
    )

    with pytest.raises(
        module.FeverousFormalAcquisitionEntrypointError,
        match="committed freeze lacks acquisition prerequisite hashes",
    ):
        module.run_from_committed_freeze(tmp_path)
    assert acquisition_calls == []


@pytest.mark.parametrize(
    "epoch",
    (
        {},
        {"source_epoch_rollover_sha256": ROLLOVER_SHA},
        {
            "source_epoch_rollover_sha256": "not-a-sha256",
            "real_train_loader_qualification": {
                "qualification_sha256": TRAIN_LOADER_QUALIFICATION_SHA,
            },
        },
    ),
)
def test_incomplete_rollover_or_loader_binding_fails_before_freeze_and_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    epoch: dict[str, object],
) -> None:
    downstream_calls: list[str] = []
    monkeypatch.setattr(
        module.rollover,
        "verify_rollover_manifest",
        lambda _project: epoch,
    )
    monkeypatch.setattr(
        module.implementation_freeze,
        "verify_committed_implementation_freeze",
        lambda _project: downstream_calls.append("freeze") or _freeze(),
    )
    monkeypatch.setattr(
        module.formal_acquisition,
        "perform_formal_acquisition_once",
        lambda **_kwargs: downstream_calls.append("acquisition") or _receipt(),
    )

    with pytest.raises(
        module.FeverousFormalAcquisitionEntrypointError,
        match="rollover lacks its qualification binding",
    ):
        module.run_from_committed_freeze(tmp_path)
    assert downstream_calls == []


def test_main_stdout_is_aggregate_only_and_parser_rejects_binding_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}
    _patch_successful_prerequisites(monkeypatch, observed)

    assert module._main(["--project", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["schema"] == module.SUMMARY_SCHEMA
    assert output["acquisition_receipt_sha256"] == ACQUISITION_SHA
    assert PRIVATE_SECRET_SENTINEL not in captured.out
    assert PRIVATE_SECRET_SENTINEL.encode("ascii").hex() not in captured.out
    assert PRIVATE_SECRET_SHA not in captured.out

    forbidden_arguments = (
        "--source-epoch-rollover-sha256",
        "--train-loader-qualification-sha256",
        "--implementation-freeze-sha256",
        "--identity-full-compile-equivalence-qualification-sha256",
        "--selection-secret",
    )
    for argument in forbidden_arguments:
        with pytest.raises(SystemExit):
            module._parser().parse_args(
                ["--project", str(tmp_path), argument, "attacker-controlled"]
            )
        capsys.readouterr()


def test_acquisition_return_binding_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module.rollover,
        "verify_rollover_manifest",
        lambda _project: _epoch(),
    )
    monkeypatch.setattr(
        module.implementation_freeze,
        "verify_committed_implementation_freeze",
        lambda _project: _freeze(),
    )
    drifted = _receipt()
    drifted["train_loader_qualification_sha256"] = "9" * 64
    monkeypatch.setattr(
        module.formal_acquisition,
        "perform_formal_acquisition_once",
        lambda **_kwargs: drifted,
    )

    with pytest.raises(
        module.FeverousFormalAcquisitionEntrypointError,
        match="completion binding drifted",
    ):
        module.run_from_committed_freeze(tmp_path)
