from __future__ import annotations

import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_formal_acquisition_entrypoint_v1 as module,
)


FREEZE_SHA = "1" * 64
QUALIFICATION_SHA = "2" * 64
ACQUISITION_SHA = "3" * 64


def _freeze() -> dict[str, object]:
    return {
        "implementation_freeze_sha256": FREEZE_SHA,
        "identity_compiler_qualification_sha256": QUALIFICATION_SHA,
    }


def _receipt() -> dict[str, object]:
    return {
        "status": "all_four_train_blocks_acquired_before_any_action_or_outcome",
        "implementation_freeze_sha256": FREEZE_SHA,
        "identity_full_compile_equivalence_qualification_sha256": (
            QUALIFICATION_SHA
        ),
        "acquisition_receipt_sha256": ACQUISITION_SHA,
        "block_counts": {"A_form": 96, "F_search": 48, "A_hold": 72, "M_search": 72},
        "all_blocks_one_acquisition": True,
        "action_retrieval_utility_or_evaluator_calls": 0,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
        # The entry point must never copy this aggregate secret digest to stdout.
        "selection_secret_sha256": "4" * 64,
    }


def test_only_project_enters_and_verified_hashes_drive_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def verify(project: object) -> dict[str, object]:
        observed["verified_project"] = project
        return _freeze()

    monkeypatch.setattr(
        module.implementation_freeze,
        "verify_committed_implementation_freeze",
        verify,
    )

    def acquire(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return _receipt()

    monkeypatch.setattr(
        module.formal_acquisition, "perform_formal_acquisition_once", acquire
    )
    summary = module.run_from_committed_freeze(tmp_path)
    assert observed == {
        "verified_project": tmp_path,
        "project": tmp_path,
        "implementation_freeze_sha256": FREEZE_SHA,
        "identity_full_compile_equivalence_qualification_sha256": (
            QUALIFICATION_SHA
        ),
    }
    serialized = json.dumps(summary, sort_keys=True)
    assert "4" * 64 not in serialized
    assert summary["private_selection_secret_logged"] is False
    assert summary["acquisition_receipt_sha256"] == ACQUISITION_SHA


def test_main_prints_only_aggregate_summary_and_rejects_binding_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(module, "run_from_committed_freeze", lambda _project: {
        "schema": module.SUMMARY_SCHEMA,
        "status": "formal_acquisition_completed_from_committed_freeze",
        "entrypoint_summary_sha256": "5" * 64,
    })
    assert module._main(["--project", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == module.SUMMARY_SCHEMA
    with pytest.raises(SystemExit):
        module._parser().parse_args(
            [
                "--project",
                str(tmp_path),
                "--implementation-freeze-sha256",
                FREEZE_SHA,
            ]
        )


def test_acquisition_return_binding_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module.implementation_freeze,
        "verify_committed_implementation_freeze",
        lambda _project: _freeze(),
    )
    drifted = _receipt()
    drifted["implementation_freeze_sha256"] = "9" * 64
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
