"""Adversarial tests for the read-only strict V2 structural verifier CLI.

The paired 960-row objects built here are synthetic codec fixtures only.  They
are not evidence of an actual recognizer run, runtime, capacity, scoring,
effect, or C1 exit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

import hegel_machine.cli as top_cli
import hegel_machine.phase2b_recognizer_input_archive_v1 as input_v1
import hegel_machine.phase2b_recognizer_input_archive_v2 as input_v2
import hegel_machine.phase2b_recognizer_prediction_archive_v1 as prediction_v1
import hegel_machine.phase2b_recognizer_prediction_archive_v2 as prediction_v2
import hegel_machine.phase2b_strict_recognizer_cli_v2 as strict_cli
import hegel_machine.phase2b_unsealed_prediction_evaluator_v2 as evaluator_v2
from hegel_machine.phase2b_runner import TOTAL_RECOGNIZER_CASE_COUNT
from hegel_machine.phase2b_trusted_wire_v1 import NON_AUTHORITATIVE_CLAIM_LEVEL


COUNT = TOTAL_RECOGNIZER_CASE_COUNT
COMMAND = "phase2b-verify-v2-structure"
INPUT_BYTES = b"I" * input_v2.ARCHIVE_HEADER_BYTES_V2

SEVEN_ROOT_FIELDS = (
    "authority_content_id",
    "envelope_id",
    "namespace_audit_id",
    "padding_sha256",
    "payload_sha256",
    "public_registry_id",
    "transform_result_id",
)
FALSE_EVIDENCE_FIELDS = (
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "partition_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "scoring_performed",
    "prediction_scored",
    "effect_evidence",
    "c1_exit_evidence",
)


@dataclass(frozen=True, slots=True)
class _PairedSyntheticV2:
    """Unbacked public-decoder stubs plus canonical prediction wire bytes."""

    input_bytes: bytes
    input_decoded: input_v2.DecodedRecognizerInputArchiveV2
    prediction_bytes: bytes
    prediction_decoded: prediction_v2.DecodedRecognizerPredictionArchiveV2


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


@pytest.fixture(scope="module")
def paired_synthetic_v2() -> _PairedSyntheticV2:
    """Build one synthetic 960-row pair; never invoke a public recognizer."""

    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_recognizer_prediction_archive_v2.py"))
    )
    freeze = namespace["execution_freeze_manifest"].__wrapped__()
    indices = [1, 0, *range(2, COUNT)]
    rows = tuple(namespace["_synthetic_root_row"](index) for index in indices)
    row_ids = tuple(row.row_id for row in rows)
    input_archive_id = namespace["_hex_id"](
        "phase2b_recognizer_input_archive_v2_", 500_001
    )
    batch_id = namespace["_hex_id"]("phase2b_trusted_wire_batch_v2_", 500_002)
    context = prediction_v2.PublicPredictionRunContextV2._issue(
        prediction_v2._CONTEXT_ISSUE_TOKEN_V2,
        batch_id=batch_id,
        input_archive_id=input_archive_id,
        input_archive_sha256=hashlib.sha256(INPUT_BYTES).hexdigest(),
        input_row_ids_root=prediction_v2._input_row_ids_root_v2(row_ids),
        execution_freeze_manifest_id=freeze.manifest_id,
    )
    records = tuple(
        prediction_v2.PublicRecognizerPredictionRecordV2._issue(
            prediction_v2._RECORD_ISSUE_TOKEN_V2,
            context=context,
            input_row=row,
            outcome=namespace["_outcome"](
                row_id=row.row_id,
                payload_sha256=row.payload_sha256,
                freeze_manifest_id=freeze.manifest_id,
                index=position,
            ),
        )
        for position, row in enumerate(rows)
    )
    prediction_bytes = prediction_v2._encode_prediction_archive_v2(
        context=context,
        records=records,
    )
    prediction_decoded = (
        prediction_v2.decode_public_recognizer_prediction_archive_v2(
            prediction_bytes
        )
    )
    base_input = namespace["_shallow_unbacked_input_archive"]()
    input_decoded = _unchecked_copy(
        base_input,
        archive=INPUT_BYTES,
        archive_id=input_archive_id,
        batch_id=batch_id,
        rows=rows,
        row_ids=row_ids,
        envelope_ids=tuple(row.envelope_id for row in rows),
        public_registry_ids=tuple(row.public_registry_id for row in rows),
        authority_content_ids=tuple(row.authority_content_id for row in rows),
        transform_result_ids=tuple(row.transform_result_id for row in rows),
    )
    return _PairedSyntheticV2(
        input_bytes=INPUT_BYTES,
        input_decoded=input_decoded,
        prediction_bytes=prediction_bytes,
        prediction_decoded=prediction_decoded,
    )


def _canonical_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def _write_pair(
    root: Path,
    pair: _PairedSyntheticV2,
) -> tuple[Path, Path]:
    input_path = root / "input-v2.bin"
    prediction_path = root / "prediction-v2.bin"
    input_path.write_bytes(pair.input_bytes)
    prediction_path.write_bytes(pair.prediction_bytes)
    return input_path, prediction_path


def test_public_surface_is_narrow_v2_only_and_has_no_mutating_arguments() -> None:
    assert strict_cli.__all__ == [
        "STRICT_RECOGNIZER_CLI_V2_COMMAND",
        "STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION",
        "STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID",
        "STRICT_RECOGNIZER_CLI_V2_POLICY_ID",
        "STRICT_RECOGNIZER_CLI_V2_GENERIC_REJECTION_REASON",
        "StrictRecognizerCliDispositionV2",
        "StrictRecognizerStructuralReceiptV2",
        "StrictRecognizerStructuralRejectionV2",
        "verify_strict_recognizer_io_structure_v2",
        "main",
    ]
    assert strict_cli.STRICT_RECOGNIZER_CLI_V2_COMMAND == COMMAND
    assert strict_cli.STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID == (
        "phase2b_strict_recognizer_cli_schema_v2_"
        "b5c47e3a850e47ca5b35adb8fd2dcd2b520962e51d63be47ebc0d87f4749ef42"
    )
    assert strict_cli.STRICT_RECOGNIZER_CLI_V2_POLICY_ID == (
        "phase2b_strict_recognizer_cli_policy_v2_"
        "b1f611259a8aa747b7c7beda7fad5f47bf9f751788f3eb80806915aaa24d623e"
    )
    verify = inspect.signature(strict_cli.verify_strict_recognizer_io_structure_v2)
    assert tuple(verify.parameters) == ("input_archive", "prediction_archive")
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in verify.parameters.values()
    )
    assert tuple(inspect.signature(strict_cli.main).parameters) == ("argv",)
    source = inspect.getsource(strict_cli)
    for forbidden in (
        "build_recognizer_prediction_archive_v2",
        "recognize_public_input_row_v2",
        "phase2b_unsealed_prediction_evaluator",
        "phase2b_evaluator",
        "docker",
        "scorer",
        "answer_key",
        "main_row_ids",
        "semantic_conflict_row_ids",
    ):
        assert forbidden not in source

    assert tuple(item.name for item in fields(strict_cli.StrictRecognizerStructuralReceiptV2)) == (
        "disposition",
        "reason",
        "schema_version",
        "policy_id",
        "claim_level",
        "receipt_id",
        "input_archive_id",
        "input_archive_sha256",
        "input_archive_version",
        "input_archive_policy_id",
        "prediction_archive_id",
        "prediction_archive_sha256",
        "prediction_archive_version",
        "prediction_archive_policy_id",
        "batch_id",
        "batch_policy_id",
        "run_context_id",
        "execution_freeze_manifest_id",
        "protocol_id",
        "case_count",
        "structural_input_archive_verified",
        "structural_prediction_archive_verified",
        "cross_archive_context_binding_verified",
        "ordered_row_identity_verified",
        "seven_input_root_columns_positionally_verified",
        "metric_results",
        "scored_rows",
        *FALSE_EVIDENCE_FIELDS,
    )
    assert tuple(item.name for item in fields(strict_cli.StrictRecognizerStructuralRejectionV2)) == (
        "disposition",
        "reason",
        "schema_version",
        "policy_id",
        "claim_level",
        "receipt",
        "metric_results",
        "scored_rows",
        "partial_output_published",
        *FALSE_EVIDENCE_FIELDS,
    )
    assert tuple(strict_cli.StrictRecognizerCliDispositionV2) == (
        strict_cli.StrictRecognizerCliDispositionV2.COMPLETE,
        strict_cli.StrictRecognizerCliDispositionV2.ABSTAIN,
    )


def test_pure_bytes_api_binds_exact_ordered_seven_roots_and_false_claims(
    monkeypatch: pytest.MonkeyPatch,
    paired_synthetic_v2: _PairedSyntheticV2,
) -> None:
    calls = {"input": 0, "prediction": 0}

    def input_decode(value: bytes) -> input_v2.DecodedRecognizerInputArchiveV2:
        calls["input"] += 1
        assert value is paired_synthetic_v2.input_bytes
        return paired_synthetic_v2.input_decoded

    def prediction_decode(
        value: bytes,
    ) -> prediction_v2.DecodedRecognizerPredictionArchiveV2:
        calls["prediction"] += 1
        assert value is paired_synthetic_v2.prediction_bytes
        return paired_synthetic_v2.prediction_decoded

    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        input_decode,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        prediction_decode,
    )

    receipt = strict_cli.verify_strict_recognizer_io_structure_v2(
        input_archive=paired_synthetic_v2.input_bytes,
        prediction_archive=paired_synthetic_v2.prediction_bytes,
    )
    assert type(receipt) is strict_cli.StrictRecognizerStructuralReceiptV2
    assert calls == {"input": 1, "prediction": 1}
    assert receipt.disposition is strict_cli.StrictRecognizerCliDispositionV2.COMPLETE
    assert receipt.claim_level == NON_AUTHORITATIVE_CLAIM_LEVEL
    assert receipt.case_count == COUNT == 960
    assert receipt.structural_input_archive_verified is True
    assert receipt.structural_prediction_archive_verified is True
    assert receipt.cross_archive_context_binding_verified is True
    assert receipt.ordered_row_identity_verified is True
    assert receipt.seven_input_root_columns_positionally_verified is True
    assert receipt.metric_results == ()
    assert receipt.scored_rows == ()
    for name in FALSE_EVIDENCE_FIELDS:
        assert getattr(receipt, name) is False


@pytest.mark.parametrize(
    ("side", "mutation"),
    (
        ("input", "row_id_subclass"),
        ("input", "row_root_subclass"),
        ("prediction", "input_row_id_subclass"),
        ("prediction", "context_scalar_subclass"),
        ("prediction", "record_root_subclass"),
    ),
)
def test_decoded_wrapper_scalars_close_before_archive_hash_or_hostile_compare(
    monkeypatch: pytest.MonkeyPatch,
    paired_synthetic_v2: _PairedSyntheticV2,
    side: str,
    mutation: str,
) -> None:
    class HostileText(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile scalar reached equality")

    input_decoded = paired_synthetic_v2.input_decoded
    prediction_decoded = paired_synthetic_v2.prediction_decoded
    if mutation == "row_id_subclass":
        input_decoded = _unchecked_copy(
            input_decoded,
            row_ids=(
                HostileText(input_decoded.row_ids[0]),
                *input_decoded.row_ids[1:],
            ),
        )
    elif mutation == "row_root_subclass":
        first = input_decoded.rows[0]
        polluted = _unchecked_copy(
            first,
            payload_sha256=HostileText(first.payload_sha256),
        )
        input_decoded = _unchecked_copy(
            input_decoded,
            rows=(polluted, *input_decoded.rows[1:]),
        )
    elif mutation == "input_row_id_subclass":
        prediction_decoded = _unchecked_copy(
            prediction_decoded,
            input_row_ids=(
                HostileText(prediction_decoded.input_row_ids[0]),
                *prediction_decoded.input_row_ids[1:],
            ),
        )
    elif mutation == "context_scalar_subclass":
        context = _unchecked_copy(
            prediction_decoded.context,
            input_archive_id=HostileText(
                prediction_decoded.context.input_archive_id
            ),
        )
        prediction_decoded = _unchecked_copy(
            prediction_decoded,
            context=context,
        )
    else:
        first_record = prediction_decoded.records[0]
        polluted_record = _unchecked_copy(
            first_record,
            input_payload_sha256=HostileText(
                first_record.input_payload_sha256
            ),
        )
        prediction_decoded = _unchecked_copy(
            prediction_decoded,
            records=(polluted_record, *prediction_decoded.records[1:]),
        )

    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        lambda value: input_decoded,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        lambda value: prediction_decoded,
    )

    class ForbiddenBoundary(BaseException):
        pass

    def forbidden_hash(value: bytes = b"") -> object:
        raise ForbiddenBoundary("invalid decoded wrapper reached archive hash")

    monkeypatch.setattr(strict_cli.hashlib, "sha256", forbidden_hash)
    with pytest.raises((TypeError, ValueError)):
        strict_cli.verify_strict_recognizer_io_structure_v2(
            input_archive=paired_synthetic_v2.input_bytes,
            prediction_archive=paired_synthetic_v2.prediction_bytes,
        )


@pytest.mark.parametrize("field_name", SEVEN_ROOT_FIELDS)
def test_each_input_root_column_is_bound_positionally(
    monkeypatch: pytest.MonkeyPatch,
    paired_synthetic_v2: _PairedSyntheticV2,
    field_name: str,
) -> None:
    input_decoded = paired_synthetic_v2.input_decoded
    first = input_decoded.rows[0]
    prefixes = {
        "authority_content_id": "phase2b_public_transform_evidence_",
        "envelope_id": "phase2b_trusted_envelope_v2_",
        "namespace_audit_id": "phase2b_namespace_audit_v2_",
        "public_registry_id": "phase2b_public_recognizer_registry_v2_",
        "transform_result_id": "phase2b_exact_transform_result_",
    }
    changed = prefixes.get(field_name, "") + "f" * 64
    polluted = _unchecked_copy(first, **{field_name: changed})
    supplied = _unchecked_copy(
        input_decoded,
        rows=(polluted, *input_decoded.rows[1:]),
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        lambda value: supplied,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        lambda value: paired_synthetic_v2.prediction_decoded,
    )
    with pytest.raises(ValueError, match="positional binding drift"):
        strict_cli.verify_strict_recognizer_io_structure_v2(
            input_archive=paired_synthetic_v2.input_bytes,
            prediction_archive=paired_synthetic_v2.prediction_bytes,
        )


def test_cli_success_is_one_canonical_stdout_line_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    paired_synthetic_v2: _PairedSyntheticV2,
) -> None:
    input_path, prediction_path = _write_pair(tmp_path, paired_synthetic_v2)
    before = {
        path: (path.stat().st_mode, path.read_bytes())
        for path in (input_path, prediction_path)
    }
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        lambda value: paired_synthetic_v2.input_decoded,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        lambda value: paired_synthetic_v2.prediction_decoded,
    )

    assert strict_cli.main(
        [
            "--input-archive",
            str(input_path),
            "--prediction-archive",
            str(prediction_path),
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert captured.out == _canonical_line(payload)
    assert captured.out.count("\n") == 1
    assert set(tmp_path.iterdir()) == {input_path, prediction_path}
    assert {
        path: (path.stat().st_mode, path.read_bytes())
        for path in (input_path, prediction_path)
    } == before


def test_cli_reads_both_bounded_files_before_verify_and_writes_stdout_once(
    monkeypatch: pytest.MonkeyPatch,
    paired_synthetic_v2: _PairedSyntheticV2,
) -> None:
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        lambda value: paired_synthetic_v2.input_decoded,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        lambda value: paired_synthetic_v2.prediction_decoded,
    )
    receipt = strict_cli.verify_strict_recognizer_io_structure_v2(
        input_archive=paired_synthetic_v2.input_bytes,
        prediction_archive=paired_synthetic_v2.prediction_bytes,
    )
    reads: list[tuple[str, int, int]] = []

    def read(path: str, *, minimum_bytes: int, maximum_bytes: int) -> bytes:
        reads.append((path, minimum_bytes, maximum_bytes))
        if len(reads) == 1:
            return paired_synthetic_v2.input_bytes
        return paired_synthetic_v2.prediction_bytes

    def verify(*, input_archive: bytes, prediction_archive: bytes) -> object:
        assert [item[0] for item in reads] == ["/input-v2", "/prediction-v2"]
        assert input_archive is paired_synthetic_v2.input_bytes
        assert prediction_archive is paired_synthetic_v2.prediction_bytes
        return receipt

    class _Sink:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def write(self, value: str) -> int:
            self.writes.append(value)
            return len(value)

    sink = _Sink()
    monkeypatch.setattr(strict_cli, "_read_bounded_regular_file_v2", read)
    monkeypatch.setattr(
        strict_cli,
        "verify_strict_recognizer_io_structure_v2",
        verify,
    )
    monkeypatch.setattr(strict_cli.sys, "stdout", sink)
    assert strict_cli.main(
        [
            "--input-archive",
            "/input-v2",
            "--prediction-archive",
            "/prediction-v2",
        ]
    ) == 0
    assert reads == [
        (
            "/input-v2",
            input_v2.ARCHIVE_HEADER_BYTES_V2,
            input_v2.MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
        ),
        (
            "/prediction-v2",
            prediction_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2,
            prediction_v2.MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
        ),
    ]
    assert sink.writes == [_canonical_line(receipt.to_mapping())]


@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("--input-archive", "/tmp/input"),
        ("--prediction-archive", "/tmp/prediction"),
        ("--input-archive", "relative", "--prediction-archive", "/tmp/prediction"),
        ("--input-archive", "/tmp/input", "--prediction-archive", "relative"),
        ("--input-archive", "/tmp/input", "--prediction-archive", "/tmp/prediction", "extra"),
        ("--input-archive", "-", "--prediction-archive", "/tmp/prediction"),
    ),
)
def test_cli_usage_is_generic_canonical_exit_two(
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
) -> None:
    assert strict_cli.main(list(arguments)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert captured.err == _canonical_line(payload)
    assert captured.err.count("\n") == 1
    assert payload["disposition"] == "ABSTAIN"
    assert payload["partial_output_published"] is False
    assert "/tmp/input" not in captured.err
    assert "/tmp/prediction" not in captured.err
    assert "relative" not in captured.err


@pytest.mark.parametrize("target_kind", ("directory", "symlink"))
def test_cli_rejects_nonregular_or_symlink_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    paired_synthetic_v2: _PairedSyntheticV2,
    target_kind: str,
) -> None:
    input_path, prediction_path = _write_pair(tmp_path, paired_synthetic_v2)
    target = tmp_path / "bad-input"
    if target_kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(input_path)

    class ForbiddenBoundary(BaseException):
        pass

    def forbidden(value: bytes) -> object:
        raise ForbiddenBoundary("invalid path reached public decoder")

    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        forbidden,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden,
    )
    assert strict_cli.main(
        [
            "--input-archive",
            str(target),
            "--prediction-archive",
            str(prediction_path),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["disposition"] == "ABSTAIN"
    assert str(target) not in captured.err


def test_cli_rejects_pre_post_fd_fingerprint_drift_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    paired_synthetic_v2: _PairedSyntheticV2,
) -> None:
    input_path, prediction_path = _write_pair(tmp_path, paired_synthetic_v2)
    actual = strict_cli._stat_fingerprint_v2
    calls = 0

    def drifting(value: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        fingerprint = actual(value)
        if calls == 2:
            return (*fingerprint[:-1], fingerprint[-1] + 1)
        return fingerprint

    class ForbiddenBoundary(BaseException):
        pass

    def forbidden(value: bytes) -> object:
        raise ForbiddenBoundary("unstable file reached public decoder")

    monkeypatch.setattr(strict_cli, "_stat_fingerprint_v2", drifting)
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        forbidden,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden,
    )
    assert strict_cli.main(
        [
            "--input-archive",
            str(input_path),
            "--prediction-archive",
            str(prediction_path),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == _canonical_line(json.loads(captured.err))
    assert str(input_path) not in captured.err


@pytest.mark.parametrize("which", ("input", "prediction"))
def test_cli_rejects_hardlinked_archive_before_any_public_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    paired_synthetic_v2: _PairedSyntheticV2,
    which: str,
) -> None:
    input_path, prediction_path = _write_pair(tmp_path, paired_synthetic_v2)
    target = input_path if which == "input" else prediction_path
    os.link(target, tmp_path / f"{which}-alias.bin")

    class ForbiddenBoundary(BaseException):
        pass

    def forbidden(value: bytes) -> object:
        raise ForbiddenBoundary("hardlink reached public decoder")

    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        forbidden,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden,
    )
    assert strict_cli.main(
        [
            "--input-archive",
            str(input_path),
            "--prediction-archive",
            str(prediction_path),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    rejection = json.loads(captured.err)
    assert captured.err == _canonical_line(rejection)
    assert rejection["disposition"] == "ABSTAIN"
    assert rejection["partial_output_published"] is False
    assert str(target) not in captured.err


def test_top_level_module_subprocess_failure_is_generic_and_nonmutating(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "empty-input.bin"
    prediction_path = tmp_path / "empty-prediction.bin"
    input_path.write_bytes(b"")
    prediction_path.write_bytes(b"")
    before = tuple(sorted(path.name for path in tmp_path.iterdir()))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hegel_machine",
            COMMAND,
            "--input-archive",
            str(input_path),
            "--prediction-archive",
            str(prediction_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
        env=environment,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert completed.stderr == _canonical_line(payload)
    assert str(input_path) not in completed.stderr
    assert str(prediction_path) not in completed.stderr
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == before


@pytest.mark.parametrize(
    ("which", "size"),
    (
        ("input", 0),
        ("input", input_v2.MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2 + 1),
        ("prediction", 0),
        ("prediction", prediction_v2.MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2 + 1),
    ),
)
def test_cli_closes_both_file_caps_before_any_public_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    paired_synthetic_v2: _PairedSyntheticV2,
    which: str,
    size: int,
) -> None:
    input_path, prediction_path = _write_pair(tmp_path, paired_synthetic_v2)
    target = input_path if which == "input" else prediction_path
    with target.open("r+b") as stream:
        stream.truncate(size)

    class ForbiddenBoundary(BaseException):
        pass

    def forbidden(value: bytes) -> object:
        raise ForbiddenBoundary("out-of-cap file reached public decoder")

    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_input_archive_v2",
        forbidden,
    )
    monkeypatch.setattr(
        strict_cli,
        "decode_public_recognizer_prediction_archive_v2",
        forbidden,
    )
    assert strict_cli.main(
        [
            "--input-archive",
            str(input_path),
            "--prediction-archive",
            str(prediction_path),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["disposition"] == "ABSTAIN"


def test_v1_bytes_cross_reject_without_calling_any_v1_decoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input-v1.bin"
    prediction_path = tmp_path / "prediction-v1.bin"
    input_path.write_bytes(input_v1.ARCHIVE_MAGIC + b"\x00" * 128)
    prediction_path.write_bytes(prediction_v1.PREDICTION_ARCHIVE_MAGIC + b"\x00" * 128)

    class ForbiddenBoundary(BaseException):
        pass

    def forbidden(*args: object, **kwargs: object) -> object:
        raise ForbiddenBoundary("strict V2 CLI called a V1/private/deep API")

    for target, name in (
        (input_v1, "decode_public_recognizer_input_archive_v1"),
        (prediction_v1, "decode_public_recognizer_prediction_archive_v1"),
        (prediction_v2, "build_recognizer_prediction_archive_v2"),
        (evaluator_v2, "evaluate_unsealed_prediction_archive_structure_v2"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    assert strict_cli.main(
        [
            "--input-archive",
            str(input_path),
            "--prediction-archive",
            str(prediction_path),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["disposition"] == "ABSTAIN"


def test_top_level_cli_early_dispatches_raw_strict_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = [
        COMMAND,
        "--input-archive",
        "/absolute/input",
        "--prediction-archive",
        "/absolute/prediction",
    ]
    seen: list[tuple[str, ...]] = []

    def delegated(argv: tuple[str, ...]) -> int:
        seen.append(argv)
        return 17

    # Early dispatch imports the current module symbol inside top_cli.main.
    monkeypatch.setattr(strict_cli, "main", delegated)
    assert top_cli.main(supplied) == 17
    assert seen == [tuple(supplied[1:])]
