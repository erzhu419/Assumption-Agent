from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from hegel_machine.phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    ExecutionFreezeManifest,
    frozen_phase2b_protocol,
)
import hegel_machine.phase2b_recognizer_runtime_cli_v2 as runtime_v2


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _freeze_manifest() -> ExecutionFreezeManifest:
    protocol = frozen_phase2b_protocol()
    exact_freeze = frozen_phase2b_exact_freeze()
    spec_id_by_kind = {
        BaselineKind(spec.baseline_id): spec.content_id
        for spec in exact_freeze.baselines
    }
    baselines = tuple(
        BaselineRegistration(
            kind=kind,
            baseline_spec_id=spec_id_by_kind[kind],
            implementation_id=f"baseline_{kind.value}",
            artifact_sha256=digest,
            frozen_before_holdout_generation=True,
        )
        for kind, digest in zip(BaselineKind, (SHA_A, SHA_B, SHA_C), strict=True)
    )
    return ExecutionFreezeManifest(
        protocol_id=protocol.protocol_id,
        exact_freeze_id=exact_freeze.freeze_id,
        git_commit="1" * 40,
        recognizer_image_digest="sha256:" + SHA_A,
        configuration_sha256=SHA_B,
        theory_version_id=initial_theory().version_id,
        adapter_implementation_sha256=SHA_C,
        selector_implementation_sha256=SHA_D,
        verifier_registry_sha256=SHA_A,
        baseline_registrations=baselines,
        isolation_profile_id=protocol.isolation_profile.profile_id,
    )


def _patch_mounts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, tuple[str, ...]]:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    monkeypatch.setattr(
        runtime_v2,
        "RECOGNIZER_INPUT_DIRECTORY_V2",
        str(input_directory),
    )
    monkeypatch.setattr(
        runtime_v2,
        "RECOGNIZER_OUTPUT_DIRECTORY_V2",
        str(output_directory),
    )
    argv = (
        "--input",
        str(input_directory),
        "--output",
        str(output_directory),
    )
    return input_directory, output_directory, argv


def _install_fake_recognizer(
    monkeypatch: pytest.MonkeyPatch,
    input_payload: bytes,
    prediction_payload: bytes,
) -> None:
    class FakeInputArchive:
        def __init__(self, payload: bytes) -> None:
            self.disposition = runtime_v2.RecognizerInputArchiveDispositionV2.COMPLETE
            self.archive = payload
            self.rows = (None,) * runtime_v2.TOTAL_RECOGNIZER_CASE_COUNT
            self.archive_id = "phase2b_recognizer_input_archive_v2_" + "a" * 64

    class FakePredictionArchive:
        def __init__(self, payload: bytes) -> None:
            self.disposition = runtime_v2.PredictionArchiveDispositionV2.COMPLETE
            self.archive = payload
            self.records = (None,) * runtime_v2.TOTAL_RECOGNIZER_CASE_COUNT
            self.archive_id = (
                "phase2b_recognizer_prediction_archive_v2_" + "b" * 64
            )

    input_value = FakeInputArchive(input_payload)
    prediction_value = FakePredictionArchive(prediction_payload)
    monkeypatch.setattr(
        runtime_v2,
        "DecodedRecognizerInputArchiveV2",
        FakeInputArchive,
    )
    monkeypatch.setattr(
        runtime_v2,
        "DecodedRecognizerPredictionArchiveV2",
        FakePredictionArchive,
    )

    def decode_input(payload: bytes) -> FakeInputArchive:
        assert payload == input_payload
        return input_value

    def build_prediction(
        *,
        input_archive: FakeInputArchive,
        execution_freeze_manifest: ExecutionFreezeManifest,
    ) -> FakePredictionArchive:
        assert input_archive is input_value
        assert execution_freeze_manifest == _freeze_manifest()
        return prediction_value

    def decode_prediction(payload: bytes) -> FakePredictionArchive:
        assert payload == prediction_payload
        return prediction_value

    monkeypatch.setattr(
        runtime_v2,
        "decode_public_recognizer_input_archive_v2",
        decode_input,
    )
    monkeypatch.setattr(
        runtime_v2,
        "build_recognizer_prediction_archive_v2",
        build_prediction,
    )
    monkeypatch.setattr(
        runtime_v2,
        "decode_public_recognizer_prediction_archive_v2",
        decode_prediction,
    )


def _write_valid_inputs(input_directory: Path, input_payload: bytes) -> None:
    (input_directory / runtime_v2.RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2).write_bytes(
        input_payload
    )
    (
        input_directory / runtime_v2.EXECUTION_FREEZE_MANIFEST_FILENAME_V2
    ).write_bytes(
        runtime_v2.encode_execution_freeze_manifest_json_v2(_freeze_manifest())
    )


def test_frozen_mount_arguments_and_filenames_are_exact() -> None:
    assert runtime_v2.RECOGNIZER_INPUT_DIRECTORY_V2 == "/phase2b/input"
    assert runtime_v2.RECOGNIZER_OUTPUT_DIRECTORY_V2 == "/phase2b/output"
    assert runtime_v2.RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2 == (
        "recognizer_input_archive_v2.bin"
    )
    assert runtime_v2.EXECUTION_FREEZE_MANIFEST_FILENAME_V2 == (
        "execution_freeze_manifest.json"
    )
    assert runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2 == (
        "recognizer_prediction_archive_v2.bin"
    )


def test_execution_freeze_manifest_codec_is_exact_closed_and_canonical() -> None:
    manifest = _freeze_manifest()
    encoded = runtime_v2.encode_execution_freeze_manifest_json_v2(manifest)
    assert encoded == encoded.strip()
    assert runtime_v2.decode_execution_freeze_manifest_json_v2(encoded) == manifest

    with pytest.raises(ValueError, match="canonical"):
        runtime_v2.decode_execution_freeze_manifest_json_v2(b"\n" + encoded)
    duplicate = b'{"protocol_id":"duplicate",' + encoded[1:]
    with pytest.raises(ValueError, match="repeats"):
        runtime_v2.decode_execution_freeze_manifest_json_v2(duplicate)
    extra = json.loads(encoded)
    extra["unexpected"] = False
    with pytest.raises(ValueError, match="closed"):
        runtime_v2.decode_execution_freeze_manifest_json_v2(
            json.dumps(extra, sort_keys=True, separators=(",", ":")).encode()
        )
    wrong_bool = json.loads(encoded)
    wrong_bool["baseline_registrations"][0][
        "frozen_before_holdout_generation"
    ] = 1
    with pytest.raises(TypeError, match="exact bool"):
        runtime_v2.decode_execution_freeze_manifest_json_v2(
            json.dumps(wrong_bool, sort_keys=True, separators=(",", ":")).encode()
        )
    with pytest.raises(TypeError, match="exact execution freeze"):
        runtime_v2.encode_execution_freeze_manifest_json_v2(
            object()  # type: ignore[arg-type]
        )


def test_success_publishes_once_read_only_and_emits_canonical_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory, output_directory, argv = _patch_mounts(monkeypatch, tmp_path)
    input_payload = b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2
    prediction_payload = b"P" * runtime_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2
    _write_valid_inputs(input_directory, input_payload)
    _install_fake_recognizer(monkeypatch, input_payload, prediction_payload)

    assert runtime_v2.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    receipt = json.loads(captured.out)
    assert captured.out == json.dumps(
        receipt,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    assert receipt["disposition"] == "COMPLETE"
    assert receipt["prediction_case_count"] == 960
    assert receipt["output_filename"] == (
        runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
    )
    output = output_directory / runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
    assert output.read_bytes() == prediction_payload
    # DrvFs may synthesize execute bits when Linux metadata is disabled; the
    # security property is that no write bit survives publication.
    assert stat.S_IMODE(output.stat().st_mode) & 0o222 == 0
    assert output.stat().st_nlink == 1
    assert not (
        output_directory / runtime_v2._PENDING_OUTPUT_FILENAME_V2
    ).exists()
    assert frozenset(path.name for path in output_directory.iterdir()) == frozenset(
        {runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2}
    )


@pytest.mark.parametrize(
    "preexisting_name",
    [
        runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
        runtime_v2._PENDING_OUTPUT_FILENAME_V2,
        "unexpected-output.bin",
    ],
)
def test_nonempty_output_namespace_is_rejected_before_decode_and_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    preexisting_name: str,
) -> None:
    input_directory, output_directory, argv = _patch_mounts(monkeypatch, tmp_path)
    input_payload = b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2
    _write_valid_inputs(input_directory, input_payload)
    preexisting = output_directory / preexisting_name
    preexisting.write_bytes(b"preexisting-output")

    def forbidden_decode(payload: bytes) -> object:
        raise AssertionError("decoder must not run with a nonempty output namespace")

    monkeypatch.setattr(
        runtime_v2,
        "decode_public_recognizer_input_archive_v2",
        forbidden_decode,
    )

    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "disposition": "ABSTAIN",
        "reason": runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2,
    }
    assert preexisting.read_bytes() == b"preexisting-output"
    assert tuple(path.name for path in output_directory.iterdir()) == (
        preexisting_name,
    )


def test_early_fsync_failure_leaves_pending_and_no_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory, output_directory, argv = _patch_mounts(monkeypatch, tmp_path)
    input_payload = b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2
    prediction_payload = b"P" * runtime_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2
    _write_valid_inputs(input_directory, input_payload)
    _install_fake_recognizer(monkeypatch, input_payload, prediction_payload)

    def fail_fsync(descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(runtime_v2.os, "fsync", fail_fsync)
    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason"] == (
        runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2
    )
    assert (
        output_directory / runtime_v2._PENDING_OUTPUT_FILENAME_V2
    ).read_bytes() == prediction_payload
    assert not (
        output_directory / runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
    ).exists()


def test_late_fsync_failure_reports_abstain_with_final_only_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory, output_directory, argv = _patch_mounts(monkeypatch, tmp_path)
    input_payload = b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2
    prediction_payload = b"P" * runtime_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2
    _write_valid_inputs(input_directory, input_payload)
    _install_fake_recognizer(monkeypatch, input_payload, prediction_payload)

    real_fsync = os.fsync
    real_unlink = os.unlink
    pending_was_unlinked = False

    def track_pending_unlink(
        path: str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal pending_was_unlinked
        real_unlink(path, dir_fd=dir_fd)
        if path == runtime_v2._PENDING_OUTPUT_FILENAME_V2:
            pending_was_unlinked = True

    def fail_first_fsync_after_pending_unlink(descriptor: int) -> None:
        if pending_was_unlinked:
            raise OSError("injected late fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(runtime_v2.os, "unlink", track_pending_unlink)
    monkeypatch.setattr(runtime_v2.os, "fsync", fail_first_fsync_after_pending_unlink)

    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "disposition": "ABSTAIN",
        "reason": runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2,
    }
    assert pending_was_unlinked is True
    final = output_directory / runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
    assert final.read_bytes() == prediction_payload
    assert not (
        output_directory / runtime_v2._PENDING_OUTPUT_FILENAME_V2
    ).exists()
    assert frozenset(path.name for path in output_directory.iterdir()) == frozenset(
        {runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2}
    )


@pytest.mark.parametrize(
    ("missing_name", "extra_name"),
    [
        (None, "answer_archive_v2.bin"),
        (None, "gate_manifest.json"),
        (None, "holdout_generator_state.bin"),
        (runtime_v2.RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2, None),
        (runtime_v2.EXECUTION_FREEZE_MANIFEST_FILENAME_V2, None),
    ],
)
def test_input_namespace_requires_exact_two_names_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    missing_name: str | None,
    extra_name: str | None,
) -> None:
    input_directory, output_directory, argv = _patch_mounts(monkeypatch, tmp_path)
    input_payload = b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2
    _write_valid_inputs(input_directory, input_payload)
    if missing_name is not None:
        (input_directory / missing_name).unlink()
    if extra_name is not None:
        (input_directory / extra_name).write_bytes(b"not-public-recognizer-input")

    def forbidden_decode(payload: bytes) -> object:
        raise AssertionError("decoder must not see a non-exact input namespace")

    monkeypatch.setattr(
        runtime_v2,
        "decode_public_recognizer_input_archive_v2",
        forbidden_decode,
    )

    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason"] == (
        runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2
    )
    assert tuple(output_directory.iterdir()) == ()


def test_input_and_output_directory_inode_alias_is_rejected_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir()
    shared_path = str(shared_directory)
    monkeypatch.setattr(runtime_v2, "RECOGNIZER_INPUT_DIRECTORY_V2", shared_path)
    monkeypatch.setattr(runtime_v2, "RECOGNIZER_OUTPUT_DIRECTORY_V2", shared_path)
    input_payload = b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2
    _write_valid_inputs(shared_directory, input_payload)

    def forbidden_decode(payload: bytes) -> object:
        raise AssertionError("decoder must not run with aliased mount directories")

    monkeypatch.setattr(
        runtime_v2,
        "decode_public_recognizer_input_archive_v2",
        forbidden_decode,
    )
    argv = ("--input", shared_path, "--output", shared_path)

    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason"] == (
        runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2
    )
    assert not (
        shared_directory / runtime_v2.RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
    ).exists()
    assert not (
        shared_directory / runtime_v2._PENDING_OUTPUT_FILENAME_V2
    ).exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_input_archive_aliases_fail_before_decode_without_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    link_kind: str,
) -> None:
    input_directory, output_directory, argv = _patch_mounts(monkeypatch, tmp_path)
    target = tmp_path / "aliased-input.bin"
    target.write_bytes(b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2)
    archive_path = input_directory / runtime_v2.RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2
    if link_kind == "symlink":
        archive_path.symlink_to(target)
    else:
        os.link(target, archive_path)
    (
        input_directory / runtime_v2.EXECUTION_FREEZE_MANIFEST_FILENAME_V2
    ).write_bytes(
        runtime_v2.encode_execution_freeze_manifest_json_v2(_freeze_manifest())
    )

    def forbidden_decode(payload: bytes) -> object:
        raise AssertionError("decoder must not see an aliased input")

    monkeypatch.setattr(
        runtime_v2,
        "decode_public_recognizer_input_archive_v2",
        forbidden_decode,
    )
    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason"] == (
        runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2
    )
    assert tuple(output_directory.iterdir()) == ()


@pytest.mark.parametrize("mutation_kind", ("archive_bytes", "namespace_churn"))
def test_input_mutation_during_builder_is_rejected_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation_kind: str,
) -> None:
    input_directory, output_directory, argv = _patch_mounts(monkeypatch, tmp_path)
    input_payload = b"I" * runtime_v2.ARCHIVE_HEADER_BYTES_V2
    prediction_payload = b"P" * runtime_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2
    _write_valid_inputs(input_directory, input_payload)
    _install_fake_recognizer(monkeypatch, input_payload, prediction_payload)
    fake_builder = runtime_v2.build_recognizer_prediction_archive_v2

    def mutating_builder(**kwargs: object) -> object:
        result = fake_builder(**kwargs)
        if mutation_kind == "archive_bytes":
            (
                input_directory / runtime_v2.RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2
            ).write_bytes(b"J" * len(input_payload))
        else:
            transient = input_directory / "transient-answer-manifest.json"
            transient.write_bytes(b"forbidden")
            transient.unlink()
        return result

    monkeypatch.setattr(
        runtime_v2,
        "build_recognizer_prediction_archive_v2",
        mutating_builder,
    )
    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason"] == (
        runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2
    )
    assert tuple(output_directory.iterdir()) == ()


def test_wrong_argument_order_is_atomic_generic_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = (
        "--output",
        runtime_v2.RECOGNIZER_OUTPUT_DIRECTORY_V2,
        "--input",
        runtime_v2.RECOGNIZER_INPUT_DIRECTORY_V2,
    )
    assert runtime_v2.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "disposition": "ABSTAIN",
        "reason": runtime_v2.RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2,
    }
