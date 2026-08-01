from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import shutil
import tempfile

import pytest

from replication_runtime.gscl_narrative_extractor_v1 import contract
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_multi_pack_worker as multi,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    multi_pack_worker as support,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_qwen_runtime as qwen_runtime,
)


@pytest.fixture
def secure_tmp_path() -> Path:
    root = Path(
        tempfile.mkdtemp(
            prefix="gscl-closed-multi-",
            dir="/var/tmp",
        )
    )
    root.chmod(0o700)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _manifest(root: Path) -> tuple[Path, bytes]:
    batch = root / "batch"
    batch.mkdir(mode=0o700)
    input_path = batch / "input.json"
    input_raw = b'{"synthetic":"source-free"}\n'
    input_path.write_bytes(input_raw)
    input_path.chmod(0o600)
    body = {
        "batches": [
            {
                "input_file_sha256": hashlib.sha256(
                    input_raw
                ).hexdigest(),
                "input_path": str(input_path),
                "output_path": str(batch / "output.json"),
                "sequence": 0,
            }
        ],
        "lineage": "source_free_qualification",
        "schema": multi.INPUT_MANIFEST_SCHEMA,
        "work_root": str(root),
    }
    envelope = {
        **body,
        "self_sha256": hashlib.sha256(
            contract.canonical_json_bytes(
                body,
                newline=False,
            )
        ).hexdigest(),
    }
    raw = contract.canonical_json_bytes(envelope)
    path = root / "manifest.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return path, raw


def test_manifest_schema_reuses_only_secure_envelope(
    secure_tmp_path: Path,
) -> None:
    path, raw = _manifest(secure_tmp_path)
    root, batches, file_hash, lineage = (
        support._decode_manifest(path)  # noqa: SLF001
    )
    assert multi.INPUT_MANIFEST_SCHEMA == (
        support.INPUT_MANIFEST_SCHEMA
    )
    assert root == secure_tmp_path
    assert len(batches) == 1
    assert file_hash == hashlib.sha256(raw).hexdigest()
    assert lineage == "source_free_qualification"


def test_closed_choice_output_schema_is_mechanism_specific() -> None:
    assert "closed_choice" in multi.SAFE_RECEIPT_SCHEMA
    assert multi.SAFE_RECEIPT_SCHEMA != (
        support.SAFE_RECEIPT_SCHEMA
    )


def test_selection_commitment_binds_order_and_multiplicity() -> None:
    left = hashlib.sha256(b"left").hexdigest()
    right = hashlib.sha256(b"right").hexdigest()
    assert multi._selection_commitment((left, right)) != (  # noqa: SLF001
        multi._selection_commitment((right, left))  # noqa: SLF001
    )
    assert multi._selection_commitment((left, left)) != (  # noqa: SLF001
        multi._selection_commitment((left,))  # noqa: SLF001
    )


def test_selection_commitment_rejects_non_hash() -> None:
    with pytest.raises(
        multi.MultiPackWorkerError,
        match="closed_choice_selection_commitments_invalid",
    ):
        multi._selection_commitment(("not-a-hash",))  # noqa: SLF001


def test_batch_selection_commitment_binds_sequence_and_count() -> None:
    commitment = hashlib.sha256(b"selection").hexdigest()
    row = {
        "selection_receipt_commitment": commitment,
        "selection_receipt_count": 2,
        "sequence": 0,
    }
    assert multi._batch_selection_commitments_sha256(  # noqa: SLF001
        (row,)
    ) != multi._batch_selection_commitments_sha256(  # noqa: SLF001
        ({**row, "sequence": 1},)
    )
    assert multi._batch_selection_commitments_sha256(  # noqa: SLF001
        (row,)
    ) != multi._batch_selection_commitments_sha256(  # noqa: SLF001
        ({**row, "selection_receipt_count": 1},)
    )


def test_formal_entry_has_one_closed_runtime_and_no_injection() -> None:
    parameters = set(
        inspect.signature(
            multi.run_formal_multi_pack
        ).parameters
    )
    assert parameters == {
        "input_manifest_path",
        "model_root",
        "model_manifest_path",
        "safe_receipt_path",
    }
    source = inspect.getsource(
        multi.run_formal_multi_pack
    )
    assert source.count(
        "qwen_closed.LocalQwenClosedChoiceRuntime("
    ) == 1
    assert "qwen_closed.process_formal_pack(" in source
    assert "LocalQwenRuntime(" not in source
    assert ".generate(" not in source
    assert "runtime=" not in parameters
    assert "scorer=" not in parameters
    assert "parser=" not in parameters
    assert "predictions=" not in parameters


def test_formal_entry_binds_zero_generation_and_teacher_forcing() -> None:
    source = inspect.getsource(
        multi.run_formal_multi_pack
    )
    assert '"free_form_generation_count": 0' in source
    assert (
        '"teacher_forced_forward_log_softmax"' in source
    )
    assert "runtime._runtime_receipt_sha256" in source
    assert "runtime._double_run_receipt_sha256" in source
    assert (
        "contract.canonical_json_bytes(runtime_receipt)"
        in source
    )
    assert "selection_receipt_commitments_sha256" in source
    assert "_teacher_forced_backend_commitment(runtime)" in source


def test_runtime_exposes_one_read_only_selection_commitment_property() -> None:
    source = inspect.getsource(
        qwen_runtime.LocalQwenClosedChoiceRuntime
    )
    assert source.count(
        "def selection_receipt_commitments"
    ) == 1
    assert "return tuple(self._selection_receipt_commitments)" in source
