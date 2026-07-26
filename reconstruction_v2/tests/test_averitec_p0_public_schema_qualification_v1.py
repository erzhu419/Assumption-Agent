from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from assumption_agent.benchmarks.averitec_p0_public_schema_qualification_v1 import (
    AveritecP0QualificationError,
    qualify_source,
    write_receipt_exclusive,
)


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - Git object identity.
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def _row(
    *,
    claim: str,
    label: str,
    claim_type: str,
    question: str,
    answer: str,
    answer_type: str = "Extractive",
) -> dict[str, object]:
    return {
        "claim": claim,
        "required_reannotation": False,
        "label": label,
        "justification": "PRIVATE JUSTIFICATION",
        "claim_date": "2020-01-01",
        "speaker": "PRIVATE SPEAKER",
        "original_claim_url": "https://private.invalid/claim",
        "cached_original_claim_url": "",
        "fact_checking_article": "https://private.invalid/fact-check",
        "reporting_source": "PRIVATE REPORTER",
        "location_ISO_code": "ZZ",
        "claim_types": [claim_type],
        "fact_checking_strategies": ["Written Evidence"],
        "questions": [
            {
                "question": question,
                "answers": [
                    {
                        "answer": answer,
                        "answer_type": answer_type,
                        "source_url": "https://private.invalid/evidence",
                        "cached_source_url": "",
                        "source_medium": "web text",
                    }
                ],
            }
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    root = tmp_path / "source"
    (root / "data").mkdir(parents=True)
    payloads = {
        "train": [
            _row(
                claim="PRIVATE TRAIN CLAIM",
                label="Supported",
                claim_type="Causal Claim",
                question="PRIVATE TRAIN QUESTION",
                answer="PRIVATE TRAIN ANSWER",
            ),
            _row(
                claim="PRIVATE TRAIN QUOTE",
                label="Refuted",
                claim_type="Quote Verification",
                question="PRIVATE QUOTE QUESTION",
                answer="PRIVATE QUOTE ANSWER",
            ),
        ],
        "dev": [
            _row(
                claim="PRIVATE DEV CLAIM",
                label="Not Enough Evidence",
                claim_type="Numerical Claim",
                question="PRIVATE DEV QUESTION",
                answer="PRIVATE DEV ANSWER",
            )
        ],
    }
    bindings: dict[str, dict[str, object]] = {}
    for split, rows in payloads.items():
        raw = json.dumps(rows, ensure_ascii=False).encode("utf-8")
        relative = f"data/{split}.json"
        (root / relative).write_bytes(raw)
        bindings[split] = {
            "git_blob_sha1": _git_blob_sha1(raw),
            "relative_path": relative,
            "size_bytes": len(raw),
        }
    return root, bindings


def test_qualification_emits_aggregate_schema_without_raw_values(
    tmp_path: Path,
) -> None:
    root, bindings = _fixture(tmp_path)
    receipt = qualify_source(source_root=root, expected_files=bindings)
    rendered = json.dumps(receipt, sort_keys=True)

    assert receipt["status"] == "qualified_public_non_scoring_schema_topology"
    assert (
        receipt["split_receipts"]["train"]["exclusive_family_count"]
        == {"CAUSAL_CLAIM": 1, "QUOTE_VERIFICATION": 1}
    )
    assert (
        receipt["split_receipts"]["dev"]["exclusive_family_count"]
        == {"NUMERICAL_CLAIM": 1}
    )
    assert receipt["access_boundary"] == {
        "action_model_evaluator_qrel_or_score_count": 0,
        "individual_claim_question_answer_url_justification_or_identifier_output_count": 0,
        "private_cohort_or_secret_count": 0,
        "public_source_split_parse_count": 2,
    }
    for forbidden in (
        "PRIVATE TRAIN CLAIM",
        "PRIVATE TRAIN QUESTION",
        "PRIVATE TRAIN ANSWER",
        "PRIVATE JUSTIFICATION",
        "private.invalid",
    ):
        assert forbidden not in rendered


def test_qualification_rejects_git_blob_identity_mismatch(
    tmp_path: Path,
) -> None:
    root, bindings = _fixture(tmp_path)
    bindings["train"]["git_blob_sha1"] = "0" * 40
    with pytest.raises(
        AveritecP0QualificationError,
        match="Git blob identity",
    ):
        qualify_source(source_root=root, expected_files=bindings)


def test_receipt_is_exclusive_mode_0600_and_self_hashed(
    tmp_path: Path,
) -> None:
    mode_probe = tmp_path / "mode_probe"
    probe_fd = os.open(mode_probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(probe_fd)
    if stat.S_IMODE(mode_probe.stat().st_mode) != 0o600:
        pytest.skip("temporary filesystem does not preserve POSIX mode bits")

    root, bindings = _fixture(tmp_path)
    receipt = qualify_source(source_root=root, expected_files=bindings)
    output = tmp_path / "receipt.json"
    write_receipt_exclusive(output, receipt)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="ascii")) == receipt
    with pytest.raises(
        AveritecP0QualificationError,
        match="exclusively",
    ):
        write_receipt_exclusive(output, receipt)
