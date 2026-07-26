from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import stat

import pytest

from assumption_agent.benchmarks import maud_extraction_p1_acquisition_v1 as subject
from assumption_agent.benchmarks import maud_extraction_p1_download_v1 as download
from assumption_agent.benchmarks import maud_extraction_p1_source_v1 as source


def _prepared(split: str, blocks: tuple[str, ...]) -> source.PreparedSplit:
    contracts = []
    for ordinal, block in enumerate(blocks):
        items = tuple(
            source.PreparedItem(
                work_id=hashlib.sha256(
                    f"{split}-item-{ordinal}-{index}".encode()
                ).hexdigest(),
                question=f"question {index}",
                deal_point_type=source.DEAL_POINT_TYPES[index],
                family=source._TYPE_TO_FAMILY[source.DEAL_POINT_TYPES[index]],
                spans=(
                    None
                    if block == "F_search"
                    else (source.GoldSpan(0, 1, "x"),)
                ),
                merged_intervals=(
                    None if block == "F_search" else ((0, 1),)
                ),
            )
            for index in range(22)
        )
        context = f"{split}-{ordinal}-" + "x" * 5000
        contracts.append(
            source.PreparedContract(
                split=split,
                block=block,
                work_id=hashlib.sha256(
                    f"{split}-contract-{ordinal}".encode()
                ).hexdigest(),
                normalized_title_sha256=hashlib.sha256(
                    f"{split}-title-{ordinal}".encode()
                ).hexdigest(),
                context=context,
                context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                items=items,
            )
        )
    return source.PreparedSplit(
        split=split,
        contracts=tuple(contracts),
        excluded_contract_count=1,
        source_contract_count=len(contracts) + 1,
    )


def _mode600(path: Path) -> None:
    os.chmod(path, 0o600)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _download_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Path]]:
    payloads = {
        "train": b"train-source-bytes",
        "dev": b"dev-source-bytes",
        "test": b"test-source-bytes",
    }
    frozen = tuple(
        download.FrozenSource(
            split=split,
            relative_path=f"data/{split}.json",
            size_bytes=len(payload),
            git_blob_sha1=hashlib.sha1(
                f"blob {len(payload)}\0".encode("ascii") + payload
            ).hexdigest(),
        )
        for split, payload in payloads.items()
    )
    monkeypatch.setattr(download, "SOURCES", frozen)

    @contextmanager
    def opener(url: str):
        row = next(item for item in frozen if item.url == url)
        yield io.BytesIO(payloads[row.split])

    root = tmp_path / "download"
    download.download_pinned_sources(root, opener=opener)
    return root / "download.receipt.json", {
        split: root / "source_bytes" / f"{split}.json"
        for split in payloads
    }


def _initial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict[str, source.PreparedSplit]]:
    receipt_path, paths = _download_custody(tmp_path, monkeypatch)
    secret = tmp_path / "secret"
    secret.write_bytes(b"s" * 32)
    _mode600(secret)
    prepared = {
        "TRAIN": _prepared("TRAIN", ("A_form", "F_search")),
        "DEV": _prepared("DEV", ("A_hold",) * 4),
    }

    def parser(path, *, split, selection_secret, **kwargs):
        assert selection_secret == b"s" * 32
        return prepared[split]

    root = tmp_path / "acquisition"
    subject.run_initial_acquisition(
        train_path=paths["train"],
        dev_path=paths["dev"],
        download_receipt_path=receipt_path,
        secret_path=secret,
        output_root=root,
        parser=parser,
    )
    return root, secret, receipt_path, prepared


def _promotion(
    root: Path, path: Path, *, promoted: bool
) -> dict[str, object]:
    initial = json.loads((root / "initial.receipt.json").read_text("ascii"))
    gold = initial["private_archives"]["A_hold_gold"]
    comparison = {
        "contract_count": 4,
        "nonzero_contract_count": 4,
        "net": {
            "numerator": 1 if promoted else 0,
            "denominator": 10,
        },
        "exact_sign_flip_reference_tail": {
            "numerator": 1 if promoted else 8,
            "denominator": 16,
        },
    }
    body = {
        "schema": subject.PROMOTION_SCHEMA,
        "study_id": subject.STUDY_ID,
        "study_design_self_sha256": subject.STUDY_DESIGN_SELF_SHA256,
        "source_custody_self_sha256": download.SOURCE_CUSTODY_SELF_SHA256,
        "initial_acquisition_receipt_self_sha256": initial["self_sha256"],
        "A_hold_action_archive_file_sha256": "1" * 64,
        "A_hold_action_archive_semantic_sha256": "2" * 64,
        "A_hold_gold_file_sha256": gold["file_sha256"],
        "A_hold_gold_semantic_sha256": gold["semantic_sha256"],
        "incumbent_evaluator_id": "E0_FIXED_GENERAL_COVERAGE",
        "challenger_evaluator_id": "E1_AFORM_CENTERED_RIDGE_L2_1",
        "challenger_model_sha256": "3" * 64,
        "challenger_model_self_sha256": "4" * 64,
        "E1_minus_E0_comparison": comparison,
        "promoted": promoted,
        "M_search_authorized": promoted,
        "promotion_rule": subject.PROMOTION_RULE,
        "retry_replay_resample_refit_or_gate_change_count": 0,
        "online_evaluator_API_or_fine_tune_count": 0,
    }
    payload = subject.self_hashed(body)
    path.write_bytes(subject.canonical_bytes(payload))
    _mode600(path)
    return payload


def test_initial_acquisition_binds_download_separates_gold_and_omits_f(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _secret, _receipt, _prepared_rows = _initial(
        tmp_path, monkeypatch
    )
    initial = json.loads((root / "initial.receipt.json").read_text("ascii"))
    assert initial["F_search_gold_pack_created"] is False
    assert initial["download_binding"]["opened_splits"].keys() == {
        "train",
        "dev",
    }
    assert (root / "F_search.action.private.json").is_file()
    assert not (root / "F_search.gold.sealed.private.json").exists()
    action = json.loads((root / "A_hold.action.private.json").read_text("ascii"))
    encoded = json.dumps(action, sort_keys=True)
    assert "spans" not in encoded and "merged_intervals" not in encoded
    assert (root / subject.INITIAL_MEMBERSHIP_NAME).is_file()
    for path in root.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_test_parse_requires_exact_real_promoted_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, secret, receipt_path, _prepared_rows = _initial(
        tmp_path, monkeypatch
    )
    promotion = tmp_path / "promotion.json"
    _promotion(root, promotion, promoted=True)
    test_path = receipt_path.parent / "source_bytes" / "test.json"

    def parser(path, *, split, selection_secret, test_parse_capability):
        assert split == "TEST"
        test_parse_capability.validate()
        return _prepared("TEST", ("M_search",))

    receipt = subject.run_test_acquisition(
        test_path=test_path,
        download_receipt_path=receipt_path,
        secret_path=secret,
        promotion_receipt_path=promotion,
        output_root=root,
        parser=parser,
    )
    assert receipt["status"].startswith("promotion_authorized")
    assert receipt["cross_split_title_overlap_count"] == 0
    assert (root / "M_search.gold.sealed.private.json").is_file()


def test_forged_or_nonpromotion_receipt_cannot_burn_test_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, secret, receipt_path, _prepared_rows = _initial(
        tmp_path, monkeypatch
    )
    promotion = tmp_path / "promotion.json"
    _promotion(root, promotion, promoted=False)
    with pytest.raises(subject.MaudAcquisitionError):
        subject.run_test_acquisition(
            test_path=receipt_path.parent / "source_bytes" / "test.json",
            download_receipt_path=receipt_path,
            secret_path=secret,
            promotion_receipt_path=promotion,
            output_root=root,
        )
    assert not (root / "test_parse.attempt.json").exists()

    forged = json.loads(promotion.read_text("ascii"))
    forged["schema"] = "promotion"
    body = dict(forged)
    body.pop("self_sha256")
    forged["self_sha256"] = subject.semantic_sha256(body)
    promotion.write_bytes(subject.canonical_bytes(forged))
    with pytest.raises(subject.MaudAcquisitionError):
        subject.run_test_acquisition(
            test_path=receipt_path.parent / "source_bytes" / "test.json",
            download_receipt_path=receipt_path,
            secret_path=secret,
            promotion_receipt_path=promotion,
            output_root=root,
        )
    assert not (root / "test_parse.attempt.json").exists()
