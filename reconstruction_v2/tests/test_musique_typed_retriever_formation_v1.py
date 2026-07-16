from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks.musique_typed_retriever_formation_v1 import (
    CLAIM_SCOPE,
    IMPLEMENTATION_RELATIVE_FILES,
    PRIVATE_ROW_SCHEMA,
    TOP_K,
    MuSiQueTypedFormationError,
    RetrievalParagraph,
    TypedRetrievalProgram,
    enumerate_programs,
    form_musique_typed_retriever,
    load_formation_receipt,
    load_frozen_program,
    retrieve,
    unicode_casefold_tokens,
    verify_live_implementation,
)
from assumption_agent.models import stable_hash


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _row(index: int) -> dict[str, Any]:
    paragraphs = [
        {
            "idx": 0,
            "title": "Alpha",
            "text": "Alpha links Beta.",
            "is_supporting": True,
        },
        {
            "idx": 1,
            "title": "NoiseOne",
            "text": "Copper circles quietly.",
            "is_supporting": False,
        },
        {
            "idx": 2,
            "title": "NoiseTwo",
            "text": "Silver squares remain.",
            "is_supporting": False,
        },
        {
            "idx": 3,
            "title": "NoiseThree",
            "text": "Green triangles wait.",
            "is_supporting": False,
        },
        {
            "idx": 4,
            "title": "NoiseFour",
            "text": "Orange lines continue.",
            "is_supporting": False,
        },
        {
            "idx": 5,
            "title": "NoiseFive",
            "text": "Violet arcs pause.",
            "is_supporting": False,
        },
        {
            "idx": 6,
            "title": "Beta",
            "text": "Beta reveals the target.",
            "is_supporting": True,
        },
    ]
    return {
        "schema": PRIVATE_ROW_SCHEMA,
        "split": "train",
        "item_id": f"private-item-{index:02d}",
        "question": "What follows ALPHA?",
        "corpus": paragraphs,
        "answers": [f"SecretAnswer{index}", f"Hidden Alias {index}"],
        "normalized_answers": [f"secretanswer{index}", f"hidden alias {index}"],
        "support_indices": [0, 6],
        "source_row_sha256": stable_hash({"private-source": index}),
    }


def _receipt_body(train_raw: bytes, rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_file = {
        "split": "train",
        "count": 12,
        "file_sha256": __import__("hashlib").sha256(train_raw).hexdigest(),
        "item_commitment_set_sha256": stable_hash(
            [stable_hash(row) for row in rows]
        ),
    }
    split_files = [
        train_file,
        {
            "split": "holdout-a",
            "count": 6,
            "file_sha256": "a" * 64,
            "item_commitment_set_sha256": "b" * 64,
        },
        {
            "split": "holdout-b",
            "count": 6,
            "file_sha256": "c" * 64,
            "item_commitment_set_sha256": "d" * 64,
        },
    ]
    return {
        "schema": "musique-official-core-comparison-v1-acquisition",
        "decision": "private_pack_formed_no_model_execution_authorized",
        "source": {"claim_scope": CLAIM_SCOPE},
        "ordering": {"synthetic": True},
        "counts": {
            "source_train_rows": 12,
            "eligible_rows": 12,
            "selected_rows": 24,
            "splits": {"train": 12, "holdout-a": 6, "holdout-b": 6},
            "oracle_disagreements": 0,
        },
        "commitments": {
            "private_pack_sha256": stable_hash(split_files),
            "split_files": split_files,
        },
        "private_boundary": {"private_paths_persisted_publicly": False},
        "oracles": {"synthetic": True},
        "safety": {"model_calls": 0, "online_judge_calls": 0},
    }


def _write_receipt(path: Path, body: Mapping[str, Any]) -> None:
    payload = dict(body)
    payload["acquisition_sha256"] = stable_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _bundle(tmp_path: Path) -> dict[str, Any]:
    rows = [_row(index) for index in range(12)]
    train_raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    train_path = tmp_path / "private-train-only" / "train.jsonl"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(train_raw)
    receipt_body = _receipt_body(train_raw, rows)
    receipt_path = tmp_path / "public" / "acquisition.json"
    _write_receipt(receipt_path, receipt_body)
    return {
        "rows": rows,
        "train_raw": train_raw,
        "train_path": train_path,
        "receipt_body": receipt_body,
        "receipt_path": receipt_path,
    }


def _corpus_from_row(row: Mapping[str, Any]) -> tuple[RetrievalParagraph, ...]:
    return tuple(
        RetrievalParagraph(paragraph["idx"], paragraph["title"], paragraph["text"])
        for paragraph in row["corpus"]
    )


def test_finite_typed_grammar_tokenizer_and_executable_graph() -> None:
    tokens = unicode_casefold_tokens("Straße CAFÉ ΚαληΜΈΡΑ 中文_42")
    assert tokens[0:2] == ("strasse", "café")
    assert "καλημέρα" in tokens
    assert tokens[-2:] == ("中文", "42")

    programs = tuple(enumerate_programs())
    assert len(programs) == 84
    assert all(not program.type_issues() for program in programs)
    assert all(program.top_k == TOP_K for program in programs)
    assert {program.seed_algorithm for program in programs} == {"bm25", "tfidf"}
    assert {program.expansion_mode for program in programs} == {
        "none",
        "token_one_hop",
        "entity_token_one_hop",
    }

    row = _row(0)
    corpus = _corpus_from_row(row)
    baseline = TypedRetrievalProgram("bm25", 2, 1, "none", 0)
    expanded = TypedRetrievalProgram("bm25", 2, 1, "entity_token_one_hop", 1)
    assert 6 not in retrieve(baseline, row["question"], corpus)
    assert 6 in retrieve(expanded, row["question"], corpus)
    assert len(retrieve(expanded, row["question"], corpus)) == 5


def test_train_only_formation_deduplicates_crossfits_and_emits_safe_freeze(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "safe-output"
    result = form_musique_typed_retriever(
        bundle["train_path"],
        bundle["receipt_path"],
        output_dir=output,
    )
    receipt = result.receipt
    assert result.program.expansion_mode != "none"
    assert receipt["selection_receipt"]["support_hit_count"] == 24
    assert receipt["selection_receipt"]["support_label_count"] == 24
    assert receipt["selection_receipt"]["support_recall_at_5_numerator"] == 1
    assert receipt["selection_receipt"]["support_recall_at_5_denominator"] == 1
    assert receipt["search_receipt"]["candidate_count"] == 84
    assert 0 < receipt["search_receipt"]["behavior_unique_count"] < 84
    assert receipt["search_receipt"]["behavior_alias_count"] > 0
    assert [
        row["path"] for row in receipt["implementation"]["files"]
    ] == list(IMPLEMENTATION_RELATIVE_FILES)
    assert len(receipt["implementation"]["set_sha256"]) == 64
    assert receipt["crossfit_receipt"]["fold_count"] == 4
    assert receipt["crossfit_receipt"]["selected_program_stable"] is True
    assert receipt["crossfit_receipt"]["selected_behavior_stable"] is True
    assert all(
        fold["held_out_item_count"] == 3
        and fold["fit_item_count"] == 9
        and fold["held_out_invalid_count"] == 0
        for fold in receipt["crossfit_receipt"]["folds"]
    )
    assert receipt["offline_contract"] == {
        "partition": "train",
        "model_calls": 0,
        "network_calls": 0,
        "online_evaluator_calls": 0,
        "development_execution_authorized": False,
        "sealed_execution_authorized": False,
    }

    public_raw = (output / "formation.receipt.json").read_text() + (
        output / "frozen_program.json"
    ).read_text()
    for index in range(12):
        assert f"private-item-{index:02d}" not in public_raw
        assert f"SecretAnswer{index}" not in public_raw
    assert "What follows ALPHA" not in public_raw
    assert "Alpha links Beta" not in public_raw
    assert '"corpus"' not in public_raw
    assert '"support_indices"' not in public_raw
    loaded_receipt = load_formation_receipt(output / "formation.receipt.json")
    loaded_program = load_frozen_program(
        output / "frozen_program.json",
        receipt_path=output / "formation.receipt.json",
    )
    assert loaded_receipt["receipt_hash"] == receipt["receipt_hash"]
    assert loaded_program == result.program
    frozen_envelope = json.loads((output / "frozen_program.json").read_text())
    assert frozen_envelope["implementation"] == receipt["implementation"]


def test_exact_train_path_is_physically_isolated_and_no_sibling_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    forbidden_a = bundle["train_path"].parent / "holdout-a.jsonl"
    forbidden_b = bundle["train_path"].parent / "holdout-b.jsonl"
    forbidden_a.write_text("must remain unopened", encoding="utf-8")
    forbidden_b.write_text("must remain unopened", encoding="utf-8")
    opened: list[Path] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any):
        opened.append(self.resolve())
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    form_musique_typed_retriever(bundle["train_path"], bundle["receipt_path"])
    assert set(opened) == {
        bundle["train_path"].resolve(),
        bundle["receipt_path"].resolve(),
        *(
            Path(__file__).resolve().parents[1] / relative
            for relative in IMPLEMENTATION_RELATIVE_FILES
        ),
    }
    assert forbidden_a.resolve() not in opened
    assert forbidden_b.resolve() not in opened

    wrong_name = bundle["train_path"].with_name("other.jsonl")
    wrong_name.write_bytes(bundle["train_raw"])
    with pytest.raises(MuSiQueTypedFormationError, match="exact train.jsonl"):
        form_musique_typed_retriever(wrong_name, bundle["receipt_path"])

    private_receipt = bundle["train_path"].parent / "acquisition.json"
    private_receipt.write_bytes(bundle["receipt_path"].read_bytes())
    with pytest.raises(MuSiQueTypedFormationError, match="physically separated"):
        form_musique_typed_retriever(bundle["train_path"], private_receipt)

    with pytest.raises(MuSiQueTypedFormationError, match="disjoint"):
        form_musique_typed_retriever(
            bundle["train_path"],
            bundle["receipt_path"],
            output_dir=bundle["train_path"].parent / "unsafe-output",
        )


def test_split_file_and_pack_commitment_drift_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle["train_path"].write_bytes(bundle["train_raw"] + b"\n")
    with pytest.raises(MuSiQueTypedFormationError, match="TRAIN file hash"):
        form_musique_typed_retriever(bundle["train_path"], bundle["receipt_path"])

    bundle = _bundle(tmp_path / "pack-drift")
    drifted_receipt = copy.deepcopy(bundle["receipt_body"])
    drifted_receipt["commitments"]["private_pack_sha256"] = "0" * 64
    _write_receipt(bundle["receipt_path"], drifted_receipt)
    with pytest.raises(MuSiQueTypedFormationError, match="private pack commitment"):
        form_musique_typed_retriever(bundle["train_path"], bundle["receipt_path"])

    bundle = _bundle(tmp_path / "split-drift")
    rows = copy.deepcopy(bundle["rows"])
    rows[0]["split"] = "not-train"
    raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    bundle["train_path"].write_bytes(raw)
    drifted_receipt = _receipt_body(raw, rows)
    _write_receipt(bundle["receipt_path"], drifted_receipt)
    with pytest.raises(MuSiQueTypedFormationError, match="non-TRAIN row"):
        form_musique_typed_retriever(bundle["train_path"], bundle["receipt_path"])


def test_safe_receipt_and_frozen_program_tamper_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "safe-output"
    form_musique_typed_retriever(
        bundle["train_path"], bundle["receipt_path"], output_dir=output
    )

    program_path = output / "frozen_program.json"
    tampered_program = json.loads(program_path.read_text())
    tampered_program["program"]["title_weight"] = 4
    program_path.write_text(json.dumps(tampered_program), encoding="utf-8")
    with pytest.raises(MuSiQueTypedFormationError, match="canonical|hash"):
        load_frozen_program(program_path, receipt_path=output / "formation.receipt.json")

    receipt_path = output / "formation.receipt.json"
    tampered_receipt = json.loads(receipt_path.read_text())
    tampered_receipt["selection_receipt"]["support_hit_count"] -= 1
    receipt_path.write_text(json.dumps(tampered_receipt), encoding="utf-8")
    with pytest.raises(MuSiQueTypedFormationError, match="receipt hash"):
        load_formation_receipt(receipt_path)


def test_repository_train_and_output_must_be_ignored_and_untracked(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text(
        "private-train-only/\nignored-output/\n", encoding="utf-8"
    )
    bundle = _bundle(repository)
    subprocess.run(
        ["git", "-C", str(repository), "add", "-f", "private-train-only/train.jsonl"],
        check=True,
    )
    with pytest.raises(MuSiQueTypedFormationError, match="must be untracked"):
        form_musique_typed_retriever(
            bundle["train_path"],
            bundle["receipt_path"],
            output_dir=repository / "ignored-output",
        )

    ignored_repository = tmp_path / "ignored-repository"
    ignored_repository.mkdir()
    subprocess.run(["git", "init", "-q", str(ignored_repository)], check=True)
    (ignored_repository / ".gitignore").write_text(
        "private-train-only/\nignored-output/\n", encoding="utf-8"
    )
    ignored_bundle = _bundle(ignored_repository)
    result = form_musique_typed_retriever(
        ignored_bundle["train_path"],
        ignored_bundle["receipt_path"],
        output_dir=ignored_repository / "ignored-output",
    )
    assert result.receipt["status"] == "formed_train_only"

    outside_bundle = _bundle(tmp_path / "outside-repository")
    with pytest.raises(MuSiQueTypedFormationError, match="must be git-ignored"):
        form_musique_typed_retriever(
            outside_bundle["train_path"],
            outside_bundle["receipt_path"],
            output_dir=repository / "public-output",
        )


def test_train_and_output_reject_existing_ancestor_symlinks(tmp_path: Path) -> None:
    real_root = tmp_path / "real-bundle"
    bundle = _bundle(real_root)
    linked_root = tmp_path / "linked-bundle"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(MuSiQueTypedFormationError, match="symlink component"):
        form_musique_typed_retriever(
            linked_root / "private-train-only" / "train.jsonl",
            bundle["receipt_path"],
        )

    real_output_parent = tmp_path / "real-output-parent"
    real_output_parent.mkdir()
    linked_output_parent = tmp_path / "linked-output-parent"
    linked_output_parent.symlink_to(real_output_parent, target_is_directory=True)
    with pytest.raises(MuSiQueTypedFormationError, match="symlink component"):
        form_musique_typed_retriever(
            bundle["train_path"],
            bundle["receipt_path"],
            output_dir=linked_output_parent / "safe-output",
        )


def test_frozen_program_requires_receipt_and_live_implementation_can_be_verified(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    output = tmp_path / "safe-output"
    result = form_musique_typed_retriever(
        bundle["train_path"], bundle["receipt_path"], output_dir=output
    )
    program_path = output / "frozen_program.json"
    receipt_path = output / "formation.receipt.json"
    with pytest.raises(MuSiQueTypedFormationError, match="receipt is required"):
        load_frozen_program(program_path)

    source_root = Path(__file__).resolve().parents[1]
    live_copy = tmp_path / "live-implementation"
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        source = source_root / relative
        destination = live_copy / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    assert (
        verify_live_implementation(
            result.receipt["implementation"], project_root=live_copy
        )["set_sha256"]
        == result.receipt["implementation"]["set_sha256"]
    )
    assert load_frozen_program(
        program_path,
        receipt_path=receipt_path,
        verify_live=True,
        implementation_root=live_copy,
    ) == result.program

    models_copy = live_copy / "assumption_agent" / "models.py"
    models_copy.write_text(
        models_copy.read_text(encoding="utf-8") + "\n# deterministic tamper\n",
        encoding="utf-8",
    )
    with pytest.raises(MuSiQueTypedFormationError, match="live implementation drifted"):
        verify_live_implementation(
            result.receipt["implementation"], project_root=live_copy
        )
    with pytest.raises(MuSiQueTypedFormationError, match="live implementation drifted"):
        load_frozen_program(
            program_path,
            receipt_path=receipt_path,
            verify_live=True,
            implementation_root=live_copy,
        )
