from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import sc100_synthetic_shadow_v1 as runner
from assumption_agent.benchmarks.sc100_shadow_gold_adapter_v1 import AdaptedShadowRecord
from assumption_agent.models import stable_hash


ALL_IDS = [
    *(f"S{i:02d}" for i in range(1, 13)),
    *(f"C{i:02d}" for i in range(1, 7)),
    *(f"N{i:02d}" for i in range(1, 7)),
]
REASONS = {
    "N01": "public_entity",
    "N02": "attorney_fee_dispute",
    "N03": "multiple_plaintiffs",
    "N04": "payment_not_requested",
    "N05": "conflicting_claim_amount",
    "N06": "non_california_venue",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _with_pop_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = stable_hash(result)
    return result


def _operator_receipt(**values: Any) -> dict[str, Any]:
    receipt = dict(values)
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt


def _adapted() -> tuple[AdaptedShadowRecord, ...]:
    rows = []
    for case_id in ALL_IDS:
        if case_id.startswith("S"):
            rows.append(AdaptedShadowRecord(case_id, "required_positive", "fill", {"id": case_id}, None))
        elif case_id.startswith("C"):
            rows.append(
                AdaptedShadowRecord(case_id, "coverage_probe", "coverage_probe_fill", {"id": case_id}, None)
            )
        else:
            rows.append(AdaptedShadowRecord(case_id, "true_negative", "reject", None, REASONS[case_id]))
    return tuple(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    project = tmp_path
    corpus = project / "reference/corpus"
    prompts = corpus / "prompts"
    prompts.mkdir(parents=True)
    seed = "unit-shadow-seed"
    order = sorted(
        ALL_IDS,
        key=lambda case_id: hashlib.sha256(
            seed.encode() + b"\x00" + case_id.encode()
        ).digest(),
    )
    for case_id in ALL_IDS:
        (prompts / f"{case_id}.md").write_text(f"unit prompt {case_id}\n", encoding="utf-8")
    (corpus / "gold.jsonl").write_text("unit-gold-placeholder\n", encoding="utf-8")
    payload = {
        path.relative_to(corpus).as_posix(): _sha(path)
        for path in corpus.rglob("*")
        if path.is_file()
    }
    spec: dict[str, Any] = {
        "schema_version": "synthetic-sc100-shadow-corpus-v1",
        "seed": seed,
        "case_order": order,
        "cohorts": {
            "required_positive": {"count": 12},
            "coverage_probe": {"count": 6},
            "true_negative": {"count": 6},
        },
        "payload_sha256": payload,
        "corpus_self_hash": {"algorithm": "sha256", "value": "0" * 64},
    }
    canonical = json.dumps(
        spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    spec["corpus_self_hash"]["value"] = hashlib.sha256(canonical).hexdigest()
    spec_path = corpus / "corpus_spec.json"
    _dump(spec_path, spec)

    blank = project / "blank.pdf"
    blank.write_bytes(b"%PDF-unit-blank\n")
    oracle_source = project / "oracle.py"
    oracle_source.write_text("# unit oracle binding\n", encoding="utf-8")
    candidate_source = project / "candidate.py"
    candidate_source.write_text("# unit candidate\n", encoding="utf-8")
    candidate_test = project / "test_candidate.py"
    candidate_test.write_text("# unit candidate test\n", encoding="utf-8")
    qualification = _with_pop_hash(
        {
            "oracle_source_sha256": _sha(oracle_source),
            "corpus_binding": {
                "corpus_self_hash": spec["corpus_self_hash"]["value"],
                "corpus_spec_file_sha256": _sha(spec_path),
                "gold_file_sha256": _sha(corpus / "gold.jsonl"),
            },
            "qualification": {
                "oracle_ready_for_frozen_measurement": True,
                "model_calls": 0,
                "ruoli_calls": 0,
                "online_judge_calls": 0,
            },
            "boundary": {
                "may_measure_one_frozen_synthetic_shadow": True,
                "oracle_change_after_first_shadow_outcome_allowed": False,
            },
        },
        "result_hash",
    )
    qualification_path = project / "oracle-result.json"
    _dump(qualification_path, qualification)
    bound = [
        spec_path,
        corpus / "gold.jsonl",
        blank,
        oracle_source,
        qualification_path,
        candidate_source,
        candidate_test,
    ]
    candidate_binding = {
        "operator_version": "unit-operator-v2",
        "source_path": candidate_source.relative_to(project).as_posix(),
        "source_sha256": _sha(candidate_source),
        "test_path": candidate_test.relative_to(project).as_posix(),
        "test_sha256": _sha(candidate_test),
    }
    candidate_binding["candidate_id"] = stable_hash(candidate_binding)
    manifest: dict[str, Any] = {
        "schema": runner.PREREGISTRATION_SCHEMA,
        "formal_decision_budget": 1,
        "corpus_spec_path": spec_path.relative_to(project).as_posix(),
        "gold_path": (corpus / "gold.jsonl").relative_to(project).as_posix(),
        "blank_pdf_path": blank.relative_to(project).as_posix(),
        "oracle_source_path": oracle_source.relative_to(project).as_posix(),
        "oracle_qualification_result_path": qualification_path.relative_to(project).as_posix(),
        "blank_sha256": _sha(blank),
        "candidate_binding": candidate_binding,
        "runtime_binding": {"image_id": "sha256:" + "1" * 64},
        "formal_paths": {
            "root": "formal",
            "report": "formal/report.json",
            "decision_lock": "formal/decision.lock.json",
            "outputs": "formal/outputs",
        },
        "file_bindings": [
            {"path": path.relative_to(project).as_posix(), "sha256": _sha(path)}
            for path in bound
        ],
    }
    manifest = _with_pop_hash(manifest, "manifest_hash")
    manifest_path = project / "manifest.json"
    _dump(manifest_path, manifest)
    return project, manifest_path, order


def _oracle_receipt(blank: Path, filled: Path, gold: Mapping[str, Any], qualified: bool) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "sc100-shadow-oracle-v1",
        "qualified": qualified,
        "failure_codes": [] if qualified else ["unit_coverage_failure"],
        "bindings": {
            "blank_sha256": _sha(blank),
            "filled_sha256": _sha(filled),
            "semantic_gold_sha256": stable_hash(gold),
        },
        "target_widget_count": 1,
        "runtime": {},
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def test_full_generation_barrier_redaction_coverage_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The repository's configured pytest temp root is a Windows mount that does
    # not expose POSIX 0600 modes; use native /tmp for the lock-mode assertion.
    native_tmp = Path(tempfile.mkdtemp(prefix="sc100-shadow-unit-", dir="/tmp"))
    project, manifest_path, _ = _fixture(native_tmp)
    generated: set[str] = set()
    gold_loaded = False

    def operator(**kwargs: Any) -> Mapping[str, Any]:
        nonlocal gold_loaded
        assert set(kwargs) == {"instruction", "blank_pdf", "output_pdf"}
        assert not gold_loaded
        case_id = str(kwargs["instruction"]).split()[-1]
        generated.add(case_id)
        common = {
            "operator_version": "unit-operator-v2",
            "input_sha256": _sha(Path(kwargs["blank_pdf"])),
            "instruction_sha256": hashlib.sha256(
                str(kwargs["instruction"]).encode("utf-8")
            ).hexdigest(),
            "source_unchanged": True,
            "partial_output_created": False,
            "raw_case_text_persisted": False,
        }
        if case_id.startswith("N"):
            return _operator_receipt(
                **common,
                action="reject",
                reason_code=REASONS[case_id],
                output_pdf=None,
            )
        output = Path(kwargs["output_pdf"])
        output.write_bytes(f"%PDF-unit-filled-{case_id}\n".encode())
        return _operator_receipt(
            **common,
            action="fill",
            output_sha256=_sha(output),
            mutation_count=30,
            atomic_publish=True,
            temporary_cleanup_verified=True,
        )

    def load_gold(_: Path) -> tuple[AdaptedShadowRecord, ...]:
        nonlocal gold_loaded
        assert generated == set(ALL_IDS)
        gold_loaded = True
        return _adapted()

    def oracle(**kwargs: Any) -> Mapping[str, Any]:
        assert gold_loaded
        gold = kwargs["semantic_gold"]
        qualified = not str(gold["id"]).startswith("C")
        return _oracle_receipt(kwargs["blank_pdf"], kwargs["filled_pdf"], gold, qualified)

    monkeypatch.setattr(runner, "load_shadow_gold_jsonl", load_gold)
    report = runner.run_sc100_synthetic_shadow(
        project_root=project,
        preregistration_path=manifest_path,
        operator=operator,
        oracle=oracle,
    )
    assert report["generation_parallelism"] == 24
    assert report["oracle_parallelism"] == 18
    assert report["all_generation_joined_before_any_oracle"] is True
    assert report["counts"]["required_positive_qualified"] == 12
    assert report["counts"]["true_negative_exact"] == 6
    assert report["counts"]["coverage_probe_qualified"] == 0
    assert report["synthetic_feasibility_passed"] is True
    assert {row["result_code"] for row in report["coverage_probe"]} == {"coverage_starved"}
    serialized = json.dumps(report)
    assert "unit prompt" not in serialized
    assert '"semantic_gold"' not in serialized
    lock = project / "formal/decision.lock.json"
    assert os.stat(lock).st_mode & 0o777 == 0o600
    assert runner.verify_existing_sc100_synthetic_shadow(
        project_root=project, preregistration_path=manifest_path
    ) == report


def test_negative_partial_write_is_not_exact_reject(tmp_path: Path) -> None:
    prompt = tmp_path / "N01.md"
    prompt.write_text("unit N01\n", encoding="utf-8")
    blank = tmp_path / "blank.pdf"
    blank.write_bytes(b"%PDF-blank\n")
    case = runner.GenerationCase("N01", prompt, _sha(prompt), tmp_path / "out/candidate.pdf")

    def operator(**kwargs: Any) -> Mapping[str, Any]:
        Path(kwargs["output_pdf"]).write_bytes(b"partial")
        return _operator_receipt(
            action="reject",
            operator_version="unit-operator-v2",
            input_sha256=_sha(blank),
            instruction_sha256=_sha(prompt),
            reason_code="public_entity",
            output_pdf=None,
            source_unchanged=True,
            partial_output_created=False,
            raw_case_text_persisted=False,
        )

    row = runner._run_generation(case, blank_pdf=blank, blank_sha256=_sha(blank), operator=operator)
    gold = AdaptedShadowRecord("N01", "true_negative", "reject", None, "public_entity")
    codes = runner._validate_generation(row, gold, _sha(blank), "unit-operator-v2")
    assert "negative_output_created" in codes
    assert "negative_partial_output_created" in codes


def test_manifest_hash_and_lock_tamper_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_decision_budget"] = 2
    _dump(manifest_path, manifest)
    with pytest.raises(runner.SC100SyntheticShadowError):
        runner.run_sc100_synthetic_shadow(
            project_root=project,
            preregistration_path=manifest_path,
            operator=lambda **_: {},
            oracle=lambda **_: {},
        )


def test_qualification_fixture_command_is_immutable_and_network_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from assumption_agent.benchmarks import sc100_shadow_oracle_qualification_v1 as qualification

    blank = tmp_path / "blank.pdf"
    filled = tmp_path / "filled.pdf"
    oracle_source = tmp_path / "oracle.py"
    blank.write_bytes(b"blank")
    filled.write_bytes(b"filled")
    oracle_source.write_text("# oracle\n", encoding="utf-8")
    gold = {"unit": True}
    receipt = {
        "qualified": True,
        "failure_codes": [],
        "bindings": {
            "filled_sha256": _sha(filled),
            "semantic_gold_sha256": stable_hash(gold),
        },
        "runtime": {
            "pypdf": "5.1.0",
            "pillow": "12.3.0",
            "pdftotext": "24.02.0",
            "pdftoppm": "24.02.0",
        },
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        commands.append(command)
        if command[:3] == ["docker", "container", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt), stderr="")

    monkeypatch.setattr(qualification.subprocess, "run", fake_run)
    qualification._run_fixture(
        project=tmp_path,
        manifest_hash="a" * 64,
        runtime={
            "image_id": "sha256:" + "b" * 64,
            "fixture_timeout_seconds": 1,
            "pypdf_version": "5.1.0",
            "pillow_version": "12.3.0",
            "poppler_version": "24.02.0",
        },
        oracle_path=oracle_source,
        blank_path=blank,
        fixture={
            "fixture_id": "S01",
            "kind": "positive_canary",
            "mutation_class": None,
            "filled_path": filled,
            "filled_sha256": _sha(filled),
            "semantic_gold": gold,
            "semantic_gold_sha256": stable_hash(gold),
            "expected_qualified": True,
            "must_include_failure_codes": (),
        },
    )
    command = commands[0]
    for pair in (
        ("--pull", "never"),
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
    ):
        index = command.index(pair[0])
        assert command[index : index + 2] == list(pair)
    assert "--read-only" in command
    assert all(not item.startswith("http") for item in command)
