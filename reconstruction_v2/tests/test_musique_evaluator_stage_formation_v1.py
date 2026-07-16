from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import musique_evaluator_stage_formation_v1 as stage
from assumption_agent.benchmarks import musique_recursive_study_acquisition_v1 as acquisition
from assumption_agent.models import stable_hash
from tests.test_musique_recursive_study_blocks_and_m1_v1 import (
    _study_fixture,
    _write_block,
)


def _form_a_and_f3(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    a_block = _write_block(tmp_path, "A_form", raw_by_block["A_form"])
    f3_block = _write_block(tmp_path, "F3", raw_by_block["F3"])
    a_private = tmp_path / "a-form.private.evidence.json"
    a_public = tmp_path / "a-form.public.receipt.json"
    f3_private = tmp_path / "f3.private.evidence.json"
    f3_public = tmp_path / "f3.public.receipt.json"
    a_receipt = stage.form_a_form_stage(
        block_path=a_block,
        acquisition_receipt_path=receipt_path,
        private_evidence_output_path=a_private,
        public_receipt_output_path=a_public,
    )
    f3_receipt = stage.form_f3_stage(
        block_path=f3_block,
        acquisition_receipt_path=receipt_path,
        a_form_private_evidence_path=a_private,
        a_form_public_receipt_path=a_public,
        private_evidence_output_path=f3_private,
        public_receipt_output_path=f3_public,
    )
    return {
        "acquisition": receipt_path,
        "raw": raw_by_block,
        "a_private": a_private,
        "a_public": a_public,
        "a_receipt": a_receipt,
        "f3_private": f3_private,
        "f3_public": f3_public,
        "f3_receipt": f3_receipt,
    }


def test_a_form_and_f3_use_full_fixed_dsl_and_emit_hash_safe_receipts(
    tmp_path: Path,
) -> None:
    formed = _form_a_and_f3(tmp_path)
    candidate_count = len(stage.fixed_typed_programs())
    assert candidate_count == 84
    for key, expected_stage in (
        ("a_receipt", "A_form"),
        ("f3_receipt", "F3"),
    ):
        receipt = formed[key]
        assert receipt["stage"] == expected_stage
        assert receipt["candidate_set_binding"]["candidate_count"] == candidate_count
        assert receipt["execution"]["work_unit_count"] == (
            candidate_count * acquisition.BLOCK_COUNT
        )
        assert receipt["execution"]["retrieval_call_count"] == (
            candidate_count * acquisition.BLOCK_COUNT
        )
        assert receipt["execution"]["retrieval_terminal_count"] == (
            candidate_count * acquisition.BLOCK_COUNT
        )
        assert receipt["execution"][
            "all_terminals_joined_before_support_scoring"
        ] is True
        assert receipt["offline_contract"] == {
            "model_calls": 0,
            "generator_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "measurement_blocks_accessed": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        }
    assert formed["a_receipt"]["core_receipt"]["partition"] == "A_form"
    assert formed["f3_receipt"]["core_receipt"]["partition"] == "F3"
    assert formed["f3_receipt"]["core_receipt"]["anchor_accessed"] is False
    assert formed["f3_receipt"]["core_receipt"]["measurement_accessed"] is False

    a_evidence, _a_cache, a_public = stage.load_a_form_bundle(
        private_evidence_path=formed["a_private"],
        public_receipt_path=formed["a_public"],
    )
    f3_evidence, _f3_cache, f3_public = stage.load_f3_bundle(
        private_evidence_path=formed["f3_private"],
        public_receipt_path=formed["f3_public"],
        a_form_private_evidence_path=formed["a_private"],
        a_form_public_receipt_path=formed["a_public"],
    )
    assert len(a_evidence) == len(f3_evidence) == candidate_count
    assert a_public["receipt_sha256"] == formed["a_receipt"]["receipt_sha256"]
    assert f3_public["receipt_sha256"] == formed["f3_receipt"]["receipt_sha256"]

    public_text = formed["a_public"].read_text("utf-8") + formed[
        "f3_public"
    ].read_text("utf-8")
    for private_text in (
        "private-A_form-",
        "private-F3-",
        "Which private record",
        "root evidence",
        '"items"',
        '"support_indices"',
    ):
        assert private_text not in public_text


def test_exact_partition_and_rehashed_core_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    a_hold = _write_block(tmp_path, "A_hold", raw_by_block["A_hold"])
    with pytest.raises(Exception, match="exact named block|identity|hash"):
        stage.form_a_form_stage(
            block_path=a_hold,
            acquisition_receipt_path=receipt_path,
            private_evidence_output_path=tmp_path / "wrong.private.json",
            public_receipt_output_path=tmp_path / "wrong.public.json",
        )

    formed = _form_a_and_f3(tmp_path / "valid")
    tampered = json.loads(formed["a_public"].read_text("utf-8"))
    tampered["core_receipt"]["challenger_rule"]["macro_weight"] += 1
    core_body = dict(tampered["core_receipt"])
    core_body.pop("formation_sha256")
    tampered["core_receipt"]["formation_sha256"] = stable_hash(core_body)
    outer_body = dict(tampered)
    outer_body.pop("receipt_sha256")
    tampered["receipt_sha256"] = stable_hash(outer_body)
    formed["a_public"].write_text(
        json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(stage.MuSiQueEvaluatorStageFormationError, match="core"):
        stage.load_a_form_bundle(
            private_evidence_path=formed["a_private"],
            public_receipt_path=formed["a_public"],
            verify_live=False,
        )


def test_formation_api_has_no_candidate_operator_or_result_injection() -> None:
    forbidden = {
        "candidate_programs",
        "evidence",
        "operator",
        "operator_factory",
        "result_injection",
        "results",
        "retriever",
        "runner",
    }
    for function in (stage.form_a_form_stage, stage.form_f3_stage):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
