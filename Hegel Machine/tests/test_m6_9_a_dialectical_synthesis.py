"""Unit tests for Stage-2 Hegel M6.9-A Dialectical Two-Branch Synthesis Engine."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import (
    PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY,
    reconcile_thesis_and_antithesis_publicly,
)


def test_public_container_benchmark_script_parses_and_has_zero_private_leaks() -> None:
    ast.parse(PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY)
    for forbidden in ("answer_rubric", "ground_truth", "/tests/", "97703", "0.236", "3.41", "ruoli.dev"):
        assert forbidden not in PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY


def test_biomystery_unanimous_and_nonempty_dominance_synthesis() -> None:
    bm_task = m6f.FrontierBenchmarkTask(
        task_id="bm__test",
        short_name="bm_test",
        benchmark="BioMysteryBench-Preview",
        wave="wave_1",
        scientific_domain="Genomics",
        wormhole_family="fam_bm",
        instruction_text="Identify the organism.",
        agent_image_tag="mock",
        verifier_image_tag="mock",
        declared_artifacts=(),
    )

    class DummyClient:
        def call_structured(self, **_kwargs):
            return {
                "ok": True,
                "parsed": {
                    "selected_branch": "antithesis",
                    "final_answer": "Arabidopsis thaliana",
                    "evidence_audit_rationale": "99.8% BLAST identity in Antithesis.",
                },
                "prompt_tokens": 10,
                "cached_tokens": 0,
                "completion_tokens": 5,
                "wall_ms": 1.0,
            }

    with tempfile.TemporaryDirectory(prefix="m69_bm_test_") as td:
        td_p = Path(td)
        t_dir = td_p / "t"
        a_dir = td_p / "a"
        t_dir.mkdir()
        a_dir.mkdir()
        (t_dir / "02_human_readable_trajectory.md").write_text("Guess: Zea mays\n", encoding="utf-8")
        (a_dir / "02_human_readable_trajectory.md").write_text("BLAST 99.8% Arabidopsis thaliana\n", encoding="utf-8")

        # 1. Unanimous fast-path (0 LLM tokens!)
        u_res = reconcile_thesis_and_antithesis_publicly(
            task=bm_task,
            synth_cid=None,
            thesis_dir=t_dir,
            thesis_summary={"submitted_final_answer": "Homo sapiens"},
            thesis_tar=td_p / "none.tar",
            antithesis_dir=a_dir,
            antithesis_summary={"submitted_final_answer": "homo sapiens"},
            antithesis_tar=td_p / "none2.tar",
            client=DummyClient(),  # type: ignore[arg-type]
            stage_subdir="test",
        )
        assert u_res["selection_mechanism"] == "unanimous_consensus"
        assert u_res["llm_tokens"]["prompt"] == 0

        # 2. Disagreement adjudicated via blinded evidence witness
        d_res = reconcile_thesis_and_antithesis_publicly(
            task=bm_task,
            synth_cid=None,
            thesis_dir=t_dir,
            thesis_summary={"submitted_final_answer": "Zea mays"},
            thesis_tar=td_p / "none.tar",
            antithesis_dir=a_dir,
            antithesis_summary={"submitted_final_answer": "Arabidopsis thaliana"},
            antithesis_tar=td_p / "none2.tar",
            client=DummyClient(),  # type: ignore[arg-type]
            stage_subdir="test",
        )
        assert d_res["selected_branch"] == "antithesis"
        assert d_res["synthesized_final_answer"] == "Arabidopsis thaliana"
