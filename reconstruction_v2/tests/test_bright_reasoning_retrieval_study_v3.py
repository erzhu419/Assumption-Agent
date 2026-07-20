from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as v1
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v3 as v3
from replication_runtime.bright_query_generator_v1 import contract


def test_v3_activation_redirects_stages_but_reuses_v2_corpus() -> None:
    names = (
        "VERSION",
        "CORPUS_RESULT_SCHEMA",
        "FORMAL_ROOT_RELATIVE",
        "CORPUS_RESULT_RELATIVE",
        "PUBLIC_STAGE_RESULTS",
        "STAGE_PREDECESSORS",
        "_load_corpus",
        "_run_qwen",
    )
    original = {name: getattr(v1, name) for name in names}
    try:
        v3._activate_v3()
        assert v1.VERSION == v3.VERSION
        assert v1.CORPUS_RESULT_SCHEMA == v3.CORPUS_RESULT_SCHEMA
        assert v1.FORMAL_ROOT_RELATIVE == v3.FORMAL_ROOT_RELATIVE
        assert v1.PUBLIC_STAGE_RESULTS["M_search"] == v3.M_RESULT_RELATIVE
        assert v1.STAGE_PREDECESSORS["G_form"] == v3.CORPUS_RESULT_RELATIVE
        assert v1._load_corpus is v3._load_corpus_v3
        assert v1._run_qwen is v3._run_qwen_v3
    finally:
        for name, value in original.items():
            setattr(v1, name, value)


def test_qwen_v3_runner_binds_worker_schedule_and_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    item = v1.ViewItem(
        ordinal=0,
        family="BIOLOGY",
        commitment="a" * 64,
        query="Why does a cold glass become wet?",
        excluded_ids=(),
    )
    completion = json.dumps(
        {
            "entity_query": "cold glass water droplets",
            "relation_query": "glass temperature relation to surface moisture",
            "mechanism_query": "water vapor condensation mechanism on cold surfaces",
            "constraint_query": "humid air sealed room cold drinking glass conditions",
        },
        separators=(",", ":"),
    )
    output = contract.output_payload(
        [
            contract.build_output_item(
                ordinal=0,
                completion=completion,
                completion_token_count=50,
                query=item.query,
            )
        ]
    )
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(contract.canonical_json_bytes(output))
        schedule = {
            "batch_count": 1,
            "batch_sizes": [1],
            "input_item_count": 1,
            "maximum_padded_prompt_tokens": 200,
            "maximum_prompt_tokens": 200,
            "oversized_singleton_count": 0,
            "padded_prompt_token_budget": 4096,
            "status": "passed",
            "valid_generation_count": 1,
        }
        return SimpleNamespace(
            returncode=0,
            stderr=b"",
            stdout=(json.dumps(schedule, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii"),
        )

    monkeypatch.setattr(v3.subprocess, "run", fake_run)
    monkeypatch.setattr(
        v1,
        "_network_trace_receipt",
        lambda root, prefix: {
            "external_connect_syscall_count": 0,
            "external_send_syscall_count": 0,
            "loopback_bind_count": 1,
            "trace_file_count": 1,
            "trace_set_sha256": "b" * 64,
        },
    )
    parsed, receipt = v3._run_qwen_v3(tmp_path, stage, (item,))
    assert parsed["items"][0]["generation_valid"] is True
    assert receipt["schedule"]["batch_sizes"] == [1]
    assert observed["timeout"] == v3.QWEN_TIMEOUT_SECONDS
    assert v3.QWEN_WORKER_MODULE in observed["command"]
