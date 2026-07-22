from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import tatqa_p20_label_free_runtime_v1 as features
from replication_runtime.tatqa_p20_v1 import typed_plan_contract as contract
from replication_runtime.tatqa_p20_v1 import typed_plan_worker as worker


def _item() -> features.LabelFreeRuntimeItem:
    return features.LabelFreeRuntimeItem(
        item_id="f" * 64,
        question="Compare Acme revenue for 2023 and 2024.",
        units=(
            features.RuntimeUnit("T:0", "TABLE HEADER | company | revenue | year"),
            features.RuntimeUnit("T:1", "Acme | 100 | 2023"),
            features.RuntimeUnit("T:2", "Acme | 130 | 2024"),
            features.RuntimeUnit("P:1", "Acme described the annual change."),
            features.RuntimeUnit("P:2", "The filing also discusses costs."),
        ),
    )


def _valid_completion() -> str:
    return json.dumps(
        {
            "entity_facets": ["Acme"],
            "metric_facets": ["revenue"],
            "time_facets": ["2023", "2024"],
            "operation": "COMPARE",
            "relation_query": "Acme revenue comparison 2023 2024",
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _prompt_receipt() -> dict[str, object]:
    return {
        "prompt_sha256": "a" * 64,
        "prompt_token_count": 100,
        "prompt_projection_sha256": "b" * 64,
    }


def test_projection_is_deterministic_truncated_and_identity_free() -> None:
    first = contract.project_item(_item(), 0)
    second = contract.project_item(_item(), 0)
    assert first == second
    assert first.ordinal == 0
    assert first.table_header == _item().units[0].text
    assert first.paragraph_leads == tuple(row.text for row in _item().units[3:])
    assert _item().item_id not in repr(first)


def test_input_roundtrip_is_canonical_and_exact_shape() -> None:
    row = contract.project_item(_item(), 0)
    payload = contract.input_payload((row,))
    raw = contract.canonical_json_bytes(payload)
    assert contract.parse_input(raw) == (row,)
    bad = dict(payload)
    bad["items"] = [{**payload["items"][0], "family": "TABLE"}]
    with pytest.raises(contract.TatqaP20TypedPlanRuntimeError):
        contract.parse_input(contract.canonical_json_bytes(bad))


def test_valid_completion_survives_and_invalid_completion_totalizes() -> None:
    item = contract.project_item(_item(), 0)
    valid = contract.build_output_item(
        item=item,
        completion=_valid_completion(),
        completion_token_count=30,
        **_prompt_receipt(),
    )
    assert valid["generation_valid"] is True
    assert valid["plan"]["operation"] == "COMPARE"
    invalid = contract.build_output_item(
        item=item,
        completion='{"entity_facets":[],"answer":"130"}',
        completion_token_count=7,
        **_prompt_receipt(),
    )
    assert invalid["generation_valid"] is False
    assert invalid["plan"]["relation_query"] == item.question
    assert set(invalid["plan"]) == {
        "entity_facets",
        "metric_facets",
        "time_facets",
        "operation",
        "relation_query",
    }
    output = contract.output_payload((valid,))
    assert contract.parse_output(contract.canonical_json_bytes(output)) == output


def test_prompt_is_label_free_inert_and_freezes_exact_output_schema() -> None:
    item = contract.project_item(_item(), 0)
    prompt = worker.prompt_for(item)
    assert item.question in prompt
    assert item.table_header in prompt
    assert "answer the question" in worker.SYSTEM_PROMPT
    lowered = prompt.casefold()
    assert "gold" not in lowered
    assert "item_id" not in lowered
    assert "answer_from" not in lowered
    for field in (
        "entity_facets",
        "metric_facets",
        "time_facets",
        "operation",
        "relation_query",
    ):
        assert field in prompt


def test_noncanonical_or_markdown_completion_is_totalized_not_retried() -> None:
    item = contract.project_item(_item(), 0)
    fenced = "```json\n" + _valid_completion() + "\n```"
    row = contract.build_output_item(
        item=item,
        completion=fenced,
        completion_token_count=40,
        **_prompt_receipt(),
    )
    assert row["generation_valid"] is False
    assert row["plan"]["operation"] == "OTHER"


def test_duplicate_completion_keys_fail_into_totalizer() -> None:
    item = contract.project_item(_item(), 0)
    duplicate = _valid_completion().replace(
        '"entity_facets":["Acme"]',
        '"entity_facets":["wrong"],"entity_facets":["Acme"]',
    )
    row = contract.build_output_item(
        item=item,
        completion=duplicate,
        completion_token_count=40,
        **_prompt_receipt(),
    )
    assert row["generation_valid"] is False


class _TwoTokensPerCharacter:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text) * 2))}


def test_tokenizer_aware_projection_fits_without_sampling_or_retry() -> None:
    item = contract.PlanInput(
        ordinal=0,
        question="q" * contract.MAXIMUM_QUESTION_CHARACTERS,
        table_header="h" * contract.MAXIMUM_TABLE_HEADER_CHARACTERS,
        paragraph_leads=(
            "p" * contract.MAXIMUM_PARAGRAPH_LEAD_CHARACTERS,
        )
        * contract.MAXIMUM_PARAGRAPH_LEADS,
    )
    tokenizer = _TwoTokensPerCharacter()
    first = worker.fitted_prompt_for(item, tokenizer)
    second = worker.fitted_prompt_for(item, tokenizer)
    assert first == second
    assert first.token_count <= worker.MAXIMUM_USER_PROMPT_TOKENS
    assert first.paragraph_lead_count < len(item.paragraph_leads)
    assert len(first.prompt_sha256) == len(first.projection_sha256) == 64


def test_model_context_must_fit_full_prompt_plus_completion() -> None:
    exact = SimpleNamespace(
        config=SimpleNamespace(
            max_position_embeddings=worker.MINIMUM_MODEL_CONTEXT_TOKENS
        )
    )
    assert worker._require_model_context(exact) == worker.MINIMUM_MODEL_CONTEXT_TOKENS
    short = SimpleNamespace(
        config=SimpleNamespace(
            max_position_embeddings=worker.MINIMUM_MODEL_CONTEXT_TOKENS - 1
        )
    )
    with pytest.raises(contract.TatqaP20TypedPlanRuntimeError, match="context"):
        worker._require_model_context(short)


def test_worker_terminal_interval_wraps_actual_model_load_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    item = contract.project_item(_item(), 0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_bytes(
        contract.canonical_json_bytes(contract.input_payload((item,)))
    )
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768))
    events = []

    def load(_path):
        events.append("load")
        return model, object()

    completion = contract.build_output_item(
        item=item,
        completion=_valid_completion(),
        completion_token_count=30,
        **_prompt_receipt(),
    )

    def generate(**_kwargs):
        events.append("generate")
        return contract.output_payload((completion,))

    ticks = iter((111, 333))
    monkeypatch.setattr(worker, "_load_model", load)
    monkeypatch.setattr(worker, "generate", generate)
    monkeypatch.setattr(worker.time, "monotonic_ns", lambda: next(ticks))

    assert worker.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(tmp_path / "model"),
        ]
    ) == 0
    terminal = json.loads(capsys.readouterr().out)
    assert events == ["load", "generate"]
    assert terminal["model_execution_started_monotonic_ns"] == 111
    assert terminal["model_execution_finished_monotonic_ns"] == 333
    assert output_path.is_file()
