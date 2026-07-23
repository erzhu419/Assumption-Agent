from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from unittest import mock
import urllib.error

import pytest

from replication_runtime.birco_gpt54_semantic_v1 import contract, worker
from replication_runtime.birco_official_hipporag_v1 import contract as hippo_contract


def _plan() -> contract.Plan:
    return contract.Plan(
        facets=(
            contract.Facet(0, "REQUIRED", "has condition A", 4),
            contract.Facet(1, "EXCLUDED", "must not have condition B", 3),
        ),
        edges=(contract.PlanEdge(0, 1, "REFINES"),),
        generation_valid=True,
    )


def _candidates(count: int = 2) -> tuple[contract.CandidateProjection, ...]:
    base = (
        "Alpha satisfies A. It explicitly excludes B.",
        "Gamma is unrelated. Delta is uncertain.",
    )
    return tuple(
        contract.project_candidate_text(
            base[ordinal % len(base)] + f" Candidate {ordinal}.",
            candidate_ordinal=ordinal,
        )
        for ordinal in range(count)
    )


def test_projection_is_query_independent_uniform_and_has_utf8_byte_offsets() -> None:
    text = "首段。Second unit! Third clause; Fourth. Fifth. 尾段。"
    projection = contract.project_candidate_text(text, candidate_ordinal=9)
    assert projection.ordinal == 9
    assert 1 <= len(projection.evidence_units) <= contract.MAXIMUM_EVIDENCE_UNITS
    encoded = text.encode("utf-8")
    for unit in projection.evidence_units:
        containing = encoded[unit.byte_start : unit.byte_end].decode("utf-8")
        assert containing == unit.text
    assert projection.evidence_units[0].text.startswith("首段")
    assert projection.evidence_units[-1].text.endswith("尾段。")
    assert contract.candidate_projection_from_text(
        projection.projection_text, candidate_ordinal=9
    ) == projection
    with pytest.raises(contract.BircoP1GptContractError, match="canonical"):
        contract.candidate_projection_from_text(
            projection.projection_text + " ", candidate_ordinal=9
        )


def test_valid_plan_is_accepted_and_cycle_is_totalized_without_retry() -> None:
    valid = json.dumps(
        {
            "edges": [{"source": 0, "target": 1, "type": "REQUIRES"}],
            "facets": [
                {"ordinal": 0, "text": "alpha", "type": "REQUIRED", "weight": 4},
                {"ordinal": 1, "text": "beta", "type": "PREFERRED", "weight": 2},
            ],
        }
    )
    parsed = contract.parse_plan_completion(valid, query="alpha and beta")
    assert parsed.generation_valid
    assert parsed.edges[0].edge_type == "REQUIRES"

    cyclic = json.dumps(
        {
            "edges": [
                {"source": 0, "target": 1, "type": "REQUIRES"},
                {"source": 1, "target": 0, "type": "REFINES"},
            ],
            "facets": [
                {"ordinal": 0, "text": "alpha", "type": "REQUIRED", "weight": 4},
                {"ordinal": 1, "text": "beta", "type": "PREFERRED", "weight": 2},
            ],
        }
    )
    fallback = contract.parse_plan_completion(cyclic, query="alpha and beta")
    assert not fallback.generation_valid
    assert len(fallback.facets) >= 2
    assert not fallback.edges

    mixed_cycle = json.dumps(
        {
            "edges": [
                {"source": 0, "target": 1, "type": "REQUIRES"},
                {"source": 1, "target": 0, "type": "CONTRASTS_WITH"},
            ],
            "facets": [
                {"ordinal": 0, "text": "alpha", "type": "REQUIRED", "weight": 4},
                {"ordinal": 1, "text": "beta", "type": "EXCLUDED", "weight": 4},
            ],
        }
    )
    assert not contract.parse_plan_completion(
        mixed_cycle, query="alpha not beta"
    ).generation_valid


def test_matrix_requires_complete_exact_batch_and_totalizes_whole_batch() -> None:
    plan = _plan()
    candidates = _candidates()
    valid = json.dumps(
        {
            "candidates": [
                {"ordinal": 0, "rows": [[4, 0, 0], [1, 3, 1]]},
                {"ordinal": 1, "rows": [[0, 0, None], [0, 0, None]]},
            ]
        }
    )
    matrix, accepted = contract.parse_matrix_completion(
        valid, plan=plan, candidates=candidates
    )
    assert accepted
    assert matrix[0][0].support == 4
    assert matrix[0][1].contradiction == 3

    incomplete = json.dumps(
        {"candidates": [{"ordinal": 0, "rows": [[4, 0, 0], [1, 3, 1]]}]}
    )
    totalized, accepted = contract.parse_matrix_completion(
        incomplete, plan=plan, candidates=candidates
    )
    assert not accepted
    assert set(totalized) == {0, 1}
    assert all(
        cell == contract.FacetEvidence(0, 0, None)
        for rows in totalized.values()
        for cell in rows
    )


def test_raw_has_no_plan_and_requires_integer_complete_scores() -> None:
    candidates = _candidates()
    formal_candidates = _candidates(10)
    pool_hash = contract.common_projection_sha256(
        objective="rank direct relevance",
        query="condition A",
        candidates=formal_candidates,
    )
    payload = contract.raw_input(
        work_id="opaque-work",
        objective="rank direct relevance",
        query="condition A",
        candidates=formal_candidates,
        batch_ordinal=0,
        batch_count=1,
        pool_candidate_count=10,
        pool_common_projection_sha256=pool_hash,
    )
    assert "plan" not in payload
    assert payload["batch_common_projection_sha256"] == pool_hash
    assert payload["pool_common_projection_sha256"] == pool_hash
    values, accepted = contract.parse_raw_completion(
        '{"scores":[{"ordinal":0,"score":97},{"ordinal":1,"score":4}]}',
        candidates=candidates,
    )
    assert accepted and values == {0: 97, 1: 4}
    values, accepted = contract.parse_raw_completion(
        '{"scores":[{"ordinal":0,"score":97.5},{"ordinal":1,"score":4}]}',
        candidates=candidates,
    )
    assert not accepted and values == {0: 0, 1: 0}


class _Response:
    def __init__(self, value: object) -> None:
        self.raw = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, maximum: int) -> bytes:
        return self.raw[:maximum]


def _provider() -> worker.Provider:
    return worker.Provider(
        api_base="https://ruoli.dev",
        api_origin="https://ruoli.dev",
        api_key="unit-test-secret",
        label="plus",
    )


def test_worker_performs_one_request_and_never_persists_secret_or_completion() -> None:
    payload = contract.planner_input(
        work_id="opaque-work", objective="rank candidates", query="alpha and beta"
    )
    completion = json.dumps(
        {
            "facets": [
                {"ordinal": 0, "type": "REQUIRED", "text": "alpha", "weight": 4},
                {"ordinal": 1, "type": "PREFERRED", "text": "beta", "weight": 2},
            ],
            "edges": [],
        }
    )
    wire = {"choices": [{"message": {"content": completion}}]}
    with mock.patch.object(worker, "_open_no_redirect", return_value=_Response(wire)) as opened:
        terminal = worker.execute_one(mode="plan", payload=payload, provider=_provider())
    assert opened.call_count == 1
    wire_request = opened.call_args.args[0]
    assert hashlib.sha256(wire_request.data).hexdigest() == terminal[
        "model_request_sha256"
    ]
    assert terminal["generation_valid"]
    assert terminal["attempt_count"] == 1
    serialized = json.dumps(terminal)
    assert "unit-test-secret" not in serialized
    assert completion not in serialized
    assert terminal["raw_completion_persisted"] is False
    body = dict(terminal)
    claimed = body.pop("self_sha256")
    assert contract.semantic_hash(body) == claimed


def test_worker_transport_failure_is_one_terminal_totalized_attempt() -> None:
    candidates = _candidates(10)
    pool_hash = contract.common_projection_sha256(
        objective="rank candidates", query="alpha", candidates=candidates
    )
    payload = contract.raw_input(
        work_id="opaque-work",
        objective="rank candidates",
        query="alpha",
        candidates=candidates,
        batch_ordinal=0,
        batch_count=1,
        pool_candidate_count=10,
        pool_common_projection_sha256=pool_hash,
    )
    failure = urllib.error.URLError("offline")
    with mock.patch.object(worker, "_open_no_redirect", side_effect=failure) as opened:
        terminal = worker.execute_one(mode="raw", payload=payload, provider=_provider())
    assert opened.call_count == 1
    assert terminal["attempt_count"] == 1
    assert not terminal["transport_succeeded"]
    assert not terminal["generation_valid"]
    assert terminal["action"]["scores"] == [
        {"ordinal": ordinal, "score": 0} for ordinal in range(10)
    ]
    assert terminal["retry_replay_resample_or_provider_switch_count"] == 0


def test_provider_environment_rejects_conflicting_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*worker._KEY_ALIASES, *worker._BASE_ALIASES, *worker._MODEL_ALIASES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", "a")
    monkeypatch.setenv("RUOLI_GPT_KEY", "b")
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev")
    with pytest.raises(worker.BircoP1GptWorkerError, match="aliases conflict"):
        worker.load_provider_from_environment()


def test_provider_rejects_header_injection_and_redirects_are_disabled() -> None:
    with pytest.raises(worker.BircoP1GptWorkerError, match="safe header"):
        worker.Provider(
            api_base="https://ruoli.dev",
            api_origin="https://ruoli.dev",
            api_key="secret\r\nX-Leak: yes",
        )
    handler = worker._NoRedirect()
    request = worker.urllib.request.Request("https://ruoli.dev/v1/chat/completions")
    assert handler.redirect_request(
        request,
        None,
        307,
        "redirect",
        {},
        "https://attacker.invalid/collect",
    ) is None


def test_cli_input_must_be_canonical_and_output_is_exclusive(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text('{"schema":"x"}', encoding="ascii")
    with pytest.raises(worker.BircoP1GptWorkerError, match="canonical"):
        worker._read_canonical_object(input_path)

    output = tmp_path / "terminal.json"
    worker._write_exclusive(output, {"ok": True})
    assert os.stat(output).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        worker._write_exclusive(output, {"ok": True})


def test_manual_projection_rejects_overlap_and_text_span_mismatch() -> None:
    first = contract.EvidenceUnit(0, 0, 5, "alpha")
    overlapping = contract.EvidenceUnit(1, 4, 8, "beta")
    with pytest.raises(contract.BircoP1GptContractError, match="overlap"):
        contract.CandidateProjection(0, (first, overlapping))
    mismatched = contract.EvidenceUnit(0, 0, 6, "alpha")
    with pytest.raises(contract.BircoP1GptContractError, match="byte span"):
        contract.CandidateProjection(0, (mismatched,))


def test_work_identity_and_batch_accounting_are_not_model_visible() -> None:
    payload = contract.raw_input(
        work_id="opaque-secret-work-id",
        objective="rank candidates",
        query="alpha",
        candidates=_candidates(10),
        batch_ordinal=0,
        batch_count=1,
        pool_candidate_count=10,
        pool_common_projection_sha256=contract.common_projection_sha256(
            objective="rank candidates", query="alpha", candidates=_candidates(10)
        ),
    )
    visible = worker._model_payload("raw", payload)
    serialized = json.dumps(visible)
    assert "opaque-secret-work-id" not in serialized
    assert "batch_ordinal" not in visible
    assert "batch_common_projection_sha256" not in visible
    assert "pool_common_projection_sha256" not in visible


def test_pool_common_projection_hash_is_identical_to_hippo_contract() -> None:
    candidates = _candidates(10)
    objective = "rank candidates"
    query = "alpha"
    semantic_hash = contract.common_projection_sha256(
        objective=objective, query=query, candidates=candidates
    )
    documents = [
        {"ordinal": row.ordinal, "text": row.projection_text}
        for row in candidates
    ]
    assert semantic_hash == hippo_contract.common_projection_sha256(
        objective=objective,
        query=query,
        documents=documents,
    )


def test_cli_consumes_durable_claim_before_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = contract.planner_input(
        work_id="opaque-work",
        objective="rank candidates",
        query="alpha and beta",
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_bytes(contract.canonical_json_bytes(payload))
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", "unit-test-secret")
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev")
    monkeypatch.setenv("ASSUMPTION_V2_MODEL", contract.MODEL_ID)
    monkeypatch.setenv("BIRCO_P1_PROVIDER_LABEL", "plus")
    for name in ("RUOLI_GPT_KEY", "GPT5_API_KEY", "RUOLI_BASE_URL", "GPT5_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    completion = json.dumps(
        {
            "facets": [
                {"ordinal": 0, "type": "REQUIRED", "text": "alpha", "weight": 4},
                {"ordinal": 1, "type": "PREFERRED", "text": "beta", "weight": 2},
            ],
            "edges": [],
        }
    )
    wire = {"choices": [{"message": {"content": completion}}]}

    def opened(_request, *, timeout):
        assert timeout == worker.MODEL_TIMEOUT_SECONDS
        claim = output_path.with_name(output_path.name + ".attempt.json")
        assert claim.is_file()
        assert json.loads(claim.read_text(encoding="ascii"))["status"].startswith(
            "consumed_before"
        )
        return _Response(wire)

    monkeypatch.setattr(worker, "_open_no_redirect", opened)
    assert worker.main(
        ["--mode", "plan", "--input", str(input_path), "--output", str(output_path)]
    ) == 0
    assert output_path.is_file()
    with pytest.raises(FileExistsError):
        worker.main(
            ["--mode", "plan", "--input", str(input_path), "--output", str(output_path)]
        )
