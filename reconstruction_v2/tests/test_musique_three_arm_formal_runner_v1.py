from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence

import pytest

import assumption_agent.benchmarks.musique_three_arm_formal_runner_v1 as runner
from assumption_agent.benchmarks.musique_three_arm_formal_runner_v1 import (
    ARM_IDS,
    CONTEXT_SERIALIZATION_VERSION,
    ITEM_COUNT,
    ITEM_INPUT_VERSION,
    MODEL_ID,
    MODEL_OUTPUT_TOKEN_BUDGET,
    MODEL_REQUEST_BODY_BYTE_BUDGET,
    MuSiQueFormalRunnerError,
    MuSiQueNoReplayError,
    OFFICIAL_ADAPTER_ID,
    OFFICIAL_BINDING_RECEIPT_RELATIVE,
    PRIVATE_INDEX_VERSION,
    PROMPT_VERSION,
    WORKER_PLAN_VERSION,
    WORK_UNIT_COUNT,
    current_runner_implementation_binding,
    provider_identity_binding,
    run_formal_musique_three_arm,
    run_synthetic_musique_three_arm_for_tests,
)
from assumption_agent.benchmarks.musique_typed_retriever_formation_v1 import (
    OPERATOR_VERSION,
    TypedRetrievalProgram,
    current_implementation_binding,
)
from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
)
from replication_runtime.noaa_gsod_v1.development_runner import (
    ProviderTransportUnavailable,
)
from tests.test_musique_development_custody_freeze_v1 import (
    _custody as _protocol_custody,
    _freeze as _protocol_freeze,
)


PLUS_CHANNEL = "ruoli_plus"
PRO_CHANNEL = "ruoli_pro"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_env(path: Path, *, key: str) -> None:
    path.write_text(
        "\n".join(
            (
                "ASSUMPTION_V2_API_BASE=https://ruoli.dev",
                f"ASSUMPTION_V2_API_KEY={key}",
                f"ASSUMPTION_V2_MODEL={MODEL_ID}",
                "ASSUMPTION_V2_PROVIDER_CHAIN=openai_compatible",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _write_typed_freeze(root: Path) -> tuple[Path, Path]:
    typed_root = root / "typed"
    typed_root.mkdir()
    implementation = current_implementation_binding()
    program = TypedRetrievalProgram(
        seed_algorithm="bm25",
        title_weight=2,
        text_weight=1,
        expansion_mode="entity_token_one_hop",
        expansion_weight=1,
    )
    receipt_body = {
        "schema": "musique_typed_retriever_formation_v1_receipt",
        "formation_version": "synthetic-test-only",
        "implementation": implementation,
        "offline_contract": {
            "development_execution_authorized": False,
            "online_evaluator_calls": 0,
        },
        "raw_content_persisted": False,
        "selection_receipt": {"selected_program_hash": program.program_hash},
    }
    receipt = {**receipt_body, "receipt_hash": stable_hash(receipt_body)}
    receipt_path = typed_root / "formation.receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    envelope = {
        "formation_receipt_hash": receipt["receipt_hash"],
        "implementation": implementation,
        "operator_version": OPERATOR_VERSION,
        "program": program.to_dict(),
        "program_hash": program.program_hash,
        "raw_content_persisted": False,
    }
    program_path = typed_root / "frozen_program.json"
    program_path.write_text(json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_path, program_path


def _paragraphs(index: int) -> list[dict[str, Any]]:
    return [
        {"idx": 0, "title": "Alpha", "paragraph_text": "Alpha links Beta."},
        {"idx": 1, "title": "Noise One", "paragraph_text": "Copper circles quietly."},
        {"idx": 2, "title": "Noise Two", "paragraph_text": "Silver squares remain."},
        {"idx": 3, "title": "Noise Three", "paragraph_text": "Green triangles wait."},
        {"idx": 4, "title": "Noise Four", "paragraph_text": "Orange lines continue."},
        {"idx": 5, "title": "Noise Five", "paragraph_text": "Violet arcs pause."},
        {
            "idx": 6,
            "title": "Beta",
            "paragraph_text": f"Beta reveals secret-{index}.",
        },
    ]


def _fixture(tmp_path: Path) -> dict[str, Any]:
    custody = _protocol_custody(tmp_path)
    frozen = _protocol_freeze(tmp_path, custody)
    development = frozen["development_root"]
    plus_env = tmp_path / "plus.env"
    pro_env = tmp_path / "pro.env"
    _write_env(plus_env, key="synthetic-plus-key")
    _write_env(pro_env, key="synthetic-pro-key")
    return {
        "development": development,
        "public_freeze": frozen["public_path"],
        "custody_receipt": custody["receipt"],
        "acquisition_receipt": custody["acquisition"],
        "private_index": development / "private_index.json",
        "plus_env": plus_env,
        "pro_env": pro_env,
        "worker": json.loads((development / "worker_plan.json").read_text()),
    }


def _official_retrieve(
    *,
    question: str,
    paragraphs: Sequence[Mapping[str, object]],
    work_root: Path,
) -> tuple[int, ...]:
    assert question
    assert all(set(row) == {"idx", "title", "paragraph_text"} for row in paragraphs)
    assert work_root.name
    return (0, 1, 2, 3, 4)


class _ParallelTransport:
    def __init__(self, *, plus_unavailable: bool = False) -> None:
        self.plus_unavailable = plus_unavailable
        self.barrier = threading.Barrier(WORK_UNIT_COUNT)
        self.lock = threading.Lock()
        self.generator_calls: list[tuple[str, str]] = []
        self.saw_complete_claim_set = True
        self.output_root: Path | None = None

    def complete(self, *, credential: Any, request: Any) -> str:
        if request.purpose == "provider_transport_canary":
            if self.plus_unavailable and "plus" in credential.channel_id:
                raise ProviderTransportUnavailable("synthetic unavailable")
            return "canary-ok"
        assert self.output_root is not None
        claim_files = tuple((self.output_root / "generation_state").glob("*/claim.json"))
        self.saw_complete_claim_set &= (
            (self.output_root / "generation.claim-set.json").is_file()
            and len(claim_files) == WORK_UNIT_COUNT
        )
        with self.lock:
            self.generator_calls.append((credential.channel_id, request.purpose))
        self.barrier.wait(timeout=30)
        _prefix, anonymous, arm = request.purpose.split(":", 2)
        ordinal = int(anonymous.rsplit("_", 1)[1])
        if arm == ARM_IDS[0]:
            return "not-json"
        return json.dumps({"answer": f"Answer {ordinal}"}, separators=(",", ":"))


def test_synthetic_protocol_is_parallel_gold_delayed_and_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["development"] / "formal_execution"
    transport = _ParallelTransport()
    transport.output_root = output
    private_reads: list[int] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any):
        if self.resolve(strict=False) == fixture["private_index"].resolve():
            private_reads.append(len(transport.generator_calls))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    primary = runner.evaluate_aliases_primary
    secondary = runner.evaluate_aliases_secondary

    def primary_guard(prediction: str, aliases: Sequence[str]):
        assert prediction != ""
        return primary(prediction, aliases)

    def secondary_guard(prediction: str, aliases: Sequence[str]):
        assert prediction != ""
        return secondary(prediction, aliases)

    monkeypatch.setattr(runner, "evaluate_aliases_primary", primary_guard)
    monkeypatch.setattr(runner, "evaluate_aliases_secondary", secondary_guard)
    report = run_synthetic_musique_three_arm_for_tests(
        development_root=fixture["development"],
        public_freeze_path=fixture["public_freeze"],
        custody_receipt_path=fixture["custody_receipt"],
        acquisition_receipt_path=fixture["acquisition_receipt"],
        output_root=output,
        plus_env_file=fixture["plus_env"],
        pro_env_file=fixture["pro_env"],
        plus_channel_id=PLUS_CHANNEL,
        pro_channel_id=PRO_CHANNEL,
        transport=transport,
        official_retrieve=_official_retrieve,
    )
    assert len(transport.generator_calls) == WORK_UNIT_COUNT
    assert transport.saw_complete_claim_set is True
    assert private_reads == [0, WORK_UNIT_COUNT, WORK_UNIT_COUNT]
    assert report["formal_evidence"] is False
    assert report["formal_evidence_valid"] is False
    assert report["call_ledger"]["ruoli_external_calls"]["generator_calls"] == WORK_UNIT_COUNT
    assert report["call_ledger"]["retries"] == 0
    assert report["concurrency"]["observed_maximum_model_calls"] == WORK_UNIT_COUNT
    assert report["offline_evaluation"]["intention_to_treat_invalid_generator_output_as_incorrect"] is True
    canonical = report["offline_evaluation"]["arm_aggregates"][ARM_IDS[0]]
    typed = report["offline_evaluation"]["arm_aggregates"][ARM_IDS[1]]
    assert canonical["generator_output_contract_valid_count"] == 0
    assert typed["answer_exact"]["sum"] == {"denominator": 1, "numerator": ITEM_COUNT}
    exact_pair = report["offline_evaluation"]["pairwise_item_counts"][
        "assumption_minus_canonical"
    ]["answer_exact"]
    assert exact_pair == {
        "gain_count": ITEM_COUNT,
        "harm_count": 0,
        "paired_count": ITEM_COUNT,
        "paired_delta_sum": {"denominator": 1, "numerator": ITEM_COUNT},
        "tie_count": 0,
    }
    public_text = (output / "report.public.json").read_text(encoding="utf-8")
    for forbidden in (
        "Answer 0",
        "Which city is linked",
        "Alpha links Beta",
        "accepted_aliases",
        '"prediction"',
        '"support_indices"',
    ):
        assert forbidden not in public_text
    assert (output / "evaluation.private.json").is_file()


def test_plus_transport_unavailable_selects_pro_for_entire_18_call_batch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["development"] / "formal_execution"
    transport = _ParallelTransport(plus_unavailable=True)
    transport.output_root = output
    report = run_synthetic_musique_three_arm_for_tests(
        development_root=fixture["development"],
        public_freeze_path=fixture["public_freeze"],
        custody_receipt_path=fixture["custody_receipt"],
        acquisition_receipt_path=fixture["acquisition_receipt"],
        output_root=output,
        plus_env_file=fixture["plus_env"],
        pro_env_file=fixture["pro_env"],
        plus_channel_id=PLUS_CHANNEL,
        pro_channel_id=PRO_CHANNEL,
        transport=transport,
        official_retrieve=_official_retrieve,
    )
    assert report["selected_provider_label"] == "pro"
    assert report["call_ledger"]["ruoli_external_calls"]["constant_canary_calls"] == 2
    assert {label for label, _purpose in transport.generator_calls} == {PRO_CHANNEL}


def test_formal_entry_has_no_injected_dependencies_and_consumed_output_cannot_replay(
    tmp_path: Path,
) -> None:
    parameters = set(inspect.signature(run_formal_musique_three_arm).parameters)
    assert "transport" not in parameters
    assert "official_retrieve" not in parameters

    fixture = _fixture(tmp_path)
    output = fixture["development"] / "formal_execution"
    transport = _ParallelTransport()
    transport.output_root = output
    run_synthetic_musique_three_arm_for_tests(
        development_root=fixture["development"],
        public_freeze_path=fixture["public_freeze"],
        custody_receipt_path=fixture["custody_receipt"],
        acquisition_receipt_path=fixture["acquisition_receipt"],
        output_root=output,
        plus_env_file=fixture["plus_env"],
        pro_env_file=fixture["pro_env"],
        plus_channel_id=PLUS_CHANNEL,
        pro_channel_id=PRO_CHANNEL,
        transport=transport,
        official_retrieve=_official_retrieve,
    )
    with pytest.raises(MuSiQueNoReplayError, match="replay"):
        run_synthetic_musique_three_arm_for_tests(
            development_root=fixture["development"],
            public_freeze_path=fixture["public_freeze"],
            custody_receipt_path=fixture["custody_receipt"],
            acquisition_receipt_path=fixture["acquisition_receipt"],
            output_root=output,
            plus_env_file=fixture["plus_env"],
            pro_env_file=fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            transport=transport,
            official_retrieve=_official_retrieve,
        )
    shutil.rmtree(output)
    with pytest.raises(MuSiQueNoReplayError, match="already consumed|replay"):
        run_synthetic_musique_three_arm_for_tests(
            development_root=fixture["development"],
            public_freeze_path=fixture["public_freeze"],
            custody_receipt_path=fixture["custody_receipt"],
            acquisition_receipt_path=fixture["acquisition_receipt"],
            output_root=output,
            plus_env_file=fixture["plus_env"],
            pro_env_file=fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            transport=transport,
            official_retrieve=_official_retrieve,
        )

    oversized = "x" * MODEL_REQUEST_BODY_BYTE_BUDGET
    with pytest.raises(MuSiQueFormalRunnerError, match="64 KiB"):
        runner._generator_request(
            item_id="synthetic",
            arm=ARM_IDS[0],
            question="q",
            context=oversized,
        )


def test_gold_in_worker_input_fails_closed_before_generator_calls(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source = fixture["development"] / "inputs" / "development_item_00.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["answers"] = ["forbidden"]
    source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    output = fixture["development"] / "formal_execution"
    transport = _ParallelTransport()
    transport.output_root = output
    with pytest.raises(MuSiQueFormalRunnerError, match="schema|gold|hash|drift"):
        run_synthetic_musique_three_arm_for_tests(
            development_root=fixture["development"],
            public_freeze_path=fixture["public_freeze"],
            custody_receipt_path=fixture["custody_receipt"],
            acquisition_receipt_path=fixture["acquisition_receipt"],
            output_root=output,
            plus_env_file=fixture["plus_env"],
            pro_env_file=fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            transport=transport,
            official_retrieve=_official_retrieve,
        )
    assert transport.generator_calls == []


def test_actual_use_rechecks_item_and_typed_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    worker = json.loads(
        (fixture["development"] / runner.WORKER_PLAN_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    item = worker["items"][0]
    item_path = fixture["development"] / item["input_relative_path"]
    original_item = item_path.read_bytes()
    item_path.write_bytes(original_item + b" ")
    with pytest.raises(MuSiQueFormalRunnerError, match="actual use point"):
        runner._load_gold_free_item(item_path, item)
    item_path.write_bytes(original_item)

    typed = worker["typed_binding"]
    receipt_path = fixture["development"] / typed[
        "formation_receipt_relative_path"
    ]
    program_path = fixture["development"] / typed[
        "frozen_program_relative_path"
    ]
    program_path.write_bytes(program_path.read_bytes() + b" ")
    with pytest.raises(MuSiQueFormalRunnerError, match="actual use point"):
        runner._prepare_item(
            development_root=fixture["development"],
            item=item,
            typed_binding=typed,
            typed_program_path=program_path,
            typed_receipt_path=receipt_path,
            official_retrieve=_official_retrieve,
            official_work_root=tmp_path / "official-work",
        )


def test_atomic_preclaim_failure_consumes_authorization_and_writes_safe_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["development"] / "formal_execution"
    transport = _ParallelTransport()
    transport.output_root = output
    original = runner.atomic_write_hashed_json_v2

    def fail_claim_set(path: str | Path, body: Mapping[str, Any], **kwargs: Any):
        if Path(path).name == runner.CLAIM_SET_FILENAME:
            raise OSError("synthetic atomic claim-set failure")
        return original(path, body, **kwargs)

    monkeypatch.setattr(runner, "atomic_write_hashed_json_v2", fail_claim_set)
    with pytest.raises(OSError, match="claim-set"):
        run_synthetic_musique_three_arm_for_tests(
            development_root=fixture["development"],
            public_freeze_path=fixture["public_freeze"],
            custody_receipt_path=fixture["custody_receipt"],
            acquisition_receipt_path=fixture["acquisition_receipt"],
            output_root=output,
            plus_env_file=fixture["plus_env"],
            pro_env_file=fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            transport=transport,
            official_retrieve=_official_retrieve,
        )
    assert transport.generator_calls == []
    assert (fixture["development"] / runner.CONSUMPTION_MARKER_RELATIVE_PATH).is_file()
    failures = tuple(fixture["development"].glob("execution.failure.*.json"))
    assert failures
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["private_prediction_or_terminal_may_be_persisted"] is False
    assert failure["gold_open_may_have_started"] is False
    assert failure["private_evaluation_may_have_started"] is False
    shutil.rmtree(output)
    with pytest.raises(MuSiQueNoReplayError, match="consumed|replay"):
        run_synthetic_musique_three_arm_for_tests(
            development_root=fixture["development"],
            public_freeze_path=fixture["public_freeze"],
            custody_receipt_path=fixture["custody_receipt"],
            acquisition_receipt_path=fixture["acquisition_receipt"],
            output_root=output,
            plus_env_file=fixture["plus_env"],
            pro_env_file=fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            transport=transport,
            official_retrieve=_official_retrieve,
        )


def test_post_generation_failure_receipt_is_stage_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["development"] / "formal_execution"
    transport = _ParallelTransport()
    transport.output_root = output

    def fail_after_terminal_join(**_kwargs: Any):
        raise RuntimeError("synthetic post-generation evaluator failure")

    monkeypatch.setattr(runner, "_offline_evaluate", fail_after_terminal_join)
    with pytest.raises(RuntimeError, match="post-generation"):
        run_synthetic_musique_three_arm_for_tests(
            development_root=fixture["development"],
            public_freeze_path=fixture["public_freeze"],
            custody_receipt_path=fixture["custody_receipt"],
            acquisition_receipt_path=fixture["acquisition_receipt"],
            output_root=output,
            plus_env_file=fixture["plus_env"],
            pro_env_file=fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            transport=transport,
            official_retrieve=_official_retrieve,
        )
    failures = tuple(fixture["development"].glob("execution.failure.*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["private_prediction_or_terminal_may_be_persisted"] is True
    assert failure["gold_open_may_have_started"] is True
    assert failure["private_evaluation_may_have_started"] is True
    assert failure["public_content_leak_detected"] is False


def test_formal_core_rejects_injection_symlink_root_and_direct_cli_bootstraps(
    tmp_path: Path,
) -> None:
    with pytest.raises(MuSiQueFormalRunnerError, match="cannot accept injected"):
        runner._run_core(
            development_root=tmp_path,
            public_freeze_path=tmp_path / "freeze.json",
            custody_receipt_path=tmp_path / "custody.json",
            acquisition_receipt_path=tmp_path / "acquisition.json",
            output_root=tmp_path / "formal_execution",
            plus_env_file=tmp_path / "plus.env",
            pro_env_file=tmp_path / "pro.env",
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            formal_runtime=runner._FormalRuntimePaths(
                runtime_python=Path(sys.executable),
                local_llm_model=tmp_path,
                local_embedding_model=tmp_path,
            ),
            synthetic_transport=_ParallelTransport(),
            synthetic_official_retrieve=_official_retrieve,
        )

    synthetic_formal_fixture = _fixture(tmp_path / "formal-anchor-fixture")
    with pytest.raises(
        MuSiQueFormalRunnerError, match="registered public trust roots"
    ):
        runner._run_core(
            development_root=synthetic_formal_fixture["development"],
            public_freeze_path=synthetic_formal_fixture["public_freeze"],
            custody_receipt_path=synthetic_formal_fixture["custody_receipt"],
            acquisition_receipt_path=synthetic_formal_fixture[
                "acquisition_receipt"
            ],
            output_root=synthetic_formal_fixture["development"]
            / "formal_execution",
            plus_env_file=synthetic_formal_fixture["plus_env"],
            pro_env_file=synthetic_formal_fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            formal_runtime=runner._FormalRuntimePaths(
                runtime_python=Path(sys.executable),
                local_llm_model=tmp_path,
                local_embedding_model=tmp_path,
            ),
        )

    fixture = _fixture(tmp_path / "symlink-fixture")
    linked = tmp_path / "linked-development"
    linked.symlink_to(fixture["development"], target_is_directory=True)
    transport = _ParallelTransport()
    transport.output_root = linked / "formal_execution"
    with pytest.raises(MuSiQueFormalRunnerError, match="symbolic-link"):
        run_synthetic_musique_three_arm_for_tests(
            development_root=linked,
            public_freeze_path=fixture["public_freeze"],
            custody_receipt_path=fixture["custody_receipt"],
            acquisition_receipt_path=fixture["acquisition_receipt"],
            output_root=linked / "formal_execution",
            plus_env_file=fixture["plus_env"],
            pro_env_file=fixture["pro_env"],
            plus_channel_id=PLUS_CHANNEL,
            pro_channel_id=PRO_CHANNEL,
            transport=transport,
            official_retrieve=_official_retrieve,
        )
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "--public-freeze" in completed.stdout
