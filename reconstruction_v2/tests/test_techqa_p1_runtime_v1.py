from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import time

import pytest

from assumption_agent.benchmarks import (
    techqa_p0_public_source_qualification_v1 as p0,
)
from assumption_agent.benchmarks import techqa_p1_formal_v1 as formal
from assumption_agent.benchmarks import (
    techqa_p1_official_hipporag_v1 as adapter,
)
from assumption_agent.benchmarks import techqa_p1_runtime_v1 as runtime


def _family_text(family: str, token: str) -> tuple[str, str]:
    if family == formal.INFORMATION:
        return (
            f"Reference details for {token}",
            f"Configuration metadata concerning {token}.",
        )
    if family == formal.PROCEDURE:
        return (
            f"How to configure {token}",
            f"Steps for installing {token}.",
        )
    if family == formal.TROUBLESHOOT:
        return (
            f"Fix error {token}",
            f"The component {token} cannot start.",
        )
    raise AssertionError(family)


def _raw_sources() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    shared = {
        f"shared-{index:02d}": {
            "_id": f"shared-{index:02d}",
            "text": f"Unrelated support body {index} with token S-{index}.",
            "title": f"Shared technote {index}",
        }
        for index in range(49)
    }
    corpus: dict[str, object] = dict(shared)
    train: list[dict[str, object]] = []
    dev: list[dict[str, object]] = []
    for split, per_family, target in (
        ("train", 48, train),
        ("dev", 24, dev),
    ):
        for family in formal.FAMILIES:
            for index in range(per_family):
                token = f"{split}-{family.casefold()}-unique-{index:03d}"
                gold = f"gold-{token}"
                title, text = _family_text(family, token)
                gold_text = f"Exact answer material for {token}."
                corpus[gold] = {
                    "_id": gold,
                    "text": gold_text,
                    "title": f"Technote for {token}",
                }
                target.append(
                    {
                        "ANSWERABLE": "Y",
                        "DOC_IDS": sorted([*shared, gold]),
                        "DOCUMENT": gold,
                        "END_OFFSET": len("Exact"),
                        "QUESTION_ID": f"question-{token}",
                        "QUESTION_TEXT": text,
                        "QUESTION_TITLE": title,
                        "START_OFFSET": 0,
                    }
                )
    train.append(
        {
            "ANSWERABLE": "Y",
            "DOC_IDS": sorted(
                [*shared, "gold-train-information-unique-000"]
            ),
            "DOCUMENT": "shared-00",
            "END_OFFSET": 9,
            "QUESTION_ID": "ineligible-over-bound-query",
            "QUESTION_TEXT": "",
            "QUESTION_TITLE": "X" * 5_000,
            "START_OFFSET": 0,
        }
    )
    corpus["never-selected"] = {
        "_id": "never-selected",
        "text": "This entire unreferenced document must not be retained.",
        "title": "Unreferenced",
    }
    return train, dev, corpus


def _write_bytes(path: Path, raw: bytes, mode: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return hashlib.sha256(raw).hexdigest()


def _write_json(
    path: Path,
    value: Mapping[str, object] | list[object],
    mode: int,
    *,
    canonical: bool,
) -> str:
    raw = (
        runtime.canonical_bytes(value)
        if canonical
        else json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return _write_bytes(path, raw, mode)


def _asset(path: Path, sha256: str, mode: int) -> dict[str, object]:
    return {"mode": mode, "path": str(path), "sha256": sha256}


def _runtime_paths(root: Path) -> dict[str, object]:
    directories = {}
    for name in (
        "project_root",
        "typed_site_root",
        "official_overlay_root",
        "hipporag_source_root",
        "p16_site_root",
        "official_base_site_root",
        "smollm_model_root",
        "minilm_model_root",
    ):
        path = root / "runtime_assets" / name
        path.mkdir(parents=True)
        directories[name] = str(path)
    files = {}
    for name in ("typed_python", "official_python", "strace_path"):
        path = root / "runtime_assets" / name
        path.write_bytes(b"synthetic executable")
        path.chmod(0o700)
        files[name] = str(path)
    return {
        **directories,
        **files,
    }


@dataclass(frozen=True)
class _Fixture:
    config_path: Path
    work_root: Path
    config: runtime.FormalConfig
    secret: bytes
    known_private_token: str


@pytest.fixture
def secure_tmp_path() -> Iterable[Path]:
    root = Path(tempfile.mkdtemp(prefix="techqa-runtime-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def secure_module_path() -> Iterable[Path]:
    root = Path(
        tempfile.mkdtemp(prefix="techqa-runtime-module-", dir="/tmp")
    )
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fixture(root: Path) -> _Fixture:
    train, dev, corpus = _raw_sources()
    source_root = root / "source"
    train_path = source_root / p0.TRAIN_QA_BASENAME
    dev_path = source_root / p0.DEV_QA_BASENAME
    corpus_path = source_root / p0.CORPUS_BASENAME
    train_sha = _write_json(
        train_path, train, 0o600, canonical=False
    )
    dev_sha = _write_json(dev_path, dev, 0o600, canonical=False)
    corpus_sha = _write_json(
        corpus_path, corpus, 0o600, canonical=False
    )

    def eligibility_rows(
        rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        result = []
        for row in rows:
            if row["QUESTION_ID"] == "ineligible-over-bound-query":
                continue
            title = str(row["QUESTION_TITLE"])
            text = str(row["QUESTION_TEXT"])
            serialized = formal.core.serialize_query_text(title, text)
            result.append(
                {
                    "family": formal.operational_family(title, text),
                    "normalized_query_sha256": hashlib.sha256(
                        formal.core.normalize_text(
                            serialized,
                            field="synthetic eligibility query",
                        ).encode("utf-8")
                    ).hexdigest(),
                    "question_id": row["QUESTION_ID"],
                }
            )
        return result

    eligible_by_split = {
        "TRAIN": eligibility_rows(train),
        "DEV": eligibility_rows(dev),
    }
    eligibility = p0.self_hashed(
        {
            "cohort_HMAC_action_qrel_evaluator_or_score_count": 0,
            "eligibility_rule_version": (
                runtime.P0_ELIGIBILITY_RULE_VERSION
            ),
            "eligible_answerable_rows_by_split": eligible_by_split,
            "eligible_row_count_by_split": {
                split: len(rows)
                for split, rows in eligible_by_split.items()
            },
            "schema": runtime.P0_ELIGIBILITY_SCHEMA,
            "source_member_content_sha256": {
                p0.TRAIN_QA_BASENAME: train_sha,
                p0.DEV_QA_BASENAME: dev_sha,
                p0.CORPUS_BASENAME: corpus_sha,
            },
            "study_id": p0.STUDY_ID,
        }
    )
    eligibility_path = root / "receipts" / "eligibility.private.json"
    eligibility_sha = _write_json(
        eligibility_path, eligibility, 0o600, canonical=True
    )
    receipt = p0.self_hashed(
        {
            "access_boundary": {
                "action_model_qrel_evaluator_or_score_count": 0,
                "cohort_assignment_or_secret_count": 0,
                "individual_query_document_or_span_value_output_count": 0,
                "online_or_API_evaluation_count": 0,
                "source_archive_full_extraction_count": 0,
                "source_archive_whitelisted_member_extraction_count": 3,
            },
            "archive": {
                "target_members": {
                    p0.TRAIN_QA_BASENAME: {
                        "content_sha256": train_sha,
                        "size_bytes": train_path.stat().st_size,
                    },
                    p0.DEV_QA_BASENAME: {
                        "content_sha256": dev_sha,
                        "size_bytes": dev_path.stat().st_size,
                    },
                    p0.CORPUS_BASENAME: {
                        "content_sha256": corpus_sha,
                        "size_bytes": corpus_path.stat().st_size,
                    },
                }
            },
            "qualified_source_persistence": {
                "exact_private_regular_file_count": 3,
                "full_archive_or_nonwhitelisted_member_persistence_count": 0,
                "member_byte_identity_verified_against_receipt_count": 3,
                "mode": "0600",
            },
            "private_eligibility_manifest_binding": {
                "eligible_row_count_by_split": {
                    split: len(rows)
                    for split, rows in eligible_by_split.items()
                },
                "file_sha256": eligibility_sha,
                "self_sha256": eligibility["self_sha256"],
            },
            "schema": runtime.P0_RECEIPT_SCHEMA,
            "status": runtime.P0_QUALIFIED_STATUS,
            "study_id": p0.STUDY_ID,
        }
    )
    receipt_path = root / "receipts" / "p0.safe.json"
    receipt_sha = _write_json(
        receipt_path, receipt, 0o600, canonical=True
    )

    freeze = runtime.self_hashed(
        {
            "schema": "synthetic_implementation_freeze_v1",
            "status": "frozen_before_source_access",
            "study_id": formal.STUDY_ID,
        }
    )
    freeze_path = root / "freeze" / "implementation.json"
    freeze_sha = _write_json(
        freeze_path, freeze, 0o400, canonical=True
    )
    secret = b"s" * formal.HMAC_SECRET_BYTES
    secret_path = root / "secret" / "hmac.private.bin"
    secret_sha = _write_bytes(secret_path, secret, 0o400)
    work_root = root / "formal_work"
    config_body = {
        "eligibility_manifest": _asset(
            eligibility_path, eligibility_sha, 0o600
        ),
        "hmac_secret": _asset(secret_path, secret_sha, 0o400),
        "implementation_freeze": _asset(
            freeze_path, freeze_sha, 0o400
        ),
        "qualification_receipt": _asset(
            receipt_path, receipt_sha, 0o600
        ),
        "runtime": {
            "gpu_ids": list(runtime.GPU_IDS),
            "official_timeout_seconds": 30,
            "paths": _runtime_paths(root),
        },
        "schema": runtime.CONFIG_SCHEMA,
        "source": {
            p0.TRAIN_QA_BASENAME: _asset(
                train_path, train_sha, 0o600
            ),
            p0.DEV_QA_BASENAME: _asset(
                dev_path, dev_sha, 0o600
            ),
            p0.CORPUS_BASENAME: _asset(
                corpus_path, corpus_sha, 0o600
            ),
        },
        "study_id": formal.STUDY_ID,
        "work_root": str(work_root),
    }
    config_payload = runtime.self_hashed(config_body)
    config_path = root / "formal.config.json"
    _write_json(config_path, config_payload, 0o400, canonical=True)
    return _Fixture(
        config_path=config_path,
        work_root=work_root,
        config=runtime.load_config(config_path),
        secret=secret,
        known_private_token="question-train-information-unique-000",
    )


def _json_kvitems(source: object) -> Iterable[tuple[str, object]]:
    raw = source.read()  # type: ignore[attr-defined]
    value = json.loads(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value.items()


def _synthetic_adapter_output(
    public_input: Mapping[str, object],
) -> dict[str, object]:
    cluster = adapter.validate_input(public_input)
    marker = adapter._attempt_marker(cluster)
    rows = [
        {
            "query_ordinal": row.ordinal,
            "top5_document_ordinals": list(range(adapter.TOP_K)),
        }
        for row in cluster.queries
    ]
    return adapter._self_hashed(
        {
            "attempt_marker_file_sha256": hashlib.sha256(
                adapter.canonical_bytes(marker, newline=True)
            ).hexdigest(),
            "attempt_marker_self_sha256": marker["self_sha256"],
            "cluster_ordinal": cluster.cluster_ordinal,
            "document_count": len(cluster.documents),
            "document_serialization": adapter.DOCUMENT_SERIALIZATION,
            "document_serialized_sha256": (
                cluster.document_serialized_sha256
            ),
            "fresh_index_create_count": 1,
            "index_file_count": 1,
            "index_lifecycle": adapter.INDEX_LIFECYCLE,
            "index_total_bytes": 1,
            "index_tree_sha256": "1" * 64,
            "inner_block": adapter.INNER_BLOCK,
            "inner_build_index_call_count": 1,
            "inner_input_sha256": adapter.inner_worker.stable_hash(
                adapter.inner_payload(cluster)
            ),
            "inner_output_sha256": "2" * 64,
            "inner_receipt_sha256": "3" * 64,
            "inner_retrieval_index_call_count": 0,
            "inner_serialization": adapter.INNER_SERIALIZATION,
            "online_or_API_evaluator_call_count": 0,
            "outer_binding_sha256": adapter.outer_binding(cluster),
            "outer_input_self_sha256": cluster.self_sha256,
            "query_count": len(cluster.queries),
            "query_serialization": adapter.QUERY_SERIALIZATION,
            "query_serialized_sha256": (
                cluster.query_serialized_sha256
            ),
            "retry_replay_resample_count": 0,
            "rows": rows,
            "schema": adapter.OUTPUT_SCHEMA,
            "stage": cluster.stage,
            "status": "passed_once",
            "study_id": adapter.STUDY_ID,
        }
    )


class _FakeLauncher:
    def __init__(self, *, pause: float = 0.0) -> None:
        self.pause = pause
        self.lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        public_input: Mapping[str, object],
        cluster_root: Path,
        gpu_id: str,
    ) -> Mapping[str, object]:
        cluster_root.mkdir(mode=0o700)
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            call = {
                "cluster": public_input["cluster_ordinal"],
                "gpu": gpu_id,
                "stage": public_input["stage"],
                "start": time.monotonic(),
            }
            self.calls.append(call)
        try:
            if self.pause:
                time.sleep(self.pause)
            return _synthetic_adapter_output(public_input)
        finally:
            with self.lock:
                call["end"] = time.monotonic()
                self.active -= 1


@pytest.fixture(scope="module")
def prepared_fixture(
    secure_module_path: Path,
) -> tuple[_Fixture, formal.PreparedStudy, runtime.DocumentLoadReceipt]:
    fixture = _fixture(secure_module_path)
    qualification, _freeze = runtime._verify_supporting_receipts(
        fixture.config
    )
    eligibility = runtime._load_eligibility_manifest(
        fixture.config,
        qualification_receipt=qualification,
    )
    source, receipt = runtime.load_verified_source(
        fixture.config,
        secret=fixture.secret,
        eligibility=eligibility,
        kvitems_factory=_json_kvitems,
    )
    prepared = runtime.prepare_study(source, secret=fixture.secret)
    return fixture, prepared, receipt


def test_hash_mode_binding_and_bounded_selected_document_stream(
    prepared_fixture: tuple[
        _Fixture,
        formal.PreparedStudy,
        runtime.DocumentLoadReceipt,
    ],
) -> None:
    fixture, prepared, receipt = prepared_fixture
    assert len(prepared.source.training_questions) == 144
    assert len(prepared.source.dev_questions) == 72
    assert "ineligible-over-bound-query" not in {
        row.question_id
        for row in (
            prepared.source.training_questions
            + prepared.source.dev_questions
        )
    }
    assert receipt.candidate_reference_count == 10_800
    assert (
        receipt.candidate_reference_count
        == runtime.MAX_SELECTED_CANDIDATE_REFERENCES
    )
    assert receipt.retained_unreferenced_document_count == 0
    assert "never-selected" not in prepared.source.document_by_id
    assert receipt.corpus_document_count == (
        receipt.selected_unique_document_count + 1
    )

    fixture.config.training_q_a.path.chmod(0o644)
    with pytest.raises(
        runtime.TechqaP1RuntimeError, match="metadata drifted"
    ):
        runtime._verify_supporting_receipts(fixture.config)
    fixture.config.training_q_a.path.chmod(0o600)


def test_two_gpu_schedule_is_exactly_two_complete_waves(
    prepared_fixture: tuple[
        _Fixture,
        formal.PreparedStudy,
        runtime.DocumentLoadReceipt,
    ],
    secure_tmp_path: Path,
) -> None:
    _fixture_value, prepared, _receipt = prepared_fixture
    launcher = _FakeLauncher(pause=0.03)
    run = runtime.run_hippo_stage(
        prepared.a_hold,
        launcher=launcher,
        stage_root=secure_tmp_path / "A_hold",
    )
    assert run.stage == formal.A_HOLD
    assert launcher.maximum_active == 2
    by_cluster = {
        int(row["cluster"]): row for row in launcher.calls
    }
    assert {
        cluster: row["gpu"] for cluster, row in by_cluster.items()
    } == {0: "0", 1: "1", 2: "0", 3: "1"}
    assert min(
        float(by_cluster[index]["start"]) for index in (2, 3)
    ) >= max(float(by_cluster[index]["end"]) for index in (0, 1))


def test_m_is_materialized_only_after_promotion_and_never_runs_hippo(
    prepared_fixture: tuple[
        _Fixture,
        formal.PreparedStudy,
        runtime.DocumentLoadReceipt,
    ],
    secure_tmp_path: Path,
) -> None:
    _fixture_value, prepared, _receipt = prepared_fixture
    failed_launcher = _FakeLauncher()
    failed = runtime.execute_prepared_study(
        prepared,
        launcher=failed_launcher,
        execution_root=secure_tmp_path / "failed_gate",
    )
    assert failed.result.safe_terminal["A_hold"][
        "promotion_passed"
    ] is False
    assert failed.result.m_search is None
    assert len(failed_launcher.calls) == 4

    comparison = formal.compare_exact_rows(
        left_arm="E1",
        right_arm="E0",
        rows=[
            (family, cluster, Fraction(1), Fraction(0))
            for cluster in range(4)
            for family in formal.FAMILIES
        ],
    )
    authorization = formal.authorize_m_search(comparison)
    assert authorization is not None
    m_stage = formal._materialize_m_search(prepared, authorization)
    private_archive = {"synthetic": "promotion-only-test"}
    private_hash = formal.stable_hash(private_archive)
    promoted_result = formal.FormalResult(
            safe_terminal={
                "A_hold": {"promotion_passed": True},
                "M_search_untouched_scope": (
                    "query_and_action_not_document_disjoint"
                ),
                "M_search": {
                    "actions_materialized_after_promotion": True
                },
                "cohort_gold_document_disjoint": False,
                "cohort_question_and_normalized_query_disjoint": True,
                "item_query_document_qrel_action_values_published": False,
                "online_or_API_evaluator_call_count": 0,
                "private_archive_sha256": private_hash,
                "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
                "shared_corpus_and_gold_overlap_allowed": True,
            },
        private_archive=private_archive,
        private_archive_sha256=private_hash,
        m_search=m_stage,
    )
    promoted_launcher = _FakeLauncher()
    promoted = runtime.execute_prepared_study(
        prepared,
        launcher=promoted_launcher,
        execution_root=secure_tmp_path / "passed_gate",
        finalize=lambda _results: promoted_result,
    )
    assert promoted.result.m_search is m_stage
    assert [row["stage"] for row in promoted_launcher.calls] == [
        formal.A_HOLD,
        formal.A_HOLD,
        formal.A_HOLD,
        formal.A_HOLD,
    ]
    with pytest.raises(
        runtime.TechqaP1RuntimeError, match="outside A_hold"
    ):
        runtime.public_cluster_bundles(m_stage)


def test_end_to_end_safe_terminal_and_durable_replay_rejection(
    secure_tmp_path: Path,
) -> None:
    fixture = _fixture(secure_tmp_path)
    launcher = _FakeLauncher()
    terminal = runtime.run_formal_once(
        config_path=fixture.config_path,
        launcher=launcher,
        kvitems_factory=_json_kvitems,
    )
    assert terminal["status"] == "terminal_complete_once"
    assert len(launcher.calls) == 4
    terminal_path = fixture.work_root / "formal_terminal.json"
    private_path = fixture.work_root / "formal.private.json"
    assert stat.S_IMODE(terminal_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    safe_text = json.dumps(terminal, sort_keys=True)
    assert fixture.known_private_token not in safe_text
    assert "techqa-work-v1-" not in safe_text
    assert terminal[
        "item_query_document_qrel_action_values_published"
    ] is False
    assert terminal["online_or_API_evaluator_call_count"] == 0

    with pytest.raises(
        runtime.TechqaP1RuntimeError, match="replay is forbidden"
    ):
        runtime.run_formal_once(
            config_path=fixture.config_path,
            launcher=_FakeLauncher(),
            kvitems_factory=_json_kvitems,
        )


def test_source_hash_failure_occurs_after_durable_attempt_and_is_terminal(
    secure_tmp_path: Path,
) -> None:
    fixture = _fixture(secure_tmp_path)
    original = fixture.config.training_q_a.path.read_bytes()
    mutated = bytes([original[0] ^ 1]) + original[1:]
    assert len(mutated) == len(original)
    fixture.config.training_q_a.path.write_bytes(mutated)
    fixture.config.training_q_a.path.chmod(0o600)
    terminal = runtime.run_formal_once(
        config_path=fixture.config_path,
        launcher=_FakeLauncher(),
        kvitems_factory=_json_kvitems,
    )
    assert terminal["status"] == "terminal_failed_once_no_retry"
    assert terminal["failure_stage"] == "source_load"
    assert (fixture.work_root / "formal.attempt.json").is_file()
    assert (fixture.work_root / "failure.private.json").is_file()
    assert terminal["retry_or_replay_authorized"] is False
    with pytest.raises(
        runtime.TechqaP1RuntimeError, match="replay is forbidden"
    ):
        runtime.run_formal_once(
            config_path=fixture.config_path,
            launcher=_FakeLauncher(),
            kvitems_factory=_json_kvitems,
        )
