from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import tarfile
from typing import Mapping, Sequence

import pytest

from assumption_agent.benchmarks import ebmnlp_p1_formal_controller_v1 as formal
from assumption_agent.benchmarks import ebmnlp_p1_source_qualification_v1 as source
from assumption_agent.benchmarks import ebmnlp_p1_typed_pico_core_v1 as core
from replication_runtime.ebmnlp_p1_official_v1 import contract as hippo


TRAIN_PMIDS = ("101", "102", "103", "104", "105", "106")
TEST_PMIDS = ("201", "202", "203", "204")


def _source_fixture(
    tmp_path: Path,
) -> tuple[Path, source.QualificationContract]:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        def regular(name: str, raw: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o600
            bundle.addfile(info, io.BytesIO(raw))

        root = source.ARCHIVE_ROOT
        regular(f"{root}/README.md", b"synthetic metadata\n")
        for split, pmids in (
            ("train", TRAIN_PMIDS),
            ("test/gold", TEST_PMIDS),
        ):
            for pmid in pmids:
                tokens = (
                    f"trial participant treatment outcome {pmid}\n"
                ).encode("ascii")
                regular(f"{root}/documents/{pmid}.tokens", tokens)
                regular(
                    f"{root}/documents/{pmid}.text",
                    f"Synthetic abstract {pmid}.\n".encode("ascii"),
                )
                regular(
                    f"{root}/documents/{pmid}.pos",
                    b"NN NN NN NN CD\n",
                )
                for role_index, role in enumerate(source.ROLE_ORDER):
                    labels = " ".join(
                        "1"
                        if (position + role_index) % 3 == 0
                        else "0"
                        for position in range(5)
                    )
                    regular(
                        f"{root}/annotations/aggregated/starting_spans/"
                        f"{role}/{split}/{pmid}.ann",
                        (labels + "\n").encode("ascii"),
                    )
    raw = output.getvalue()
    archive = tmp_path / "ebm_nlp_2_00.tar.gz"
    archive.write_bytes(raw)
    archive.chmod(0o600)
    contract = source.QualificationContract(
        archive_sha256=hashlib.sha256(raw).hexdigest(),
        archive_size_bytes=len(raw),
        total_public_abstract_count=10,
        train_abstract_count=6,
        test_abstract_count=4,
        blocks=source.BlockCounts(
            G_form=2,
            A_form=1,
            F_search=1,
            A_hold=1,
            M_search=1,
        ),
        maximum_archive_member_count=1_000,
        maximum_total_declared_member_bytes=10_000_000,
        maximum_document_member_bytes=1_000_000,
        maximum_label_member_bytes=1_000_000,
        maximum_ignored_regular_member_bytes=1_000_000,
        maximum_tokens_per_document=1_000,
    )
    return archive, contract


class _EmbeddingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        rows: list[tuple[float, ...]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = tuple(float(value + 1) for value in digest[:4])
            norm = math.sqrt(sum(value * value for value in raw))
            rows.append(tuple(value / norm for value in raw))
        return tuple(rows)

    def safe_runtime_receipt(self) -> Mapping[str, object]:
        return {
            "schema": (
                "ebmnlp_p1_local_minilm_embedder_v1_"
                "safe_runtime_receipt"
            ),
            "model_tree_sha256": (
                formal.EXPECTED_MINILM_TREE_SHA256
            ),
            "device": "synthetic-cpu",
            "dtype": "float64",
            "embedding_dimension": 4,
            "call_count": len(self.calls),
            "encoded_text_count": sum(len(row) for row in self.calls),
            "external_network_call_count": 0,
            "online_or_api_evaluator_call_count": 0,
            "retry_or_replay_count": 0,
        }


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    def vcount(self) -> int:
        return 3

    def ecount(self) -> int:
        return 2


class _HippoCore:
    def __init__(self) -> None:
        self.graph = _Graph()
        self.documents: list[str] = []

    def index(self, documents: list[str]) -> None:
        self.documents = list(documents)

    def retrieve(
        self, queries: list[str], *, num_to_retrieve: int
    ) -> list[_Solution]:
        assert num_to_retrieve == len(self.documents)
        return [
            _Solution(
                docs=list(self.documents),
                doc_scores=[
                    float(len(self.documents) - ordinal)
                    for ordinal in range(len(self.documents))
                ],
            )
            for _query in queries
        ]


class _HippoLauncher:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> Mapping[str, Mapping[str, object]]:
        self.batch_sizes.append(len(payloads))
        return {
            str(payload["abstract_work_id"]): (
                hippo.retrieve_abstract_with_core(
                    core=_HippoCore(), payload=payload
                )
            )
            for payload in payloads
        }

    def safe_runtime_receipt(self) -> Mapping[str, object]:
        workers = sum(self.batch_sizes)
        return {
            "schema": (
                "ebmnlp_p1_official_hipporag_batch_v2_"
                "safe_runtime_receipt"
            ),
            "status": (
                "complete_offline_outputs_verified_indexes_destroyed"
            ),
            "gpu_assignment": ["0", "1"],
            "maximum_process_count": 2,
            "maximum_processes_per_gpu": 1,
            "observed_process_peak": min(2, max(self.batch_sizes, default=0)),
            "observed_process_peak_by_gpu": {
                "0": int(workers > 0),
                "1": int(workers > 1),
            },
            "worker_attempt_count": workers,
            "worker_completed_count": workers,
            "worker_completed_count_by_gpu": {
                "0": (workers + 1) // 2,
                "1": workers // 2,
            },
            "index_destroyed_count": workers,
            "worker_cuda_attested_count": workers,
            "worker_cuda_attested_count_by_gpu": {
                "0": (workers + 1) // 2,
                "1": workers // 2,
            },
            "worker_cuda_receipt_count": workers,
            "worker_cuda_receipt_set_sha256": "a" * 64,
            "attempted_network_syscall_count": 0,
            "denied_network_syscall_count": 0,
            "local_AF_UNIX_network_syscall_count": 0,
            "network_isolation_mechanism": (
                "outer_systemd_AF_UNIX_only_IPAddressDeny_any_plus_"
                "passive_strace_socket_connect"
            ),
            "external_network_call_count": 0,
            "online_or_api_evaluator_call_count": 0,
            "retry_or_replay_count": 0,
        }


def _test_binding() -> formal.FormalExecutionBinding:
    return formal.FormalExecutionBinding(
        implementation_freeze_sha256="1" * 64,
        runtime_fingerprint_sha256="2" * 64,
        source_free_canary_sha256="3" * 64,
        execution_config_sha256="4" * 64,
        execution_freeze_sha256="5" * 64,
        live_execution_attestation_sha256="6" * 64,
        source_archive_sha256=source.FORMAL_CONTRACT.archive_sha256,
        minilm_tree_sha256=formal.EXPECTED_MINILM_TREE_SHA256,
        hipporag_source_tree_sha256=(
            formal.EXPECTED_HIPPORAG_SOURCE_TREE_SHA256
        ),
        hipporag_llm_tree_sha256=(
            formal.EXPECTED_HIPPORAG_LLM_TREE_SHA256
        ),
    )


def test_formal_runtime_receipt_requires_exact_balanced_two_gpu_cuda(
) -> None:
    launcher = _HippoLauncher()
    launcher.batch_sizes.append(64)
    formal._executor_runtime_receipt(
        launcher,
        kind="hipporag",
        formal_scope=True,
        expected_hippo_workers=64,
    )
    drifted = dict(launcher.safe_runtime_receipt())
    drifted["worker_cuda_attested_count_by_gpu"] = {
        "0": 64,
        "1": 0,
    }

    class _ReceiptOnly:
        def safe_runtime_receipt(self):
            return drifted

    with pytest.raises(
        formal.EbmNlpP1FormalControllerError,
        match="concurrency",
    ):
        formal._executor_runtime_receipt(
            _ReceiptOnly(),
            kind="hipporag",
            formal_scope=True,
            expected_hippo_workers=64,
        )


def test_actual_logistic_probe_state_is_serialized_and_bound() -> None:
    embeddings = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
    )
    first = core.fit_independent_role_probes(
        embeddings,
        {
            role: (0, 0, 1, 1)
            for role in core.ROLE_ORDER
        },
    )
    second = core.fit_independent_role_probes(
        embeddings,
        {
            role: (1, 1, 0, 0)
            for role in core.ROLE_ORDER
        },
    )
    first_payload = formal._probe_state_payload(first)
    second_payload = formal._probe_state_payload(second)
    assert first_payload["models"][0]["coefficient"]
    assert first_payload["models"][0]["intercept"]
    assert first_payload["models"][0]["classes"] == [0, 1]
    assert formal._stable_hash(first_payload) != formal._stable_hash(
        second_payload
    )


def test_valid_nonpromotion_never_prepares_or_opens_m_search(
    tmp_path: Path,
) -> None:
    archive, contract = _source_fixture(tmp_path)
    embedder = _EmbeddingExecutor()
    launcher = _HippoLauncher()
    work = tmp_path / "work"
    terminal = formal._run_study_with_terminal(
        archive_path=archive,
        work_root=work,
        contract=contract,
        embedder=embedder,
        hippo_launcher=launcher,
        execution_binding_sha256="0" * 64,
        execution_scope="source_free_synthetic_contract_test",
    )
    assert terminal["status"] == (
        "complete_valid_nonpromotion_"
        "M_action_model_view_and_gold_unopened"
    )
    assert terminal["A_hold_promotion"] is False
    assert terminal["M_search_action_or_model_view_opened"] is False
    assert terminal["M_search_gold_opened"] is False
    assert (
        terminal["M_search_documents_present_only_in_private_acquisition"]
        is True
    )
    assert launcher.batch_sizes == [1]
    assert not (work / "M_search.actions.private.json").exists()
    assert not (
        work
        / "source_private/label_open_markers/"
        "M_search.attempt_consumed.json"
    ).exists()
    assert (work / "G_form.probes.private.json").is_file()
    assert stat.S_IMODE(
        (work / "G_form.probes.private.json").stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (work / "formal_terminal.json").stat().st_mode
    ) == 0o600
    assert set(terminal["A_hold_secondary"]) == {
        "complete_at_5_family_rates",
        "recipe_selection_counts",
        "typed_selected_outside_RAW_top5",
        "undiscounted_coverage_at_5_family_means",
    }


def test_promotion_opens_m_only_after_m_actions_are_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, contract = _source_fixture(tmp_path)
    launcher = _HippoLauncher()
    observed: list[bool] = []
    original_open = formal._open_stage_labels

    def observing_open(
        acquisition: source.AcquisitionResult,
        *,
        block: str,
        promotion: bool = False,
    ):
        if block == "M_search":
            observed.append(
                (tmp_path / "work/M_search.actions.private.json").is_file()
            )
        return original_open(
            acquisition, block=block, promotion=promotion
        )

    monkeypatch.setattr(formal, "_open_stage_labels", observing_open)
    monkeypatch.setattr(
        formal,
        "_comparison_success",
        lambda _comparison, *, require_all_families: True,
    )
    terminal = formal._run_study_with_terminal(
        archive_path=archive,
        work_root=tmp_path / "work",
        contract=contract,
        embedder=_EmbeddingExecutor(),
        hippo_launcher=launcher,
        execution_binding_sha256="0" * 64,
        execution_scope="source_free_synthetic_contract_test",
    )
    assert observed == [True]
    assert launcher.batch_sizes == [1, 1]
    assert terminal["M_search_action_or_model_view_opened"] is True
    assert terminal["M_search_gold_opened"] is True
    assert terminal["joint_total_goal"] is True
    assert (
        tmp_path
        / "work/source_private/label_open_markers/"
        "M_search.complete.json"
    ).is_file()


def test_runtime_failure_closes_without_raw_payload_or_replay(
    tmp_path: Path,
) -> None:
    archive, contract = _source_fixture(tmp_path)

    def failing_embedder(
        _texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        raise RuntimeError("private text deliberately omitted")

    terminal = formal._run_study_with_terminal(
        archive_path=archive,
        work_root=tmp_path / "work",
        contract=contract,
        embedder=failing_embedder,
        hippo_launcher=_HippoLauncher(),
        execution_binding_sha256="0" * 64,
        execution_scope="source_free_synthetic_contract_test",
    )
    assert (
        terminal["status"]
        == "terminal_implementation_or_runtime_invalid"
    )
    assert terminal["primary_evaluated"] is False
    assert terminal["replay_permitted"] is False
    assert "private text" not in str(terminal)
    assert terminal["raw_PMID_token_text_label_or_action_output_count"] == 0


def test_public_formal_surface_rejects_alternate_source_contract(
    tmp_path: Path,
) -> None:
    archive, contract = _source_fixture(tmp_path)
    with pytest.raises(
        formal.EbmNlpP1FormalControllerError,
        match="exact frozen source contract",
    ):
        formal.run_formal_study(
            archive_path=archive,
            work_root=tmp_path / "work",
            contract=contract,
            embedder=_EmbeddingExecutor(),
            hippo_launcher=_HippoLauncher(),
            execution_binding=_test_binding(),
        )
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "rows",
    [
        ((True, 0.0),),
        (("1.0", 0.0),),
    ],
)
def test_embedding_validator_does_not_launder_types(rows) -> None:
    with pytest.raises(
        formal.EbmNlpP1FormalControllerError,
        match="finite real",
    ):
        formal._validate_embeddings(rows, expected_count=1)


def test_misbound_hippo_role_fails_before_a_hold_gold_open(
    tmp_path: Path,
) -> None:
    archive, contract = _source_fixture(tmp_path)

    class _Misbound(_HippoLauncher):
        def __call__(self, payloads):
            outputs = dict(super().__call__(payloads))
            key = next(iter(outputs))
            changed = json.loads(json.dumps(outputs[key]))
            changed["rows"][0]["work_id"] = "misbound"
            outputs[key] = changed
            return outputs

    work = tmp_path / "work"
    terminal = formal._run_study_with_terminal(
        archive_path=archive,
        work_root=work,
        contract=contract,
        embedder=_EmbeddingExecutor(),
        hippo_launcher=_Misbound(),
        execution_binding_sha256="0" * 64,
        execution_scope="source_free_synthetic_contract_test",
    )
    assert (
        terminal["status"]
        == "terminal_implementation_or_runtime_invalid"
    )
    assert not (
        work
        / "source_private/label_open_markers/"
        "A_hold.attempt_consumed.json"
    ).exists()


def test_post_gold_scoring_consumes_sealed_actions_without_reselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, contract = _source_fixture(tmp_path)
    original_open = formal._open_stage_labels

    def fail_reselection(*_args, **_kwargs):
        raise AssertionError("post-gold selector was called")

    def observing_open(
        acquisition: source.AcquisitionResult,
        *,
        block: str,
        promotion: bool = False,
    ):
        if block == "A_hold":
            monkeypatch.setattr(core, "select_e0", fail_reselection)
            monkeypatch.setattr(core, "select_e1", fail_reselection)
            monkeypatch.setattr(
                core, "raw_probe_ranking", fail_reselection
            )
        return original_open(
            acquisition, block=block, promotion=promotion
        )

    monkeypatch.setattr(formal, "_open_stage_labels", observing_open)
    terminal = formal._run_study_with_terminal(
        archive_path=archive,
        work_root=tmp_path / "work",
        contract=contract,
        embedder=_EmbeddingExecutor(),
        hippo_launcher=_HippoLauncher(),
        execution_binding_sha256="0" * 64,
        execution_scope="source_free_synthetic_contract_test",
    )
    assert terminal["status"].startswith("complete_valid_nonpromotion")
