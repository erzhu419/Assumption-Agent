from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_formal_controller_v1 as controller
from assumption_agent.benchmarks import feverous_p6_e2_formal_runner_v1 as runner
from replication_runtime.feverous_official_hipporag_v1.contract import RetrievalBatch


def _top5_rows(count: int, *, offset: int = 0) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((offset + row + column) % 8192 for column in range(5))
        for row in range(count)
    )


def test_formation_query_schedule_is_interleaved_and_exactly_split() -> None:
    a_claims = tuple(f"A-{index}" for index in range(96))
    f_claims = tuple(f"F-{index}" for index in range(48))
    queries, schedule = controller.formation_query_schedule(a_claims, f_claims)

    assert len(queries) == 144
    assert schedule[:6] == (
        ("A_form", 0),
        ("F_search", 0),
        ("A_form", 1),
        ("F_search", 1),
        ("A_form", 2),
        ("F_search", 2),
    )
    assert schedule[94:98] == (
        ("A_form", 47),
        ("F_search", 47),
        ("A_form", 48),
        ("A_form", 49),
    )
    assert schedule[-1] == ("A_form", 95)
    rows = _top5_rows(144)
    split = controller.split_formation_hippo_result(
        RetrievalBatch(indices=rows, receipt={}), schedule=schedule
    )
    assert split["A_form"] == tuple(
        rows[position]
        for position, coordinate in enumerate(schedule)
        if coordinate[0] == "A_form"
    )
    assert split["F_search"] == tuple(
        rows[position]
        for position, coordinate in enumerate(schedule)
        if coordinate[0] == "F_search"
    )


class _FormationHippo:
    def __init__(self, barrier: threading.Barrier, calls: list[object]) -> None:
        self.barrier = barrier
        self.calls = calls

    def retrieve(self, *, block: str, queries: tuple[str, ...]) -> RetrievalBatch:
        self.calls.append(("hippo", block, queries))
        self.barrier.wait(timeout=5)
        return RetrievalBatch(
            indices=_top5_rows(len(queries)),
            receipt={"receipt_sha256": "9" * 64},
        )


@dataclass
class _FormationRuntime:
    hippo: object
    minilm: object = object()
    ner: object = object()
    nli: object = object()


@dataclass(frozen=True)
class _PreparedSemanticDouble:
    receipt: Mapping[str, Any]


def test_default_core_uses_one_local_pool_and_one_interleaved_hippo_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    barrier = threading.Barrier(2)
    claims = {
        "A_form": tuple(f"A-{index}" for index in range(96)),
        "F_search": tuple(f"F-{index}" for index in range(48)),
    }

    monkeypatch.setattr(
        runner,
        "claims_from_block_view",
        lambda _view, *, block: claims[block],
    )

    def combined(**kwargs: object) -> runner.FormationExecution:
        calls.append(("local", kwargs))
        barrier.wait(timeout=5)
        a = runner.BlockExecution(
            block="A_form",
            items=(),
            feature_receipt={},
            receipt={"block_receipt_sha256": "1" * 64},
        )
        f = runner.BlockExecution(
            block="F_search",
            items=(),
            feature_receipt={},
            receipt={"block_receipt_sha256": "2" * 64},
        )
        return runner.FormationExecution(
            A_form=a,
            F_search=f,
            receipt={"formation_execution_receipt_sha256": "3" * 64},
        )

    monkeypatch.setattr(runner, "execute_formation_blocks", combined)
    runtime = _FormationRuntime(_FormationHippo(barrier, calls))
    prepared = controller.PreparedExecution(
        _PreparedSemanticDouble(
            {"preparation_receipt_sha256": "7" * 64}
        ),  # type: ignore[arg-type]
        {"receipt_sha256": "8" * 64},
    )
    result = controller.DefaultFormalCore().execute_formation(
        A_form_view={},
        F_search_view={},
        prepared=prepared,
        runtime=runtime,
    )

    local_calls = [row for row in calls if row[0] == "local"]
    hippo_calls = [row for row in calls if row[0] == "hippo"]
    assert len(local_calls) == len(hippo_calls) == 1
    local_kwargs = local_calls[0][1]
    assert local_kwargs["worker_count"] == 64
    assert local_kwargs["A_form_claims"] == claims["A_form"]
    assert local_kwargs["F_search_claims"] == claims["F_search"]
    assert hippo_calls[0][1] == "A_form"
    assert len(hippo_calls[0][2]) == 144
    assert len(result.A_form.hippo_top5) == 96
    assert len(result.F_search.hippo_top5) == 48
    assert result.receipt["single_official_gateway_retrieve_call"] is True
    assert result.receipt["both_physical_jobs_submitted_before_join"] is True
    assert result.receipt["semantic_preparation_receipt_sha256"] == "7" * 64
    assert result.receipt["official_hipporag_build_receipt_sha256"] == "8" * 64
    controller._assert_label_free_archive_payload(
        {"execution_receipt": result.receipt}
    )


def test_default_core_rejects_nonboolean_policy_identifiability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = runner.BlockExecution(
        block="A_hold", items=(), feature_receipt={}, receipt={}
    )
    stage = controller.LabelFreeBlockStage(
        block="A_hold",
        local=local,
        hippo_top5=(),
        hippo_receipt={},
        execution_receipt={},
    )
    monkeypatch.setattr(
        runner,
        "score_anchor_block",
        lambda **_kwargs: pytest.fail("runner scorer must not receive a string bool"),
    )
    with pytest.raises(
        controller.FeverousFormalControllerError, match="policy fields"
    ):
        controller.DefaultFormalCore().score_anchor(
            stage=stage,
            labels={},
            policy_receipt={
                "E0_selected_recipe_id": "R0_DENSE5",
                "E2_selected_recipe_id": "R1_P6_DIRECT_B2",
                "A_hold_evaluator_comparison_identifiable": "false",
            },
        )


def test_production_prerequisite_verification_never_calls_full_private_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze = {
        "implementation_freeze_sha256": "a" * 64,
        "identity_compiler_qualification_sha256": "b" * 64,
        "runtime_preflight_sha256": "c" * 64,
        "implementation_git_commit": "d" * 40,
    }
    acquisition = {
        "implementation_freeze_sha256": "a" * 64,
        "identity_full_compile_equivalence_qualification_sha256": "b" * 64,
        "acquisition_receipt_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        controller.implementation_freeze,
        "verify_committed_implementation_freeze",
        lambda _project: freeze,
    )
    envelope_calls: list[str] = []
    monkeypatch.setattr(
        controller.formal_acquisition,
        "verify_acquisition_envelope",
        lambda _project: envelope_calls.append("envelope") or acquisition,
    )
    monkeypatch.setattr(
        controller.formal_acquisition,
        "verify_acquisition_receipt",
        lambda _project: pytest.fail("future M/private packs were fully hashed"),
    )
    binding = controller.ModuleAcquisitionBoundary().verify_prerequisites(
        project=tmp_path
    )
    assert binding.acquisition_receipt_sha256 == "e" * 64
    assert envelope_calls == ["envelope"]


@dataclass(frozen=True)
class _FakeStage:
    block: str


@dataclass(frozen=True)
class _FakeFormation:
    A_form: _FakeStage
    F_search: _FakeStage


class _FakeAcquisition:
    def __init__(
        self,
        *,
        events: list[str],
        project: Path,
        paths: controller.LifecycleOutputPaths,
        preflight_receipt: Mapping[str, Any],
    ) -> None:
        self.events = events
        self.project = project
        self.paths = paths
        self.binding = controller.PrerequisiteBinding(
            implementation_freeze_sha256="a" * 64,
            acquisition_receipt_sha256="b" * 64,
            runtime_preflight_sha256=controller.stable_hash(preflight_receipt),
            implementation_git_commit="c" * 40,
        )

    def verify_prerequisites(
        self, *, project: Path
    ) -> controller.PrerequisiteBinding:
        assert project == self.project
        self.events.append("verify_prerequisites")
        return self.binding

    def assert_stable(
        self,
        *,
        project: Path,
        prerequisites: controller.PrerequisiteBinding,
    ) -> None:
        assert project == self.project
        assert prerequisites == self.binding

    def preflight_outputs(
        self,
        *,
        project: Path,
        runtime_config: object,
        output_paths: controller.LifecycleOutputPaths,
    ) -> None:
        self.events.append("preflight_outputs")
        assert not (project / output_paths.root_relative).exists()

    def load_corpus_view(self, *, project: Path) -> Mapping[str, Any]:
        self.events.append("load_corpus")
        return {"synthetic": "content is held only by the fake core"}

    def load_block_view(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        if expected_block == "M_search":
            promotion = project / self.paths.receipt_relative("A_hold_promotion")
            assert promotion.is_file(), "M view opened before promotion artifact"
        self.events.append(f"load_view:{expected_block}")
        return {"synthetic_block": expected_block}

    def load_block_labels(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        assert (project / self.paths.archive_relative(expected_block)).is_file()
        assert (project / self.paths.seal_relative(expected_block)).is_file()
        if expected_block == "A_form":
            assert (project / self.paths.archive_relative("F_search")).is_file()
            assert (project / self.paths.seal_relative("F_search")).is_file()
        self.events.append(f"load_labels:{expected_block}")
        return {"synthetic_labels_for": expected_block}

    def load_private_secret(self, *, project: Path) -> bytes:
        assert "load_labels:A_form" in self.events
        self.events.append("load_fold_secret")
        return b"synthetic-fold-secret"


class _FakeRuntimeContext(AbstractContextManager[object]):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> object:
        self.events.append("runtime_enter")
        return object()

    def __exit__(self, *exc: object) -> None:
        self.events.append("runtime_exit")


class _FakeRuntime:
    def __init__(
        self, events: list[str], preflight_receipt: Mapping[str, Any]
    ) -> None:
        self.events = events
        self.preflight_receipt = dict(preflight_receipt)

    def preflight(self, runtime_config: object) -> Mapping[str, Any]:
        self.events.append("runtime_preflight")
        return self.preflight_receipt

    def open(self, runtime_config: object) -> AbstractContextManager[object]:
        self.events.append("runtime_open")
        return _FakeRuntimeContext(self.events)

    def postflight(self, runtime: object) -> Mapping[str, Any]:
        self.events.append("runtime_postflight")
        return {
            "schema": "synthetic_runtime_bundle",
            "worker_counts": {"local": 64, "hippo": 8, "nli": 8},
            "network_calls": 0,
        }


class _FakeCore:
    def __init__(
        self,
        events: list[str],
        *,
        promoted: bool,
        identifiable: bool,
        fail_anchor: str | None = None,
    ) -> None:
        self.events = events
        self.promoted = promoted
        self.identifiable = identifiable
        self.fail_anchor = fail_anchor

    def prepare(
        self, *, corpus_view: Mapping[str, Any], runtime: object
    ) -> object:
        self.events.append("prepare")
        return object()

    def execute_formation(
        self,
        *,
        A_form_view: Mapping[str, Any],
        F_search_view: Mapping[str, Any],
        prepared: object,
        runtime: object,
    ) -> object:
        self.events.append("execute_formation:one_pool_144_and_one_hippo_call")
        return _FakeFormation(_FakeStage("A_form"), _FakeStage("F_search"))

    def formation_blocks(self, formation: object) -> Mapping[str, object]:
        assert isinstance(formation, _FakeFormation)
        return {"A_form": formation.A_form, "F_search": formation.F_search}

    def execute_anchor(
        self,
        *,
        block: str,
        view: Mapping[str, Any],
        prepared: object,
        runtime: object,
    ) -> object:
        self.events.append(f"execute_anchor:{block}")
        if self.fail_anchor == block:
            raise RuntimeError("private claim text must never reach a failure file")
        return _FakeStage(block)

    def archive_payload(self, stage: object, *, block: str) -> Mapping[str, Any]:
        assert stage == _FakeStage(block)
        self.events.append(f"archive_payload:{block}")
        return {
            "block": block,
            "item_commitment_sha256s": [controller.stable_hash([block, 0])],
            "complete_action_trace_receipts": [
                {
                    "recipe_id": "R0_DENSE5",
                    "output_top5": [0, 1, 2, 3, 4],
                    "trace_sha256": "1" * 64,
                }
            ],
            "feature_receipt": {"feature_receipt_sha256": "2" * 64},
            "raw_claim_corpus_gold_label_family_or_verdict_persisted": False,
        }

    def fit_e2(
        self,
        *,
        A_form_stage: object,
        labels: Mapping[str, Any],
        fold_secret: bytes,
    ) -> Mapping[str, Any]:
        assert labels == {"synthetic_labels_for": "A_form"}
        self.events.append("fit_E2")
        return {"fit_receipt_sha256": "3" * 64}

    def freeze_f_policies(
        self,
        *,
        F_search_stage: object,
        A_form_stage: object,
        fit_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.events.append("freeze_F_policies")
        return {
            "policy_receipt_sha256": "4" * 64,
            "E0_selected_recipe_id": "R0_DENSE5",
            "E2_selected_recipe_id": (
                "R1_P6_DIRECT_B2" if self.identifiable else "R0_DENSE5"
            ),
            "A_hold_evaluator_comparison_identifiable": self.identifiable,
        }

    def score_anchor(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        policy_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        assert isinstance(stage, _FakeStage)
        self.events.append(f"score:{stage.block}")
        body = {
            "block": stage.block,
            "A_hold_real_domain_primary_passed": (
                True if stage.block == "A_hold" else None
            ),
            "evaluator_promoted": (
                self.promoted if stage.block == "A_hold" else None
            ),
            "M_L5_passed": True if stage.block == "M_search" else None,
        }
        return {**body, "score_receipt_sha256": controller.stable_hash(body)}


def _run_synthetic(
    tmp_path: Path,
    *,
    promoted: bool,
    identifiable: bool,
    fail_anchor: str | None = None,
) -> tuple[dict[str, Any] | None, list[str], controller.LifecycleOutputPaths]:
    events: list[str] = []
    preflight = {"synthetic_runtime": "offline", "model_calls": 0}
    paths = controller.LifecycleOutputPaths(Path("synthetic_controller"))
    acquisition = _FakeAcquisition(
        events=events,
        project=tmp_path,
        paths=paths,
        preflight_receipt=preflight,
    )
    runtime = _FakeRuntime(events, preflight)
    core = _FakeCore(
        events,
        promoted=promoted,
        identifiable=identifiable,
        fail_anchor=fail_anchor,
    )
    result = controller._run_lifecycle_core(
        project=tmp_path,
        runtime_config=object(),
        acquisition_boundary=acquisition,
        runtime_boundary=runtime,
        core=core,
        output_paths=paths,
    )
    return result, events, paths


def test_nonidentifiable_still_runs_a_hold_primary_and_never_reads_m(
    tmp_path: Path,
) -> None:
    result, events, paths = _run_synthetic(
        tmp_path, promoted=False, identifiable=False
    )
    assert result is not None
    assert result["status"] == "valid_A_hold_nonpromotion_M_search_unopened"
    assert "execute_anchor:A_hold" in events
    assert "score:A_hold" in events
    assert not any("M_search" in event for event in events)
    assert result["M_search_view_opened"] is False
    assert result["M_search_labels_opened"] is False
    assert result["M_search_executed"] is False

    assert events.index("runtime_preflight") < events.index("runtime_enter")
    assert events.index("execute_formation:one_pool_144_and_one_hippo_call") < events.index(
        "load_labels:A_form"
    )
    assert events.index("load_labels:A_form") < events.index("fit_E2")
    assert events.index("freeze_F_policies") < events.index("load_view:A_hold")
    assert events.index("archive_payload:A_hold") < events.index(
        "load_labels:A_hold"
    )
    assert events.index("runtime_postflight") < events.index("runtime_exit")
    assert (tmp_path / paths.result_relative).is_file()
    assert (tmp_path / paths.receipt_relative("runtime_postflight")).is_file()
    postflight_artifact = json.loads(
        (tmp_path / paths.receipt_relative("runtime_postflight")).read_text(
            encoding="ascii"
        )
    )
    assert postflight_artifact["receipt"]["terminal_item_counts"] == {
        "A_form": 96,
        "F_search": 48,
        "A_hold": 72,
    }
    assert not (tmp_path / paths.receipt_relative("A_hold_promotion")).exists()


def test_promotion_is_the_only_path_that_opens_and_scores_m(tmp_path: Path) -> None:
    result, events, paths = _run_synthetic(
        tmp_path, promoted=True, identifiable=True
    )
    assert result is not None
    assert result["status"] == "formal_M_search_complete"
    assert events.index("score:A_hold") < events.index("load_view:M_search")
    assert events.index("load_view:M_search") < events.index(
        "execute_anchor:M_search"
    )
    assert events.index("archive_payload:M_search") < events.index(
        "load_labels:M_search"
    )
    assert events.index("load_labels:M_search") < events.index("score:M_search")
    assert result["M_search_view_opened"] is True
    assert result["M_search_labels_opened"] is True
    assert (tmp_path / paths.receipt_relative("A_hold_promotion")).is_file()


def test_archive_tamper_and_raw_private_field_fail_closed(tmp_path: Path) -> None:
    paths = controller.LifecycleOutputPaths(Path("archive_test"))
    output_root = tmp_path / paths.root_relative
    output_root.mkdir()
    with pytest.raises(controller.FeverousFormalControllerError):
        controller.persist_label_free_archive(
            project=tmp_path,
            output_paths=paths,
            block="A_form",
            marker_sha256="a" * 64,
            stage_payload={"claim": "forbidden raw claim"},
        )

    archive = controller.persist_label_free_archive(
        project=tmp_path,
        output_paths=paths,
        block="A_form",
        marker_sha256="a" * 64,
        stage_payload={
            "complete_action_trace_receipts": [],
            "feature_receipt": {"feature_receipt_sha256": "b" * 64},
        },
    )
    payload = json.loads(archive.path.read_text(encoding="ascii"))
    payload["stage"]["feature_receipt"]["feature_receipt_sha256"] = "c" * 64
    archive.path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(
        controller.FeverousFormalControllerError, match="self-hash|changed"
    ):
        controller.seal_label_free_archive(
            project=tmp_path, output_paths=paths, archive=archive
        )


def test_failure_file_contains_only_hashes_not_exception_plaintext(
    tmp_path: Path,
) -> None:
    paths = controller.LifecycleOutputPaths(Path("synthetic_controller"))
    with pytest.raises(RuntimeError, match="private claim text"):
        _run_synthetic(
            tmp_path,
            promoted=False,
            identifiable=True,
            fail_anchor="A_hold",
        )
    failure_path = tmp_path / paths.failure_relative
    raw = failure_path.read_text(encoding="ascii")
    failure = json.loads(raw)
    assert "private claim text" not in raw
    assert "RuntimeError" not in raw
    assert set(
        key
        for key in failure
        if key.startswith("exception_")
    ) == {
        "exception_type_sha256",
        "exception_message_sha256",
        "exception_type_or_message_plaintext_persisted",
    }
    assert failure["exception_type_or_message_plaintext_persisted"] is False
    assert failure["retry_replay_resample_or_replacement_authorized"] is False


def test_controller_is_bound_only_to_successor_v2_acquisition_and_roots() -> None:
    assert controller.formal_acquisition.VERSION == "feverous_p6_e2_formal_acquisition_v2"
    assert controller.formal_acquisition.FORMAL_ROOT_RELATIVE == Path(
        "artifacts/feverous_p6_e2_formal_v2"
    )
    assert controller.FORMAL_OUTPUT_ROOT_RELATIVE == (
        controller.formal_acquisition.FORMAL_ROOT_RELATIVE / "controller"
    )
    assert (
        controller.local_runtime.FORMAL_ROOT_RELATIVE
        == controller.formal_acquisition.FORMAL_ROOT_RELATIVE
    )
