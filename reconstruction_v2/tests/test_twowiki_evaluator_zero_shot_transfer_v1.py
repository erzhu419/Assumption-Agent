from __future__ import annotations

import inspect
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from assumption_agent.models import stable_hash
from assumption_agent.benchmarks import twowiki_evaluator_zero_shot_transfer_v1 as subject
from assumption_agent.benchmarks import hotpot_evaluator_portfolio_coevolution_v1 as frozen_core


PROJECT = Path(__file__).resolve().parents[1]


def _row(*, block: str = "A_hold", ordinal: int = 0) -> dict:
    member = "train.json" if block == "A_hold" else "dev.json"
    return {
        "schema": "twowiki_evaluator_zero_shot_transfer_acquisition_v1_private_row",
        "block": block,
        "source_member": member,
        "question_type": "comparison",
        "item_id": f"item-{ordinal}",
        "question": f"Question {ordinal}?",
        "corpus": [
            {
                "paragraph_idx": index,
                "paragraph_title": f"Title {index}",
                "paragraph_text": f"Paragraph {index} text.",
            }
            for index in range(10)
        ],
        "answers": ["answer"],
        "normalized_answers": ["answer"],
        "support_indices": [1, 7],
        "source_row_sha256": "1" * 64,
        "normalized_question_sha256": "2" * 64,
        "canonical_question_plus_ordered_context_sha256": "3" * 64,
        "canonical_row_sha256": "4" * 64,
    }


def _items(count: int, *, block: str = "A_hold") -> tuple[subject.StudyItem, ...]:
    return tuple(subject._row_to_item(_row(block=block, ordinal=index), expected_block=block) for index in range(count))


class _Runtime:
    def __init__(self) -> None:
        self.safe_binding = {"binding_sha256": "a" * 64}
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0
        self.calls = 0

    def retrieve(self, *, question, paragraphs, work_root):
        assert question and len(paragraphs) == 10
        with self.lock:
            self.active += 1
            self.calls += 1
            self.maximum = max(self.maximum, self.active)
        time.sleep(0.002)
        with self.lock:
            self.active -= 1
        return [0, 1, 2, 3, 4]

    def fresh_reverify(self):
        return self.safe_binding


def test_exact_design_and_public_actions_materialize_without_private_cache(monkeypatch):
    design, binding = subject._load_design(PROJECT)
    actions = subject._load_fixed_actions(PROJECT)
    assert design["design_sha256"] == subject.DESIGN_SEMANTIC_SHA256
    assert binding["file_sha256"] == subject.DESIGN_FILE_SHA256
    assert actions.retained_p.program_hash == design["fixed_actions"]["retained_P"]["program_sha256"]
    assert [row.program_hash for row in actions.a_incumbent] == design["fixed_actions"]["A_incumbent"]["Q_program_sha256s"]
    assert [row.program_hash for row in actions.a_challenger] == design["fixed_actions"]["A_challenger"]["Q_program_sha256s"]
    assert [row.program_hash for row in actions.f_incumbent] == design["fixed_actions"]["F_incumbent"]["Q_program_sha256s"]
    assert [row.program_hash for row in actions.f_challenger] == design["fixed_actions"]["F_challenger_if_promoted_active"]["Q_program_sha256s"]
    assert actions.public_binding["private_MuSiQue_formation_cache_opened"] is False
    assert not any("private_cache" in name for name in inspect.signature(subject._load_fixed_actions).parameters)


def test_twowiki_mapper_preserves_source_context_order_and_variable_supports():
    row = _row()
    row["support_indices"] = [0, 4, 9]
    item = subject._row_to_item(row, expected_block="A_hold")
    assert [paragraph.idx for paragraph in item.view.corpus] == list(range(10))
    assert [paragraph.title for paragraph in item.view.corpus] == [f"Title {index}" for index in range(10)]
    assert item.support_indices == (0, 4, 9)
    assert item.view.hipporag_paragraphs()[4] == {
        "idx": 4,
        "title": "Title 4",
        "paragraph_text": "Paragraph 4 text.",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row["corpus"].append(row["corpus"][0]),
        lambda row: row["corpus"][1].__setitem__("paragraph_idx", 4),
        lambda row: row["corpus"][1].__setitem__("paragraph_title", "Title 0"),
        lambda row: row.__setitem__("support_indices", [1, 1]),
        lambda row: row.__setitem__("support_indices", [10]),
        lambda row: row.__setitem__("source_member", "dev.json"),
    ],
)
def test_twowiki_mapper_fails_closed_on_schema_drift(mutate):
    row = _row()
    mutate(row)
    with pytest.raises(subject.TwoWikiZeroShotTransferError):
        subject._row_to_item(row, expected_block="A_hold")


def test_exact_sign_flip_requires_both_positive_net_and_alpha():
    assert subject.exact_paired_sign_flip([1, 1, 1, 1])["promoted"] is True
    zero = subject.exact_paired_sign_flip([1, -1])
    assert zero["exact_p_at_or_below_alpha"] is False
    assert zero["promoted"] is False
    negative = subject.exact_paired_sign_flip([-1] * 8)
    assert negative["positive_observed_net"] is False
    assert negative["promoted"] is False


def test_secondary_pair_is_explicitly_descriptive_and_non_gating():
    items = _items(2)
    arms = {
        "left": [(0, 1, 2, 3, 4), (0, 1, 2, 3, 4)],
        "right": [(0, 2, 3, 4, 5), (0, 2, 3, 4, 5)],
    }
    comparison = subject._descriptive_paired("left", "right", items, arms)
    subject._validate_descriptive_paired(comparison, left="left", right="right")
    assert comparison["descriptive_only"] is True
    assert comparison["affects_L5_or_epoch"] is False
    assert "promoted" not in comparison["paired_test"]


@pytest.mark.parametrize(
    "deltas",
    [
        [1, 1, 1, 1], [1, -1, 2, -2, 0], [-1] * 8,
        [3, 2, 1, -1, 0, 0], [0] * 48,
    ],
)
def test_counter_dp_sign_flip_matches_frozen_exact_distribution(deltas):
    actual = subject.exact_paired_sign_flip(deltas)
    expected = frozen_core.exact_paired_sign_flip(deltas)
    assert actual["observed_net_support_hits"] == expected["observed_net_support_hits"]
    assert actual["nonzero_pair_count"] == expected["nonzero_pair_count"]
    assert actual["p_value_numerator"] == expected["p_value_numerator"]
    assert actual["p_value_denominator"] == expected["p_value_denominator"]


def test_rrf_is_exactly_equivalent_to_frozen_fractional_core():
    rankings = (
        (9, 1, 4, 3, 0),
        (4, 5, 9, 2, 7),
        (2, 9, 6, 4, 8),
    )
    assert subject._fuse(*rankings) == frozen_core.fuse_rankings(*rankings)


def test_duplicated_retained_p_calls_are_execution_gating():
    items = _items(1)
    ranking = (0, 1, 2, 3, 4)
    direct = {(0, component): ranking for component in subject.A_COMPONENT_IDS}
    direct[(0, "challenger_P")] = (1, 0, 2, 3, 4)
    with pytest.raises(subject.TwoWikiZeroShotTransferError, match="retained-P"):
        subject._anchor_arms(items, direct)

    items = _items(1, block="M_search")
    direct = {(0, component): ranking for component in subject.M_COMPONENT_IDS}
    direct[(0, "incumbent_P")] = (1, 0, 2, 3, 4)
    with pytest.raises(subject.TwoWikiZeroShotTransferError, match="retained-P"):
        subject._search_arms(items, direct)


def test_submit_helper_submits_entire_barrier_cohort_before_first_join():
    state = {"submitted": 0, "first_result_at": None}

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            if state["first_result_at"] is None:
                state["first_result_at"] = state["submitted"]
            return self.value

    class Executor:
        def submit(self, function, value):
            state["submitted"] += 1
            return Future(function(value))

    result = subject._submit_eager_then_join(
        executor=Executor(), function=lambda value: value * 2, work_units=tuple(range(192))
    )
    assert state == {"submitted": 192, "first_result_at": 192}
    assert result[-1] == 382


def test_anchor_executes_two_complete_192_party_waves(monkeypatch, tmp_path):
    items = _items(48)
    runtime = _Runtime()
    monkeypatch.setattr(subject, "_ranking", lambda program, item: (0, 1, 2, 3, 4))
    dummy = SimpleNamespace()
    programs = {
        component: dummy
        for component in subject.A_COMPONENT_IDS
        if component not in {"canonical_RAW", "official_HippoRAG_core_item_local"}
    }
    direct, execution = subject._execute_components(
        root=tmp_path,
        items=items,
        component_ids=subject.A_COMPONENT_IDS,
        programs=programs,
        prepared=runtime,
        wave_count=2,
    )
    assert len(direct) == 384
    assert execution["retrieval_attempt_count"] == 384
    assert execution["retrieval_terminal_count"] == 384
    assert execution["official_attempt_count"] == 48
    assert execution["official_terminal_count"] == 48
    assert execution["observed_barrier_party_counts"] == [192, 192]
    assert runtime.calls == 48
    assert runtime.maximum <= 24


def test_m_search_executes_one_complete_192_party_wave(monkeypatch, tmp_path):
    items = _items(24, block="M_search")
    runtime = _Runtime()
    monkeypatch.setattr(subject, "_ranking", lambda program, item: (0, 1, 2, 3, 4))
    dummy = SimpleNamespace()
    programs = {
        component: dummy
        for component in subject.M_COMPONENT_IDS
        if component not in {"canonical_RAW", "official_HippoRAG_core_item_local"}
    }
    direct, execution = subject._execute_components(
        root=tmp_path,
        items=items,
        component_ids=subject.M_COMPONENT_IDS,
        programs=programs,
        prepared=runtime,
        wave_count=1,
    )
    assert len(direct) == 192
    assert execution["retrieval_attempt_count"] == 192
    assert execution["official_terminal_count"] == 24
    assert execution["observed_barrier_party_counts"] == [192]


def test_fresh_probe_failure_precedes_root_and_authorization(monkeypatch, tmp_path):
    root = tmp_path / "formal" / "a_hold"
    root.parent.mkdir()
    freeze = {
        "execution_root_sha256": stable_hash({"absolute_execution_root": str(root.absolute())}),
        "design_binding": {}, "source_binding": {}, "fixed_action_binding": {},
    }
    monkeypatch.setattr(subject, "_CLEAN_MODULE_CLI_ACTIVE", True)
    monkeypatch.setattr(
        subject,
        "_canonical_execution_root",
        lambda **_kwargs: root,
    )
    monkeypatch.setattr(subject, "_load_freeze", lambda *args, **kwargs: (freeze, "f" * 64))
    monkeypatch.setattr(subject, "_load_design", lambda project: ({}, {}))
    monkeypatch.setattr(subject, "_load_fixed_actions", lambda project: SimpleNamespace(public_binding={}))
    commitment = SimpleNamespace(
        block="A_hold", source_member="train.json", count=48,
        question_type_counts={"comparison": 48}, file_sha256="1" * 64,
        item_commitment_set_sha256="2" * 64,
    )
    receipt = {"acquisition_sha256": "3" * 64, "private_pack_sha256": "4" * 64}
    monkeypatch.setattr(
        subject, "_load_acquisition",
        lambda **kwargs: (receipt, b"receipt", {"A_hold": commitment}),
    )
    freeze["source_binding"] = subject._source_binding(receipt, b"receipt", commitment)
    prepared = _Runtime()
    monkeypatch.setattr(
        subject, "_verify_runtime_inputs", lambda **kwargs: ({}, prepared)
    )
    called = []

    def fail_probe(_capability):
        called.append("probe")
        raise subject.TwoWikiZeroShotTransferError("probe failed")

    monkeypatch.setattr(subject, "_fresh_probe", fail_probe)
    monkeypatch.setattr(subject, "_prepare_output", lambda path: called.append("prepare"))
    with pytest.raises(subject.TwoWikiZeroShotTransferError, match="probe failed"):
        subject.execute_a_hold_formal(
            project_root=PROJECT, pre_run_freeze_path=tmp_path / "freeze.json",
            acquisition_receipt_path=tmp_path / "receipt.json",
            selection_secret_path=tmp_path / "secret", capability_receipt_path=tmp_path / "cap",
            runtime_python=tmp_path / "python", local_llm_model=tmp_path,
            local_embedding_model=tmp_path, base_binding_receipt_path=tmp_path / "base",
            attestation_receipt_path=tmp_path / "att", execution_root=root,
        )
    assert called == ["probe"]
    assert not root.exists()


def test_no_anchor_promotion_keeps_m_search_unopened(monkeypatch, tmp_path):
    anchor = {"evaluator_epoch_transition": {"promoted": False}}
    monkeypatch.setattr(
        subject, "reverify_a_hold_public_report", lambda **kwargs: (anchor, {})
    )
    monkeypatch.setattr(
        subject, "_build_freeze_common",
        lambda **kwargs: pytest.fail("M_search bindings must not be opened after no promotion"),
    )
    with pytest.raises(subject.TwoWikiZeroShotTransferError, match="must remain unopened"):
        subject.build_m_search_pre_run_freeze(
            project_root=PROJECT,
            acquisition_receipt_path=tmp_path / "receipt",
            selection_secret_path=tmp_path / "secret",
            a_hold_pre_run_freeze_path=tmp_path / "anchor-freeze",
            a_hold_report_path=tmp_path / "anchor-report",
            capability_receipt_path=tmp_path / "capability",
            runtime_python=tmp_path / "python", local_llm_model=tmp_path,
            local_embedding_model=tmp_path, base_binding_receipt_path=tmp_path / "base",
            attestation_receipt_path=tmp_path / "att", execution_root=tmp_path / "search",
            authorization_hash="a" * 64, output_path=tmp_path / "freeze",
        )


def test_public_safety_rejects_private_content_and_absolute_locator():
    with pytest.raises(subject.TwoWikiZeroShotTransferError):
        subject._assert_public_safe({"question": "secret"})
    with pytest.raises(subject.TwoWikiZeroShotTransferError):
        subject._assert_public_safe({"private_path": "/tmp/secret"})
    subject._assert_public_safe({"question_type_counts": {"comparison": 12}})


def test_formal_calls_have_clean_cli_and_no_injection_surface():
    assert subject.formal_signatures_have_no_injection_surface()
    parser_source = inspect.getsource(subject.main)
    assert '"freeze-a-hold"' in parser_source
    assert '"run-a-hold"' in parser_source
    assert '"freeze-m-search"' in parser_source
    assert '"run-m-search"' in parser_source
    assert "--a-hold-block" not in parser_source
    assert "--m-search-block" not in parser_source
    assert "private-cache" not in parser_source


def test_formal_entrypoints_reject_direct_python_calls(tmp_path):
    common = dict(
        project_root=PROJECT, pre_run_freeze_path=tmp_path / "freeze",
        acquisition_receipt_path=tmp_path / "receipt",
        selection_secret_path=tmp_path / "secret", capability_receipt_path=tmp_path / "cap",
        runtime_python=tmp_path / "python", local_llm_model=tmp_path,
        local_embedding_model=tmp_path, base_binding_receipt_path=tmp_path / "base",
        attestation_receipt_path=tmp_path / "att", execution_root=tmp_path / "root",
    )
    with pytest.raises(subject.TwoWikiZeroShotTransferError, match="clean CLI"):
        subject.execute_a_hold_formal(**common)
    with pytest.raises(subject.TwoWikiZeroShotTransferError, match="clean CLI"):
        subject.execute_m_search_formal(
            **common,
            a_hold_pre_run_freeze_path=tmp_path / "anchor-freeze",
            a_hold_report_path=tmp_path / "anchor-report",
        )


def test_atomic_writer_is_exclusive(tmp_path):
    destination = tmp_path / "nested" / "receipt.json"
    subject._write_json_exclusive(destination, {"safe": True})
    with pytest.raises(FileExistsError):
        subject._write_json_exclusive(destination, {"safe": False})
    assert json.loads(destination.read_text()) == {"safe": True}


def test_private_execution_evidence_is_recomputed_and_public_report_must_be_exact_copy(
    tmp_path, monkeypatch,
):
    project = tmp_path / "project"
    root = project / subject.A_EXECUTION_ROOT_RELATIVE
    root.mkdir(parents=True)
    items = _items(48)
    action_binding = {"fixed": "binding"}
    dummy = SimpleNamespace()
    actions = SimpleNamespace(
        public_binding=action_binding,
        retained_p=dummy,
        a_incumbent=(dummy, dummy),
        a_challenger=(dummy, dummy),
    )
    monkeypatch.setattr(subject, "_load_private_items", lambda *args, **kwargs: items)
    monkeypatch.setattr(subject, "_load_fixed_actions", lambda _project: actions)
    monkeypatch.setattr(subject, "_ranking", lambda _program, _item: (0, 1, 2, 3, 4))
    direct = {
        (ordinal, component): (0, 1, 2, 3, 4)
        for ordinal in range(48)
        for component in subject.A_COMPONENT_IDS
    }
    arms = subject._anchor_arms(items, direct)
    metrics = {
        name: subject._aggregate(name, items, rankings)
        for name, rankings in arms.items()
    }
    primary = subject._paired(
        "challenger_portfolio", "incumbent_portfolio", items, arms
    )
    freeze = {
        "freeze_sha256": "1" * 64,
        "authorization_hash": "2" * 64,
        "execution_root_sha256": stable_hash(
            {"absolute_execution_root": str(root.absolute())}
        ),
        "fixed_action_binding": action_binding,
    }
    freeze_file_sha = "3" * 64
    consumption_body = {
        "schema": subject.CONSUMPTION_SCHEMA,
        "stage": "A_hold",
        "authorization_hash": freeze["authorization_hash"],
        "freeze_sha256": freeze["freeze_sha256"],
        "freeze_file_sha256": freeze_file_sha,
        "execution_root_sha256": freeze["execution_root_sha256"],
        "fresh_bwrap_probe_completed_before_marker": True,
        "replay_authorized": False,
        "raw_content_persisted": False,
    }
    subject._write_json_exclusive(
        root / "a_hold.authorization.consumed.json",
        {
            **consumption_body,
            "consumption_sha256": stable_hash(consumption_body),
        },
    )
    evidence_body = {
        "schema": f"{subject.VERSION}_A_hold_private_evidence",
        "freeze_sha256": freeze["freeze_sha256"],
        "item_rows": [
            {
                "item_commitment_sha256": item.view.item_commitment_sha256,
                "support_indices": list(item.support_indices),
                "component_rankings": {
                    component: list(direct[(ordinal, component)])
                    for component in subject.A_COMPONENT_IDS
                },
            }
            for ordinal, item in enumerate(items)
        ],
        "raw_question_or_corpus_persisted": False,
    }
    evidence = {
        **evidence_body,
        "evidence_sha256": stable_hash(evidence_body),
    }
    evidence_path = root / "a_hold.private.evidence.json"
    subject._write_json_exclusive(evidence_path, evidence)
    ranking_receipts = [
        {
            "ordinal_sha256": stable_hash({"ordinal": ordinal}),
            "component_id": component,
            "ranking_sha256": stable_hash(
                {"retrieved_indices": list(ranking)}
            ),
        }
        for (ordinal, component), ranking in sorted(direct.items())
    ]
    report = {
        "source_binding": {
            "measurement_block_file_sha256": "4" * 64,
            "measurement_item_commitment_set_sha256": stable_hash(
                [item.view.item_commitment_sha256 for item in items]
            ),
            "question_type_counts": {
                question_type: 12 for question_type in (
                    "bridge_comparison",
                    "comparison",
                    "compositional",
                    "inference",
                )
            },
        },
        "private_evidence_binding": {
            "file_sha256": subject._sha256_file(evidence_path),
            "evidence_sha256": evidence["evidence_sha256"],
            "private_path_persisted_publicly": False,
            "item_level_evidence_persisted_publicly": False,
        },
        "execution": {
            "ranking_receipt_set_sha256": stable_hash(ranking_receipts)
        },
        "arm_metrics": metrics,
        "challenger_minus_incumbent": primary,
    }
    private_report_path = root / "a_hold.aggregate.report.json"
    subject._write_json_exclusive(private_report_path, report)
    private_raw = private_report_path.read_bytes()
    recomputed = subject._recompute_private_execution_evidence(
        project=project,
        stage="A_hold",
        freeze=freeze,
        freeze_file_sha=freeze_file_sha,
        report=report,
        public_report_raw=private_raw,
    )
    assert recomputed == {"arm_metrics": metrics, "primary": primary}
    forged = dict(report)
    forged["challenger_minus_incumbent"] = {
        **primary,
        "net_support_hit_count": primary["net_support_hit_count"] + 1,
    }
    with pytest.raises(subject.TwoWikiZeroShotTransferError, match="byte-identical"):
        subject._recompute_private_execution_evidence(
            project=project,
            stage="A_hold",
            freeze=freeze,
            freeze_file_sha=freeze_file_sha,
            report=forged,
            public_report_raw=json.dumps(forged, sort_keys=True).encode(),
        )
