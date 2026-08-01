"""Source-free integration tests for SCAR CSSM action formation."""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

from assumption_agent.benchmarks import gscl_scar_cssm_action_v1 as action
from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice,
    document_envelope,
)
from replication_runtime.gscl_narrative_extractor_v2.contract import (
    ClosedChoiceV2Error,
)


_RUNTIME = hashlib.sha256(b"scar-action-fake-runtime").hexdigest()
_ENCODER = hashlib.sha256(b"scar-action-fake-encoder").hexdigest()


class _Backend:
    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME

    def score_batch(self, pairs: tuple[PromptAnswer, ...]):
        output = []
        for pair in pairs:
            selected = int(pair.candidate_key.endswith(".plan.one_relation"))
            count = max(1, len(pair.answer.split()))
            output.append(
                closed_choice.TeacherForcedScore(
                    total_logprob_microunits=selected * count * 1_000_000,
                    answer_token_count=count,
                    context_and_answer_token_count=count + 64,
                )
            )
        return tuple(output)

    def count_program_owned_completion_tokens(self, completion: str) -> int:
        return max(1, len(completion.encode()) // 4)


def _parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource(
            "scar.action." + hashlib.sha256(story.encode()).hexdigest()[:20],
            story,
        ),
        completion,
    )


class _Selector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, story: str):
        self.calls += 1
        return document_envelope.select_document_qualification_only(
            story,
            leaf_selector=self,
        )

    def select_story(self, story: str):
        return closed_choice.select_hierarchical_qualification_only(
            story, backend=_Backend(), narrative_parser=_parser
        )


class _ExactTextEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        unique = {text: index for index, text in enumerate(sorted(set(texts)))}
        matrix = np.zeros((len(texts), max(8, len(unique))), dtype=np.float32)
        for row, text in enumerate(texts):
            matrix[row, unique[text]] = 1.0
        return matrix


def _story() -> str:
    return " ".join(f"SharedToken{index:03d}" for index in range(44)) + "."


def _endpoint_surfaces(story: str) -> tuple[str, str]:
    envelope = _Selector()(story)
    relation = envelope.relations[0]
    mentions = {row.mention_id: row for row in envelope.mentions}
    return tuple(mentions[row].quote for row in relation.slot_mention_ids)


def _slot_id(character: str) -> str:
    return "scar-slot-v1-" + character * 64


def _item() -> dict[str, object]:
    story = _story()
    first, second = _endpoint_surfaces(story)
    left = {
        "background": story,
        "slots": [
            {"opaque_slot_id": _slot_id("a"), "surface": first},
            {"opaque_slot_id": _slot_id("b"), "surface": second},
        ],
        "system": "left system",
    }
    right = {
        "background": story,
        "slots": [
            {"opaque_slot_id": _slot_id("c"), "surface": first},
            {"opaque_slot_id": _slot_id("d"), "surface": second},
        ],
        "system": "right system",
    }
    return {
        "item_token": "scar-item-v1-" + "e" * 64,
        "variants": {
            "base": {"left": left, "right": right},
            "system_swap": {"left": right, "right": left},
        },
    }


def test_two_local_documents_are_reused_for_base_and_swap() -> None:
    selector = _Selector()
    result = action.form_scar_cssm_item_action_v1(
        _item(),
        document_selector=selector,
        encoder=_ExactTextEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    assert selector.calls == 2
    assert result["execution"] == {
        "document_call_count": 2,
        "error_code": None,
        "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
    }
    assert tuple(result["variants"]) == action.VARIANT_NAMES
    for variant in action.VARIANT_NAMES:
        arms = result["variants"][variant]["arms"]
        assert tuple(arms) == action.ARM_IDS
        assert arms["semantic_only"]["disposition"] == "ANSWER"
        assert arms["flat_structural"]["disposition"] == "ANSWER"
        assert arms["full_no_composition"]["disposition"] == "ANSWER"
        assert (
            arms["full_with_length2_composition"]["disposition"]
            == "ABSTAIN"
        )
        assert result["proposal_pools"][variant]["semantic_kbest"]
        assert result["proposal_pools"][variant]["structure_kbest"]
        diagnostic = result["diagnostics"][variant]
        assert diagnostic["structural_diagnostics_available"] is True
        assert type(diagnostic["target_color_shuffle_effective"]) is bool
        assert set(diagnostic["mapping_receipt_sha256_by_arm"]) == set(
            action.ARM_IDS
        )
        assert all(
            len(value) == 64
            for value in diagnostic["mapping_receipt_sha256_by_arm"].values()
        )
        assert diagnostic["left_binder"]["unbound_count"] == 0
        assert diagnostic["right_binder"]["unbound_count"] == 0
    base = result["variants"]["base"]["arms"]["semantic_only"]["pairs"]
    swapped = result["variants"]["system_swap"]["arms"]["semantic_only"][
        "pairs"
    ]
    assert {tuple(reversed(pair)) for pair in base} == {
        tuple(pair) for pair in swapped
    }
    evidence = result["private_mechanism_receipts"]
    assert evidence["availability"] == "COMPLETE"
    assert len(evidence["sides"]["left"]["document_envelope"]["leaf_records"]) == 1
    assert len(evidence["sides"]["right"]["document_envelope"]["leaf_records"]) == 1
    assert len(evidence["semantic_matrix"]["rows"]) == 4
    for side_name in ("left", "right"):
        side = evidence["sides"][side_name]
        for key in ("binder", "bounded_set", "slot_graph"):
            entry = side[key]["receipt"]
            raw = json.dumps(
                entry["receipt"],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii") + (b"\n" if entry["trailing_lf"] else b"")
            assert hashlib.sha256(raw).hexdigest() == entry["receipt_sha256"]
    for variant in action.VARIANT_NAMES:
        assert set(evidence["variants"][variant]) == {
            "semantic_mapping",
            "structural_mapping",
            "target_color_shuffle_mapping",
        }


def test_any_unexpected_document_typed_failure_invalidates_measurement() -> None:
    class _TypedLeaf:
        def select_story(self, story: str):
            raise ClosedChoiceV2Error("V2_PARSER_REJECTED")

    class _Failure:
        calls = 0

        def __call__(self, story: str):
            self.calls += 1
            return document_envelope.select_document_qualification_only(
                story, leaf_selector=_TypedLeaf()
            )

    selector = _Failure()
    with pytest.raises(
        action.ScarCssmActionInfrastructureError,
        match="SCAR_ACTION_TYPED_FAILURE_BLOCKED",
    ):
        action.form_scar_cssm_item_action_v1(
            _item(),
            document_selector=selector,
            encoder=_ExactTextEncoder(),
            encoder_binding_sha256=_ENCODER,
        )
    assert selector.calls == 2


def test_ambiguous_slot_set_fails_before_encoder_or_document() -> None:
    item = _item()
    item["variants"]["base"]["left"]["slots"][0]["surface"] = "K"
    item["variants"]["base"]["left"]["slots"][1]["surface"] = "K"
    # Re-establish the exact swap identity after the nested update.
    item["variants"]["system_swap"]["right"] = item["variants"]["base"]["left"]
    selector = _Selector()
    encoder = _ExactTextEncoder()
    result = action.form_scar_cssm_item_action_v1(
        item,
        document_selector=selector,
        encoder=encoder,
        encoder_binding_sha256=_ENCODER,
    )
    assert selector.calls == 0
    assert encoder.calls == 0
    assert result["execution"]["error_code"] == (
        "SLOT_BINDER_TYPED_FAILURE"
    )
    assert result["private_mechanism_receipts"] == {
        "availability": "PREMODEL_TYPED_FAILURE",
        "error_code": "SLOT_BINDER_TYPED_FAILURE",
        "semantic_matrix": None,
        "sides": {"left": None, "right": None},
        "variants": {"base": None, "system_swap": None},
    }
    assert all(
        arm["disposition"] == "ERROR"
        for variant in result["variants"].values()
        for arm in variant["arms"].values()
    )


def test_two_variant_structural_barrier_never_publishes_partial_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = action.map_slot_graphs_v1
    calls = 0

    def fail_last(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("synthetic last-arm failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(action, "map_slot_graphs_v1", fail_last)
    with pytest.raises(RuntimeError, match="synthetic last-arm failure"):
        action.form_scar_cssm_item_action_v1(
            _item(),
            document_selector=_Selector(),
            encoder=_ExactTextEncoder(),
            encoder_binding_sha256=_ENCODER,
        )
    assert calls == 6


def test_runtime_exception_and_model_failure_invalidate_item() -> None:
    class _Raise:
        def __call__(self, story: str):
            raise RuntimeError("synthetic infrastructure failure")

    with pytest.raises(RuntimeError, match="synthetic infrastructure failure"):
        action.form_scar_cssm_item_action_v1(
            _item(),
            document_selector=_Raise(),
            encoder=_ExactTextEncoder(),
            encoder_binding_sha256=_ENCODER,
        )

    class _ModelFailureLeaf:
        def select_story(self, story: str):
            raise ClosedChoiceV2Error("V2_MODEL_FORWARD_FAILED")

    def model_failure_document(story: str):
        return document_envelope.select_document_qualification_only(
            story, leaf_selector=_ModelFailureLeaf()
        )

    with pytest.raises(
        action.ScarCssmActionInfrastructureError,
        match="SCAR_ACTION_RUNTIME_INFRASTRUCTURE_FAILURE",
    ):
        action.form_scar_cssm_item_action_v1(
            _item(),
            document_selector=model_failure_document,
            encoder=_ExactTextEncoder(),
            encoder_binding_sha256=_ENCODER,
        )


def test_module_has_no_label_scorer_file_network_or_online_capability() -> None:
    source_text = inspect.getsource(action)
    for forbidden in (
        "import requests",
        "import urllib",
        "import socket",
        "import subprocess",
        "open(",
        "Path(",
        "gold_pairs=",
        "label_pack=",
        "scorer=",
    ):
        assert forbidden not in source_text
    assert tuple(
        inspect.signature(action.form_scar_cssm_item_action_v1).parameters
    ) == ("item", "document_selector", "encoder", "encoder_binding_sha256")
