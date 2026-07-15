from __future__ import annotations

from dataclasses import replace
import json

import pytest

from assumption_agent.models import stable_hash
from assumption_agent.typed_assignment_contract_v3 import (
    AssignmentReconciliationReceipt,
    ContentEvidenceProfile,
    ORGANIZE_PUBLIC_DESTINATIONS_V3,
    POSITIVE_CONTENT_EVIDENCE_BASIS,
    PUBLIC_DEFAULT_BASIS,
    PreAgentAssignmentReceipt,
    PublicDestinationSpec,
    TypedAssignmentContractError,
    evidence_set_hash,
    parse_typed_assignment_plan,
    typed_assignment_contract_hash,
)


PUBLIC_INSTRUCTION_WITH_DEFAULT = """
Organize every file into these five sibling subject folders without renaming:
LLM, trapped_ion_and_qc, black_hole, DNA, and music_history. If a paper
doesn't fit the first 4 categories, place it in the last one.
"""

PUBLIC_INSTRUCTION_WITHOUT_DEFAULT = """
Classify every file using content evidence into exactly these sibling folders:
LLM, trapped_ion_and_qc, black_hole, DNA, and music_history.
"""


def _profiles() -> tuple[ContentEvidenceProfile, ...]:
    first = ContentEvidenceProfile.from_extracted_text(
        source_name="language.pdf",
        source_sha256="1" * 64,
        source_size=101,
        media_kind="pdf",
        evidence=(("first_pages", "Transformer language model scaling"),),
    )
    second = ContentEvidenceProfile.from_extracted_text(
        source_name="misc.pdf",
        source_sha256="2" * 64,
        source_size=202,
        media_kind="pdf",
        evidence=(("first_pages", "A subject with no confident match"),),
    )
    return first, second


def _plan_payload(
    *,
    spec: PublicDestinationSpec,
    profiles: tuple[ContentEvidenceProfile, ...],
) -> dict[str, object]:
    first, second = profiles
    return {
        "contract_hash": typed_assignment_contract_hash(
            destination_spec=spec,
            profiles=profiles,
        ),
        "evidence_set_hash": evidence_set_hash(profiles),
        # Deliberately reverse agent order. The parsed plan is canonicalized.
        "assignments": [
            {
                "file_id": second.file_id,
                "destination": "music_history",
                "basis": PUBLIC_DEFAULT_BASIS,
                "evidence_ids": [],
            },
            {
                "file_id": first.file_id,
                "destination": "LLM",
                "basis": POSITIVE_CONTENT_EVIDENCE_BASIS,
                "evidence_ids": [first.evidence_ids[0]],
            },
        ],
    }


def test_public_destination_spec_is_closed_and_default_is_public_only() -> None:
    with_default = PublicDestinationSpec.from_public_instruction(
        PUBLIC_INSTRUCTION_WITH_DEFAULT
    )
    without_default = PublicDestinationSpec.from_public_instruction(
        PUBLIC_INSTRUCTION_WITHOUT_DEFAULT
    )

    assert with_default.destinations == ORGANIZE_PUBLIC_DESTINATIONS_V3
    assert with_default.default_destination == "music_history"
    assert without_default.default_destination is None
    assert "Organize every file" not in json.dumps(
        with_default.safe_payload(), sort_keys=True
    )

    with pytest.raises(TypedAssignmentContractError, match="missing"):
        PublicDestinationSpec.from_public_instruction(
            "Use LLM, trapped_ion_and_qc, black_hole, and DNA."
        )
    with pytest.raises(TypedAssignmentContractError, match="drifted"):
        with_default.verify(public_instruction=PUBLIC_INSTRUCTION_WITHOUT_DEFAULT)


def test_actual_public_last_folder_wording_is_a_declared_default() -> None:
    instruction = """
    The folders' name are as follows: LLM, trapped_ion_and_qc, black_hole,
    DNA, music_history. Each document belongs and only belongs to one subject
    folder (so if a file does not fit into any other 4 folders, it should fit
    into the last one).
    """

    spec = PublicDestinationSpec.from_public_instruction(instruction)

    assert spec.default_destination == "music_history"


def test_evidence_profile_binds_raw_content_without_persisting_it() -> None:
    profile = _profiles()[0]
    safe_text = json.dumps(profile.safe_payload(), sort_keys=True)
    agent_payload = profile.agent_payload()

    profile.verify()
    assert "language.pdf" not in safe_text
    assert "Transformer language model scaling" not in safe_text
    assert profile.source_name == agent_payload["source_name"]
    assert agent_payload["evidence"][0]["text"] == (
        "Transformer language model scaling"
    )
    assert profile.evidence_ids[0] == agent_payload["evidence"][0]["evidence_id"]

    with pytest.raises(TypedAssignmentContractError, match="basename"):
        ContentEvidenceProfile.from_extracted_text(
            source_name="../language.pdf",
            source_sha256="1" * 64,
            source_size=101,
            media_kind="pdf",
            evidence=(("first_pages", "content"),),
        )
    with pytest.raises(TypedAssignmentContractError, match="no positive evidence"):
        ContentEvidenceProfile.from_extracted_text(
            source_name="empty.pdf",
            source_sha256="3" * 64,
            source_size=0,
            media_kind="pdf",
            evidence=(),
        )


def test_plan_normalizes_order_and_enforces_exact_bijection() -> None:
    spec = PublicDestinationSpec.from_public_instruction(
        PUBLIC_INSTRUCTION_WITH_DEFAULT
    )
    profiles = _profiles()
    payload = _plan_payload(spec=spec, profiles=profiles)

    plan = parse_typed_assignment_plan(
        payload,
        destination_spec=spec,
        profiles=profiles,
    )
    assert [row.file_id for row in plan.assignments] == sorted(
        row.file_id for row in profiles
    )
    assert plan.canonical_json() == parse_typed_assignment_plan(
        plan.agent_payload(),
        destination_spec=spec,
        profiles=tuple(reversed(profiles)),
    ).canonical_json()

    unknown_top_key = {**payload, "repair": True}
    with pytest.raises(TypedAssignmentContractError, match="exactly"):
        parse_typed_assignment_plan(
            unknown_top_key,
            destination_spec=spec,
            profiles=profiles,
        )

    missing = {**payload, "assignments": payload["assignments"][:1]}
    with pytest.raises(TypedAssignmentContractError, match="cover every"):
        parse_typed_assignment_plan(
            missing,
            destination_spec=spec,
            profiles=profiles,
        )

    duplicate = {
        **payload,
        "assignments": [
            payload["assignments"][0],
            payload["assignments"][0],
        ],
    }
    with pytest.raises(TypedAssignmentContractError, match="more than once"):
        parse_typed_assignment_plan(
            duplicate,
            destination_spec=spec,
            profiles=profiles,
        )


def test_plan_rejects_cross_file_evidence_and_unpublished_default() -> None:
    profiles = _profiles()
    with_default = PublicDestinationSpec.from_public_instruction(
        PUBLIC_INSTRUCTION_WITH_DEFAULT
    )
    payload = _plan_payload(spec=with_default, profiles=profiles)
    first_assignment = payload["assignments"][1]
    assert isinstance(first_assignment, dict)
    first_assignment["evidence_ids"] = [profiles[1].evidence_ids[0]]
    with pytest.raises(TypedAssignmentContractError, match="another file"):
        parse_typed_assignment_plan(
            payload,
            destination_spec=with_default,
            profiles=profiles,
        )

    without_default = PublicDestinationSpec.from_public_instruction(
        PUBLIC_INSTRUCTION_WITHOUT_DEFAULT
    )
    payload = _plan_payload(spec=without_default, profiles=profiles)
    with pytest.raises(TypedAssignmentContractError, match="not authorized"):
        parse_typed_assignment_plan(
            payload,
            destination_spec=without_default,
            profiles=profiles,
        )


def test_pre_agent_and_reconciliation_receipts_are_safe_and_exact() -> None:
    spec = PublicDestinationSpec.from_public_instruction(
        PUBLIC_INSTRUCTION_WITH_DEFAULT
    )
    profiles = _profiles()
    plan = parse_typed_assignment_plan(
        _plan_payload(spec=spec, profiles=profiles),
        destination_spec=spec,
        profiles=profiles,
    )
    source_tree_hash = stable_hash({"source_tree": "before"})
    pre = PreAgentAssignmentReceipt.create(
        request_hash="a" * 64,
        destination_spec=spec,
        profiles=profiles,
        source_tree_hash_before=source_tree_hash,
        source_tree_hash_after_preparation=source_tree_hash,
        evidence_artifact_sha256="b" * 64,
        evidence_artifact_size=4096,
        evidence_artifact_locator_hash="c" * 64,
        plan_artifact_locator_hash="d" * 64,
    )
    reopened = {
        row.file_id: row.destination
        for row in plan.assignments
    }
    reconciliation = AssignmentReconciliationReceipt.create(
        pre_agent_receipt=pre,
        plan=plan,
        reopened_assignments=reopened,
        source_tree_hash_before_apply=source_tree_hash,
        source_tree_hash_after_apply=stable_hash({"source_tree": "after"}),
        applied_file_count=2,
        source_file_count_after_apply=0,
    )

    pre.verify(destination_spec=spec, profiles=profiles)
    reconciliation.verify(pre_agent_receipt=pre, plan=plan)
    safe_text = json.dumps(
        {
            "pre": pre.safe_payload(),
            "reconciliation": reconciliation.safe_payload(),
        },
        sort_keys=True,
    )
    assert "language.pdf" not in safe_text
    assert "Transformer language model scaling" not in safe_text
    assert reconciliation.safe_payload()["exact_assignment_reconciliation"]

    with pytest.raises(TypedAssignmentContractError, match="agent mutated"):
        replace(
            reconciliation,
            source_tree_hash_before_apply=stable_hash({"source_tree": "mutated"}),
        ).verify(pre_agent_receipt=pre, plan=plan)
    with pytest.raises(TypedAssignmentContractError, match="differs"):
        replace(
            reconciliation,
            reopened_assignment_map_hash="e" * 64,
        ).verify(pre_agent_receipt=pre, plan=plan)
