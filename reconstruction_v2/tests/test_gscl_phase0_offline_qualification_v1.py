from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json

import pytest

from assumption_agent.benchmarks import (
    gscl_phase0_offline_qualification_v1 as qualification,
)
from assumption_agent.generalized_structural_correspondence_v1 import (
    ObservableBinding,
    build_gscl_schema_registry_v1,
    strict_content_hash,
)
from assumption_agent.structural_law_residuals_v1 import (
    evaluate_bound_law,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)


def test_phase0_qualification_is_deterministic_safe_and_nonformal() -> None:
    first = qualification.run_qualification()
    second = qualification.run_qualification()

    assert first == second
    assert first["status"] == "PASS_PHASE0_KERNEL_ONLY"
    assert first["qualification_scope"] == "phase0_kernel_only"
    assert first["formal_result"] is False
    assert first["efficacy_evidence"] is False
    assert first["full_qualification_ready"] is False
    assert first["law_case_count"] == 5
    assert first["issue_ids"] == []
    assert first["all_primary_satisfied"] is True
    assert first["all_hard_negatives_rejected"] is True
    assert (
        first["all_entity_renamed_correspondences_accepted"]
        is True
    )
    assert (
        first["all_preregistered_semantic_attacks_rejected"]
        is True
    )
    assert first["same_process_byte_exact_replay"] is True
    assert set(first["declared_capability_surface"].values()) == {
        False
    }
    assert first["runtime_access_audited"] is False
    assert "capability_counts" not in first


def test_phase0_receipt_has_exact_five_law_coverage() -> None:
    receipt = qualification.run_qualification()

    assert len(receipt["law_cases"]) == 5
    assert {
        row["law_id"] for row in receipt["law_cases"]
    } == {
        "gscl.v1.t05_pair_interaction",
        "gscl.v1.t09_path_composition",
        "gscl.v1.t14_finite_equivariance",
        "gscl.v1.t15_closed_balance",
        "gscl.v1.t17_monotone_order",
    }
    for row in receipt["law_cases"]:
        assert row["primary_disposition"] == "satisfied"
        assert row["renamed_primary_disposition"] == "satisfied"
        assert row["correspondence_disposition"] == "accepted"
        assert {
            item["disposition"] for item in row["hard_negatives"]
        } == {"violated"}
        assert all(
            len(item["operator_contract_hash"]) == 64
            for item in row["hard_negatives"]
        )
        assert row["abstention_disposition"] in {
            "inconclusive",
            "not_applicable",
        }
        assert row["trusted_primary_recomputation"] is True
        assert row["receipt_builder_internal_recomputation"] is True
        assert (
            row["preregistered_semantic_attack_rejected"] is True
        )
        assert row["entity_renaming_signature_equal"] is True
        assert row["safe_private_separation"] is True


def test_phase0_receipt_commitments_recompute_exactly() -> None:
    receipt = qualification.run_qualification()
    without_self = dict(receipt)
    self_hash = without_self.pop("self_hash")

    assert strict_content_hash(without_self) == self_hash
    assert (
        strict_content_hash(receipt["issue_ids"])
        == receipt["issue_commitment"]
    )


def test_phase0_safe_receipt_omits_private_and_scoring_fields() -> None:
    encoded = json.dumps(
        qualification.run_qualification(),
        sort_keys=True,
    )

    for forbidden in (
        "episode_id",
        "binding_id",
        "receipt_id",
        "span_id",
        "start_byte",
        "end_byte",
        "raw_text",
        "query",
        "label",
        "per_item_score",
    ):
        assert forbidden not in encoded


def test_phase0_core_has_no_runtime_or_network_imports() -> None:
    source = inspect.getsource(qualification)
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(
        {
            "os",
            "pathlib",
            "socket",
            "subprocess",
            "requests",
            "httpx",
            "torch",
            "transformers",
        }
    )


def test_all_five_laws_reject_role_observable_semantic_attacks() -> None:
    registry = build_gscl_schema_registry_v1(
        build_universal_assumption_ontology_v1()
    )
    for case in qualification.build_fixed_cases(registry):
        episode, binding = qualification._build_episode_and_binding(
            registry,
            case,
            case_key=f"semantic.attack.{case.schema.law_kind.value}",
        )
        observables = {
            row.observable_id: row for row in episode.observables
        }
        kind = case.schema.law_kind.value
        if kind == "equivariance":
            target_id = "outputs_after"
            payload = qualification._vector_payload(-2, 999)
        elif kind == "monotone_order":
            target_id = "comparable_output_pairs"
            payload = {
                "pairs": [
                    {
                        "lower": qualification._rational(1),
                        "upper": qualification._rational(2),
                    },
                    {
                        "lower": qualification._rational(2),
                        "upper": qualification._rational(3),
                    },
                ]
            }
        elif kind == "closed_balance":
            target_id = "boundary_declaration"
            payload = {
                "boundary_id": "raw.unbound.boundary",
                "complete": True,
            }
        elif kind == "path_composition":
            target_id = "finite_domain"
            payload = {
                "values": ["raw.source.id", "local:aux_source"]
            }
        elif kind == "low_order_interaction":
            target_id = "components"
            payload = {
                "values": [
                    "role:component_a",
                    "role:component_b",
                    "role:unbound_component",
                ]
            }
        else:
            raise AssertionError(f"unexpected law kind: {kind}")
        observables[target_id] = replace(
            observables[target_id], value_payload=payload
        )
        attacked_episode = replace(
            episode,
            observables=tuple(observables.values()),
        )
        attacked_binding = replace(
            binding,
            episode_hash=attacked_episode.episode_hash,
            observable_bindings=tuple(
                ObservableBinding(
                    row.observable_id, row.observable_hash
                )
                for row in attacked_episode.observables
            ),
        )
        with pytest.raises(PermissionError, match="semantic_"):
            evaluate_bound_law(
                registry,
                case.schema,
                attacked_episode,
                attacked_binding,
                case.policy,
            )


def test_all_five_laws_reject_malformed_typed_payloads() -> None:
    registry = build_gscl_schema_registry_v1(
        build_universal_assumption_ontology_v1()
    )
    malformed = {
        "equivariance": (
            (
                "output_action",
                {"permutation": "bad", "signs": [1]},
            ),
            (
                "output_action",
                {"permutation": [0], "signs": [0]},
            ),
            (
                "output_action",
                {"permutation": [1], "signs": [1]},
            ),
        ),
        "monotone_order": (
            ("declared_direction", {"direction": "bad"}),
            ("declared_direction", {"direction": 2}),
        ),
        "closed_balance": (
            (
                "quantity_ledger",
                {
                    "storage_before": qualification._rational(10),
                    "storage_after": qualification._rational(13),
                    "inflows": "bad",
                    "outflows": [qualification._rational(2)],
                    "sources": [],
                    "sinks": [],
                },
            ),
        ),
        "path_composition": (
            ("first_map", {"rows": "bad"}),
        ),
        "low_order_interaction": (
            ("interaction_expectation", {"value": "bad"}),
            (
                "components",
                {
                    "values": [
                        "role:component_a",
                        "role:component_b",
                        "role:component_c",
                        "role:component_c",
                    ]
                },
            ),
            (
                "designated_pair",
                {
                    "values": [
                        "role:component_a",
                        "role:component_b",
                        "role:component_b",
                    ]
                },
            ),
        ),
    }
    for case in qualification.build_fixed_cases(registry):
        for attack_index, (target_id, payload) in enumerate(
            malformed[case.schema.law_kind.value]
        ):
            episode, binding = qualification._build_episode_and_binding(
                registry,
                case,
                case_key=(
                    f"malformed.{case.schema.law_kind.value}."
                    f"{attack_index}"
                ),
            )
            attacked_observables = tuple(
                replace(row, value_payload=payload)
                if row.observable_id == target_id
                else row
                for row in episode.observables
            )
            attacked_episode = replace(
                episode, observables=attacked_observables
            )
            attacked_binding = replace(
                binding,
                episode_hash=attacked_episode.episode_hash,
                observable_bindings=tuple(
                    ObservableBinding(
                        row.observable_id, row.observable_hash
                    )
                    for row in attacked_episode.observables
                ),
            )
            with pytest.raises(
                PermissionError,
                match=(
                    "semantic_typed_observable_payload_invalid"
                    "|semantic_interaction_"
                ),
            ):
                evaluate_bound_law(
                    registry,
                    case.schema,
                    attacked_episode,
                    attacked_binding,
                    case.policy,
                )
