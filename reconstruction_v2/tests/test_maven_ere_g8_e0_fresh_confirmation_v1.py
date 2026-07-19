from __future__ import annotations

import hashlib

from assumption_agent.benchmarks import maven_ere_g8_e0_fresh_confirmation_v1 as fresh
from assumption_agent.benchmarks import maven_ere_g8_e1_acquisition_v1 as acquisition
from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_g8_e1_formal_controller_v1 as v1


def _documents() -> tuple[acquisition._Document, ...]:
    rows = []
    index = 0
    for family in core.FAMILY_ORDER:
        for within_family in range(45):
            events = (
                acquisition._EventRow(
                    f"event-{index}-0", "Attack", ((f"attack-{index}", 0),)
                ),
                acquisition._EventRow(
                    f"event-{index}-1", "Response", ((f"response-{index}", 1),)
                ),
            )
            family_pairs = {
                observed: ((0, 1),) if observed == family else ()
                for observed in core.FAMILY_ORDER
            }
            rows.append(
                acquisition._Document(
                    split="valid",
                    split_index=index,
                    source_id=f"document-{index}",
                    title=f"Unique title {index}",
                    sentences=(
                        f"Unique sentence {index} A.",
                        f"Unique sentence {index} B.",
                        f"Unique sentence {index} C.",
                    ),
                    events=events,
                    generic_relations=(),
                    family_pairs=family_pairs,
                )
            )
            index += 1
    return tuple(rows)


def test_original_assignment_exclusion_yields_fresh_disjoint_sixty() -> None:
    documents = _documents()
    original = fresh.assign_candidates(
        documents,
        secret=b"o" * 32,
        specs=fresh.ORIGINAL_VALID_SPECS,
    )
    assert len(original.selected_components) == 60
    new = fresh.assign_candidates(
        documents,
        secret=b"n" * 32,
        specs=fresh.FRESH_SPECS,
        excluded_components=original.selected_components,
    )
    assert len(new.selected_components) == 60
    assert original.selected_components.isdisjoint(new.selected_components)
    views, labels = fresh.build_packs(
        documents,
        new,
        specs=fresh.FRESH_SPECS,
    )
    assert views["A_hold"]["item_count"] == 60
    assert labels["A_hold"]["item_count"] == 60
    assert sum(
        row["family"] == "CAUSAL" for row in labels["A_hold"]["items"]
    ) == 20


def test_original_pack_regeneration_proof_uses_public_hashes_only() -> None:
    documents = _documents()
    original = fresh.assign_candidates(
        documents,
        secret=b"o" * 32,
        specs=fresh.ORIGINAL_VALID_SPECS,
    )
    views, labels = fresh.build_packs(
        documents,
        original,
        specs=fresh.ORIGINAL_VALID_SPECS,
    )
    receipt = {"view_pack_bindings": {}, "label_pack_bindings": {}}
    for block, _quota in fresh.ORIGINAL_VALID_SPECS:
        receipt["view_pack_bindings"][block] = fresh._pack_binding(views[block])
        receipt["label_pack_bindings"][block] = fresh._pack_binding(labels[block])
    proof = fresh._validate_reconstructed_original_packs(
        receipt,
        views,
        labels,
    )
    assert set(proof) == {"A_hold", "M_search"}
    assert all(
        len(row["view_file_sha256"]) == 64 for row in proof.values()
    )


def test_fixed_g8_model_roundtrip(tmp_path) -> None:
    model = core.G8Model(
        tuple(float(index) / 10.0 for index in range(len(core.G8_FEATURE_ORDER))),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        item_count=96,
        set_observation_count=123,
    )
    payload = v1._self_hashed(core.g8_model_payload(model), "model_sha256")
    raw = v1._canonical_bytes(payload)
    path = tmp_path / "G8.model.json"
    path.write_bytes(raw)
    loaded = fresh.load_g8_model(
        path,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert loaded == model


def test_primary_requires_both_controls_and_every_family() -> None:
    passed = {
        "net_utility": 6,
        "p_value": {"numerator": 1, "denominator": 64},
        "family_net": {family: 2 for family in core.FAMILY_ORDER},
    }
    assert fresh.primary_passed(
        {
            "comparisons": {
                "E0_minus_HippoRAG": passed,
                "E0_minus_RAW": passed,
            }
        }
    )
    family_zero = dict(passed)
    family_zero["family_net"] = {
        "CAUSAL": 3,
        "SUBEVENT": 3,
        "TEMPORAL": 0,
    }
    assert not fresh.primary_passed(
        {
            "comparisons": {
                "E0_minus_HippoRAG": passed,
                "E0_minus_RAW": family_zero,
            }
        }
    )
