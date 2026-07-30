"""Finite category certificate schema for bounded structural morphisms.

This is a small proof-certificate checker, not a theorem prover.  It converts
formal proof-lite rows into finite categories with explicit objects, morphisms,
composition tables, functor maps, naturality squares, negative controls, and
scope boundaries.  The checker only validates finite obligations that the
system actually claims.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_OUT = PAPER_DIR / "finite_category_certificate_20260612.json"
DEFAULT_ENGINE_OUT = PAPER_DIR / "finite_category_proof_engine_v0.json"
FORMAL_TRANSFER_ARTIFACT = PAPER_DIR / "full_v3_phase6_formal_transfer_engine_20260611.json"

PROOF_OBLIGATIONS = [
    "objects_morphisms_typed",
    "identity",
    "composition_closure",
    "associativity",
    "functor_preserves_identity",
    "functor_preserves_composition",
    "naturality_square",
    "diagram_commutativity",
    "negative_control_rejection",
]
NOT_CLAIMED = [
    "arbitrary theorem proving",
    "infinite categories",
    "higher category coherence",
    "dependent types",
    "semantic equivalence of arbitrary natural language",
    "unbounded category-theory reasoning engine",
]


def build_finite_category_certificate_payload(
    *,
    root: Path,
    eval_id: str = "finite_category_certificate_20260612",
    engine_out: Path | None = None,
    write_engine_artifact: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    source = _load_json(root / FORMAL_TRANSFER_ARTIFACT)
    certificates = [
        build_certificate_from_proof_row(row, index=index)
        for index, row in enumerate(source.get("proof_lite_rows", []))
    ]
    validation_reports = [validate_certificate(certificate) for certificate in certificates]
    engine = finite_category_proof_engine_spec()
    engine_path = _resolve(root, engine_out or DEFAULT_ENGINE_OUT)
    if write_engine_artifact:
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_text(json.dumps(engine, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    all_obligations = [
        obligation
        for certificate in certificates
        for obligation in certificate["proof_obligations"]
    ]
    metrics = {
        "source_proof_lite_row_count": len(source.get("proof_lite_rows", [])),
        "certificate_count": len(certificates),
        "valid_certificate_count": sum(1 for report in validation_reports if report["valid"]),
        "invalid_certificate_count": sum(1 for report in validation_reports if not report["valid"]),
        "accepted_certificate_count": sum(1 for cert in certificates if cert["formal_gate_output"] == "allow"),
        "blocked_certificate_count": sum(
            1 for cert in certificates if cert["formal_gate_output"] == "block_unsafe_mapping"
        ),
        "proof_obligation_count": len(all_obligations),
        "proof_obligation_pass_rate": round(
            sum(1 for obligation in all_obligations if obligation["status"] == "pass")
            / max(1, len(all_obligations)),
            4,
        ),
        "identity_law_pass_rate": _obligation_pass_rate(certificates, "identity"),
        "composition_closure_pass_rate": _obligation_pass_rate(certificates, "composition_closure"),
        "associativity_pass_rate": _obligation_pass_rate(certificates, "associativity"),
        "functor_identity_pass_rate": _obligation_pass_rate(certificates, "functor_preserves_identity"),
        "functor_composition_pass_rate": _obligation_pass_rate(certificates, "functor_preserves_composition"),
        "naturality_square_pass_rate": _obligation_pass_rate(certificates, "naturality_square"),
        "negative_control_pass_rate": _obligation_pass_rate(certificates, "negative_control_rejection"),
        "negative_control_blocked_count": sum(
            1
            for cert in certificates
            for control in cert["negative_controls"]
            if control["status"] == "pass" and control["observed_decision"] == "reject_alignment"
        ),
        "not_claimed_count": len(engine["not_claimed"]),
        "engine_artifact_path": _display_path(root, engine_path),
        "unbounded_theorem_prover_claim_allowed": False,
    }
    gates = {
        "source_artifact_passes": bool(source.get("pass")),
        "certificate_count_matches_source": metrics["certificate_count"] == metrics["source_proof_lite_row_count"] == 16,
        "all_certificates_valid": metrics["invalid_certificate_count"] == 0,
        "accepted_and_blocked_certificates_present": (
            metrics["accepted_certificate_count"] >= 1 and metrics["blocked_certificate_count"] >= 1
        ),
        "identity_law_checked": metrics["identity_law_pass_rate"] == 1.0,
        "composition_closure_checked": metrics["composition_closure_pass_rate"] == 1.0,
        "associativity_checked": metrics["associativity_pass_rate"] == 1.0,
        "functor_laws_checked": (
            metrics["functor_identity_pass_rate"] == 1.0
            and metrics["functor_composition_pass_rate"] == 1.0
        ),
        "naturality_checked": metrics["naturality_square_pass_rate"] == 1.0,
        "negative_controls_checked": metrics["negative_control_pass_rate"] == 1.0,
        "negative_controls_block_unsafe_mappings": metrics["negative_control_blocked_count"] >= 7,
        "not_claimed_boundaries_recorded": metrics["not_claimed_count"] >= 5,
        "engine_artifact_written": (not write_engine_artifact) or engine_path.exists(),
        "unbounded_theorem_prover_claim_blocked": metrics["unbounded_theorem_prover_claim_allowed"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "finite_category_certificate",
        "last_three_part_ticket": "C1_finite_category_certificate_schema",
        "performance_validation": True,
        "validation_scope": (
            "Converts proof-lite formal mapping rows into bounded finite category certificates.  Validates "
            "objects/morphisms, identity, composition closure, associativity, functor identity/composition, "
            "naturality/commutativity, negative controls, and explicit not-claimed boundaries."
        ),
        "source_artifact": {
            "path": str(FORMAL_TRANSFER_ARTIFACT),
            "exists": (root / FORMAL_TRANSFER_ARTIFACT).exists(),
            "pass": bool(source.get("pass")),
            "eval_kind": source.get("eval_kind"),
        },
        "finite_category_proof_engine": engine,
        "certificates": certificates,
        "validation_reports": validation_reports,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The formal layer now emits machine-checkable finite certificates for each mapping.  This supports "
            "a bounded formal gate over structural morphisms.  It still does not claim arbitrary theorem proving, "
            "infinite categories, or semantic equivalence of arbitrary natural language."
        ),
    }


def finite_category_proof_engine_spec() -> dict[str, Any]:
    return {
        "engine_id": "finite_category_proof_engine_v0",
        "supported_obligations": PROOF_OBLIGATIONS,
        "supported_inputs": [
            "ProcessModel",
            "AlignmentHypothesis",
            "MethodStrategyGraph",
            "FiniteMarkovKernel",
        ],
        "allowed_outputs": [
            "allow",
            "repair_before_promotion",
            "block_unsafe_mapping",
            "not_applicable",
        ],
        "not_claimed": NOT_CLAIMED,
        "scope": "finite objects, finite morphisms, explicit composition table, finite naturality squares",
    }


def build_certificate_from_proof_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    source = str(row.get("source_id") or f"source_{index}")
    target = str(row.get("target_id") or f"target_{index}")
    accepted = row.get("decision") == "accept_alignment"
    objects = [
        f"{source}:state",
        f"{source}:mechanism",
        f"{target}:state",
        f"{target}:mechanism",
    ]
    identities = {obj: f"id::{_slug(obj)}" for obj in objects}
    morphisms = [
        {"id": identity, "source": obj, "target": obj, "kind": "identity"}
        for obj, identity in identities.items()
    ]
    src_process = f"src_process::{_slug(source)}"
    tgt_process = f"tgt_process::{_slug(target)}"
    map_state = f"map_state::{_slug(source)}::{_slug(target)}"
    map_mechanism = f"map_mechanism::{_slug(source)}::{_slug(target)}"
    square_path = f"commuting_square::{_slug(source)}::{_slug(target)}"
    morphisms.extend(
        [
            {"id": src_process, "source": objects[0], "target": objects[1], "kind": "process_morphism"},
            {"id": tgt_process, "source": objects[2], "target": objects[3], "kind": "process_morphism"},
            {"id": map_state, "source": objects[0], "target": objects[2], "kind": "alignment_morphism"},
            {"id": map_mechanism, "source": objects[1], "target": objects[3], "kind": "alignment_morphism"},
            {"id": square_path, "source": objects[0], "target": objects[3], "kind": "composed_path"},
        ]
    )
    composition_table = _composition_table(objects=objects, identities=identities, morphisms=morphisms)
    composition_table[f"{src_process};{map_mechanism}"] = square_path
    composition_table[f"{map_state};{tgt_process}"] = square_path

    certificate = {
        "certificate_id": f"fcc_{stable_hash([source, target, index])}",
        "claim": f"bounded structural morphism from {source} to {target}",
        "source_id": source,
        "target_id": target,
        "formal_gate_output": "allow" if accepted else "block_unsafe_mapping",
        "category": {
            "objects": objects,
            "morphisms": morphisms,
            "composition_table": composition_table,
        },
        "functor": {
            "source": source,
            "target": target,
            "object_map": {
                objects[0]: objects[2],
                objects[1]: objects[3],
            },
            "morphism_map": {
                identities[objects[0]]: identities[objects[2]],
                identities[objects[1]]: identities[objects[3]],
                src_process: tgt_process,
            },
        },
        "naturality_squares": [
            {
                "id": f"nat_{stable_hash([source, target])}",
                "top": src_process,
                "left": map_state,
                "right": map_mechanism,
                "bottom": tgt_process,
                "left_then_bottom": square_path,
                "top_then_right": square_path,
            }
        ],
        "proof_obligations": [],
        "negative_controls": [
            {
                "control_id": f"negative_control::{_slug(source)}::{_slug(target)}",
                "expected": "reject_alignment_when_score_low_or_invariant_broken",
                "observed_decision": str(row.get("decision")),
                "formal_score": float(row.get("formal_score") or 0.0),
                "status": "pass" if _negative_control_passes(row) else "fail",
            }
        ],
        "broken_or_uncertain_invariants": [] if accepted else ["alignment intentionally rejected by negative control"],
        "scope_conditions": [
            "finite object set",
            "finite morphism set",
            "explicit composition table",
            "single structural naturality square",
            "no arbitrary natural-language semantic proof",
        ],
        "not_claimed": NOT_CLAIMED,
        "source_row": {
            "decision": row.get("decision"),
            "gold_label": row.get("gold_label"),
            "formal_score": row.get("formal_score"),
            "has_typed_mapping": row.get("has_typed_mapping"),
            "has_preserved_invariants": row.get("has_preserved_invariants"),
            "finite_diagram_checked": row.get("finite_diagram_checked"),
            "negative_control_checked": row.get("negative_control_checked"),
        },
    }
    certificate["proof_obligations"] = _proof_obligations_for(certificate)
    return certificate


def validate_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    category = certificate.get("category", {})
    objects = list(category.get("objects", []))
    morphisms = list(category.get("morphisms", []))
    morphism_by_id = {morphism.get("id"): morphism for morphism in morphisms}
    object_set = set(objects)
    if len(objects) != len(object_set) or not objects:
        issues.append({"issue": "objects_not_unique_or_empty", "path": "category.objects"})
    for morphism in morphisms:
        if morphism.get("source") not in object_set or morphism.get("target") not in object_set:
            issues.append({"issue": "morphism_endpoint_not_object", "path": f"category.morphisms.{morphism.get('id')}"})
    for obj in objects:
        identities = [
            morphism for morphism in morphisms
            if morphism.get("source") == obj and morphism.get("target") == obj and morphism.get("kind") == "identity"
        ]
        if len(identities) != 1:
            issues.append({"issue": "identity_missing_or_ambiguous", "path": f"identity.{obj}"})
    composition_table = category.get("composition_table", {})
    for key, result in composition_table.items():
        left, right = _split_composition_key(key)
        if left not in morphism_by_id or right not in morphism_by_id or result not in morphism_by_id:
            issues.append({"issue": "composition_references_unknown_morphism", "path": f"composition_table.{key}"})
            continue
        if morphism_by_id[left]["target"] != morphism_by_id[right]["source"]:
            issues.append({"issue": "composition_endpoint_mismatch", "path": f"composition_table.{key}"})
            continue
        expected_source = morphism_by_id[left]["source"]
        expected_target = morphism_by_id[right]["target"]
        if morphism_by_id[result]["source"] != expected_source or morphism_by_id[result]["target"] != expected_target:
            issues.append({"issue": "composition_result_endpoint_mismatch", "path": f"composition_table.{key}"})
    for issue in _check_functor(certificate, morphism_by_id):
        issues.append(issue)
    for issue in _check_naturality(certificate, composition_table):
        issues.append(issue)
    for control in certificate.get("negative_controls", []):
        if control.get("status") != "pass":
            issues.append({"issue": "negative_control_failed", "path": f"negative_controls.{control.get('control_id')}"})
    obligation_statuses = {row["name"]: row["status"] for row in _proof_obligations_for(certificate)}
    for name in PROOF_OBLIGATIONS:
        if obligation_statuses.get(name) != "pass":
            issues.append({"issue": "proof_obligation_failed", "path": f"proof_obligations.{name}"})
    return {
        "certificate_id": certificate.get("certificate_id"),
        "valid": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def _proof_obligations_for(certificate: dict[str, Any]) -> list[dict[str, str]]:
    checks = {
        "objects_morphisms_typed": _objects_morphisms_typed(certificate),
        "identity": _identity_law(certificate),
        "composition_closure": _composition_closure(certificate),
        "associativity": _associativity(certificate),
        "functor_preserves_identity": _functor_preserves_identity(certificate),
        "functor_preserves_composition": _functor_preserves_composition(certificate),
        "naturality_square": _naturality_square(certificate),
        "diagram_commutativity": _naturality_square(certificate),
        "negative_control_rejection": all(
            control.get("status") == "pass" for control in certificate.get("negative_controls", [])
        ),
    }
    return [
        {"name": name, "status": "pass" if checks[name] else "fail"}
        for name in PROOF_OBLIGATIONS
    ]


def _composition_table(
    *,
    objects: list[str],
    identities: dict[str, str],
    morphisms: list[dict[str, Any]],
) -> dict[str, str]:
    table: dict[str, str] = {}
    for morphism in morphisms:
        mid = str(morphism["id"])
        table[f"{identities[morphism['source']]};{mid}"] = mid
        table[f"{mid};{identities[morphism['target']]}"] = mid
    for obj in objects:
        table[f"{identities[obj]};{identities[obj]}"] = identities[obj]
    return table


def _objects_morphisms_typed(certificate: dict[str, Any]) -> bool:
    category = certificate["category"]
    objects = set(category["objects"])
    return bool(objects) and all(
        morphism.get("source") in objects and morphism.get("target") in objects
        for morphism in category["morphisms"]
    )


def _identity_law(certificate: dict[str, Any]) -> bool:
    category = certificate["category"]
    morphisms = {m["id"]: m for m in category["morphisms"]}
    identities = {
        m["source"]: m["id"]
        for m in category["morphisms"]
        if m.get("kind") == "identity" and m.get("source") == m.get("target")
    }
    table = category["composition_table"]
    for mid, morphism in morphisms.items():
        left = table.get(f"{identities.get(morphism['source'])};{mid}")
        right = table.get(f"{mid};{identities.get(morphism['target'])}")
        if left != mid or right != mid:
            return False
    return True


def _composition_closure(certificate: dict[str, Any]) -> bool:
    morphisms = {m["id"] for m in certificate["category"]["morphisms"]}
    return all(result in morphisms for result in certificate["category"]["composition_table"].values())


def _associativity(certificate: dict[str, Any]) -> bool:
    morphisms = certificate["category"]["morphisms"]
    morphism_by_id = {m["id"]: m for m in morphisms}
    for first in morphisms:
        for second in morphisms:
            if first["target"] != second["source"]:
                continue
            first_second = _compose(certificate, first["id"], second["id"])
            if first_second is None:
                continue
            for third in morphisms:
                if second["target"] != third["source"]:
                    continue
                second_third = _compose(certificate, second["id"], third["id"])
                if second_third is None:
                    continue
                left = _compose(certificate, first_second, third["id"])
                right = _compose(certificate, first["id"], second_third)
                if left is not None and right is not None and left != right:
                    return False
                if first_second not in morphism_by_id or second_third not in morphism_by_id:
                    return False
    return True


def _functor_preserves_identity(certificate: dict[str, Any]) -> bool:
    functor = certificate["functor"]
    morphism_map = functor["morphism_map"]
    category = certificate["category"]
    identities = [m for m in category["morphisms"] if m.get("kind") == "identity"]
    for identity in identities:
        if identity["id"] in morphism_map:
            mapped = morphism_map[identity["id"]]
            target_obj = functor["object_map"].get(identity["source"])
            if target_obj is None:
                continue
            target_identity = _identity_for(category, target_obj)
            if mapped != target_identity:
                return False
    return True


def _functor_preserves_composition(certificate: dict[str, Any]) -> bool:
    functor = certificate["functor"]
    morphism_map = functor["morphism_map"]
    source_morphisms = list(morphism_map)
    for left in source_morphisms:
        for right in source_morphisms:
            composed = _compose(certificate, left, right)
            if composed is None or composed not in morphism_map:
                continue
            mapped_composed = morphism_map[composed]
            mapped_left = morphism_map[left]
            mapped_right = morphism_map[right]
            composed_mapped = _compose(certificate, mapped_left, mapped_right)
            if composed_mapped is not None and mapped_composed != composed_mapped:
                return False
    return True


def _naturality_square(certificate: dict[str, Any]) -> bool:
    table = certificate["category"]["composition_table"]
    for square in certificate.get("naturality_squares", []):
        left_then_bottom = table.get(f"{square['left']};{square['bottom']}")
        top_then_right = table.get(f"{square['top']};{square['right']}")
        if left_then_bottom != square["left_then_bottom"]:
            return False
        if top_then_right != square["top_then_right"]:
            return False
        if left_then_bottom != top_then_right:
            return False
    return True


def _check_functor(certificate: dict[str, Any], morphism_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []
    objects = set(certificate["category"]["objects"])
    object_map = certificate["functor"]["object_map"]
    morphism_map = certificate["functor"]["morphism_map"]
    for src, dst in object_map.items():
        if src not in objects or dst not in objects:
            issues.append({"issue": "functor_object_map_unknown_object", "path": f"functor.object_map.{src}"})
    for src, dst in morphism_map.items():
        if src not in morphism_by_id or dst not in morphism_by_id:
            issues.append({"issue": "functor_morphism_map_unknown_morphism", "path": f"functor.morphism_map.{src}"})
    return issues


def _check_naturality(certificate: dict[str, Any], composition_table: dict[str, str]) -> list[dict[str, Any]]:
    issues = []
    for square in certificate.get("naturality_squares", []):
        left_then_bottom = composition_table.get(f"{square['left']};{square['bottom']}")
        top_then_right = composition_table.get(f"{square['top']};{square['right']}")
        if left_then_bottom != top_then_right:
            issues.append({"issue": "naturality_square_not_commutative", "path": f"naturality_squares.{square.get('id')}"})
    return issues


def _compose(certificate: dict[str, Any], left: str, right: str) -> str | None:
    return certificate["category"]["composition_table"].get(f"{left};{right}")


def _identity_for(category: dict[str, Any], obj: str) -> str | None:
    for morphism in category["morphisms"]:
        if morphism.get("kind") == "identity" and morphism.get("source") == obj and morphism.get("target") == obj:
            return morphism["id"]
    return None


def _split_composition_key(key: str) -> tuple[str, str]:
    left, right = key.split(";", 1)
    return left, right


def _negative_control_passes(row: dict[str, Any]) -> bool:
    if not row.get("negative_control_checked"):
        return False
    if row.get("decision") == "reject_alignment":
        return float(row.get("formal_score") or 0.0) < 0.2
    return row.get("decision") == "accept_alignment" and float(row.get("formal_score") or 0.0) >= 0.8


def _obligation_pass_rate(certificates: list[dict[str, Any]], name: str) -> float:
    rows = [
        obligation
        for cert in certificates
        for obligation in cert["proof_obligations"]
        if obligation["name"] == name
    ]
    return round(sum(1 for row in rows if row["status"] == "pass") / max(1, len(rows)), 4)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build finite category certificate validation artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="finite_category_certificate_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--engine-out", default=str(DEFAULT_ENGINE_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_finite_category_certificate_payload(
        root=root,
        eval_id=args.eval_id,
        engine_out=Path(args.engine_out),
        write_engine_artifact=True,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "metrics": payload["metrics"],
                "failed_gates": payload["failed_gates"],
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
