"""Lean-style export for finite category certificates.

C2 does not turn the formal layer into a theorem prover.  It exports the C1
finite certificates into a small Lean-readable artifact with the finite data,
allowed gate outputs, expected proof obligations, and explicit not-claimed
boundaries.  If Lean is available locally, the export is syntax-checked without
requiring mathlib.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .finite_category_certificate import (
    NOT_CLAIMED,
    PROOF_OBLIGATIONS,
    build_finite_category_certificate_payload,
)


DEFAULT_LEAN_OUT = PAPER_DIR / "finite_category_certificate_20260612.lean"
DEFAULT_OUT = PAPER_DIR / "finite_category_lean_export_20260612.json"
ALLOWED_GATE_OUTPUTS = [
    "allow",
    "repair_before_promotion",
    "block_unsafe_mapping",
    "not_applicable",
]
FORBIDDEN_GENERATOR_OUTPUTS = [
    "generate_new_hypothesis",
    "synthesize_philosophical_rule",
    "auto_accept_without_live",
    "auto_apply_policy_change",
    "replace_judge",
]


def build_finite_category_lean_export_payload(
    *,
    root: Path,
    eval_id: str = "finite_category_lean_export_20260612",
    lean_out: Path | None = None,
    run_lean_if_available: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    lean_path = _resolve(root, lean_out or DEFAULT_LEAN_OUT)
    certificate_payload = build_finite_category_certificate_payload(
        root=root,
        eval_id=f"{eval_id}_source_c1",
        write_engine_artifact=False,
    )
    certificates = certificate_payload["certificates"]
    lean_text = render_lean_export(certificates)
    lean_path.parent.mkdir(parents=True, exist_ok=True)
    lean_path.write_text(lean_text, encoding="utf-8")
    text_validation = validate_lean_export_text(lean_text, certificates)
    external_check = _optional_lean_check(lean_path, run_lean_if_available=run_lean_if_available)
    metrics = {
        "certificate_count": len(certificates),
        "lean_definition_count": len(re.findall(r"^def cert\d+ : FiniteCertificate :=", lean_text, flags=re.MULTILINE)),
        "proof_obligation_name_count": len(PROOF_OBLIGATIONS),
        "supported_gate_output_count": len(ALLOWED_GATE_OUTPUTS),
        "forbidden_generator_output_count": sum(
            1 for token in FORBIDDEN_GENERATOR_OUTPUTS if token in lean_text
        ),
        "not_claimed_boundary_count": len(NOT_CLAIMED),
        "lean_text_line_count": len(lean_text.splitlines()),
        "external_lean_available": external_check["available"],
        "external_lean_check_passed": external_check["passed"],
        "full_theorem_prover_claim_allowed": False,
    }
    gates = {
        "source_certificates_pass": bool(certificate_payload["pass"]),
        "lean_file_written": lean_path.exists() and lean_path.stat().st_size > 0,
        "all_certificates_exported": metrics["lean_definition_count"] == metrics["certificate_count"] == 16,
        "lean_readable_structures_present": text_validation["lean_readable_structures_present"],
        "expected_proof_obligations_listed": text_validation["expected_proof_obligations_listed"],
        "allowed_gate_outputs_only": text_validation["allowed_gate_outputs_only"],
        "no_forbidden_generator_outputs": metrics["forbidden_generator_output_count"] == 0,
        "no_sorry_or_admit": text_validation["no_sorry_or_admit"],
        "not_claimed_boundaries_exported": text_validation["not_claimed_boundaries_exported"],
        "external_lean_check_not_failed": (
            (not external_check["attempted"]) or (not external_check["available"]) or external_check["passed"]
        ),
        "full_theorem_prover_claim_blocked": metrics["full_theorem_prover_claim_allowed"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "finite_category_lean_export",
        "last_three_part_ticket": "C2_finite_category_lean_export_stub",
        "performance_validation": True,
        "validation_scope": (
            "Exports bounded finite category certificates into a Lean-readable text artifact.  The formal "
            "engine remains a gate over externally generated candidates: it can allow, request repair, block, "
            "or mark not-applicable, but it cannot generate hypotheses, auto-apply graph changes, or replace "
            "live validation."
        ),
        "source": {
            "certificate_eval_id": certificate_payload["eval_id"],
            "certificate_pass": certificate_payload["pass"],
            "certificate_count": len(certificates),
        },
        "artifacts": {
            "lean_out": _display_path(root, lean_path),
        },
        "external_check": external_check,
        "text_validation": text_validation,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "claim_boundaries": [
            "Lean export is a proof-certificate interchange format, not a full mathlib-backed theorem prover.",
            "The formal layer is gate-only: generation remains the job of residual/LLM hypothesis generators.",
            "External Lean syntax checking is optional in this phase; mathlib integration is explicitly deferred.",
        ],
    }


def render_lean_export(certificates: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "/-",
        "Generated by assumption_os.finite_category_lean_export.",
        "This is a Lean-readable finite certificate export, not a mathlib category-theory development.",
        "The exported formal layer is gate-only and does not generate hypotheses.",
        "-/",
        "",
        "namespace AssumptionOSFiniteCategory",
        "",
        "structure FiniteMorphism where",
        "  id : String",
        "  source : String",
        "  target : String",
        "  kind : String",
        "deriving Repr",
        "",
        "structure FiniteCertificate where",
        "  certificateId : String",
        "  formalGateOutput : String",
        "  objects : List String",
        "  morphisms : List FiniteMorphism",
        "  compositionTable : List (String × String × String)",
        "  proofObligations : List (String × String)",
        "  notClaimed : List String",
        "deriving Repr",
        "",
        f"def supportedProofObligations : List String := {_lean_list(PROOF_OBLIGATIONS)}",
        f"def allowedGateOutputs : List String := {_lean_list(ALLOWED_GATE_OUTPUTS)}",
        f"def notClaimedBoundaries : List String := {_lean_list(NOT_CLAIMED)}",
        "",
    ]
    cert_names = []
    for index, certificate in enumerate(certificates):
        name = f"cert{index}"
        cert_names.append(name)
        lines.extend(_render_certificate_definition(name, certificate))
        lines.append("")
    lines.append(f"def certificates : List FiniteCertificate := [{', '.join(cert_names)}]")
    lines.append("")
    lines.append("-- Expected external checks for each certificate:")
    lines.append("-- 1. all morphism endpoints are typed by listed objects")
    lines.append("-- 2. identity and composition-table entries are closed")
    lines.append("-- 3. functor identity/composition rows preserve listed mappings")
    lines.append("-- 4. naturality-square paths commute")
    lines.append("-- 5. negative controls block unsafe mappings")
    lines.append("#eval certificates.length")
    lines.append("")
    lines.append("end AssumptionOSFiniteCategory")
    lines.append("")
    return "\n".join(lines)


def validate_lean_export_text(lean_text: str, certificates: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_gate_outputs = {
        certificate["formal_gate_output"]
        for certificate in certificates
        if certificate["formal_gate_output"] in ALLOWED_GATE_OUTPUTS
    }
    return {
        "lean_readable_structures_present": (
            "structure FiniteMorphism where" in lean_text
            and "structure FiniteCertificate where" in lean_text
            and "def certificates : List FiniteCertificate" in lean_text
        ),
        "expected_proof_obligations_listed": all(obligation in lean_text for obligation in PROOF_OBLIGATIONS),
        "all_certificate_ids_listed": all(certificate["certificate_id"] in lean_text for certificate in certificates),
        "allowed_gate_outputs_only": len(allowed_gate_outputs) >= 2
        and all(certificate["formal_gate_output"] in ALLOWED_GATE_OUTPUTS for certificate in certificates),
        "no_forbidden_generator_outputs": all(token not in lean_text for token in FORBIDDEN_GENERATOR_OUTPUTS),
        "no_sorry_or_admit": "sorry" not in lean_text and "admit" not in lean_text,
        "not_claimed_boundaries_exported": all(boundary in lean_text for boundary in NOT_CLAIMED),
        "full_theorem_prover_claim_absent": "full category-theory theorem prover" not in lean_text,
    }


def _render_certificate_definition(name: str, certificate: dict[str, Any]) -> list[str]:
    objects = [_ascii(value) for value in certificate["category"]["objects"]]
    morphisms = [
        {
            "id": _ascii(morphism["id"]),
            "source": _ascii(morphism["source"]),
            "target": _ascii(morphism["target"]),
            "kind": _ascii(morphism["kind"]),
        }
        for morphism in certificate["category"]["morphisms"]
    ]
    composition = [
        (_ascii(left), _ascii(right), _ascii(result))
        for key, result in sorted(certificate["category"]["composition_table"].items())
        for left, right in [key.split(";", 1)]
    ]
    obligations = [
        (_ascii(row["name"]), _ascii(row["status"]))
        for row in certificate["proof_obligations"]
    ]
    return [
        f"def {name} : FiniteCertificate := {{",
        f"  certificateId := {_lean_str(certificate['certificate_id'])},",
        f"  formalGateOutput := {_lean_str(certificate['formal_gate_output'])},",
        f"  objects := {_lean_list(objects)},",
        "  morphisms := [",
        *_indent([_render_morphism(morphism) + "," for morphism in morphisms], "    "),
        "  ],",
        "  compositionTable := [",
        *_indent([_render_tuple3(row) + "," for row in composition], "    "),
        "  ],",
        f"  proofObligations := {_lean_pair_list(obligations)},",
        f"  notClaimed := {_lean_list(NOT_CLAIMED)}",
        "}",
    ]


def _render_morphism(morphism: dict[str, str]) -> str:
    return (
        "{ "
        f"id := {_lean_str(morphism['id'])}, "
        f"source := {_lean_str(morphism['source'])}, "
        f"target := {_lean_str(morphism['target'])}, "
        f"kind := {_lean_str(morphism['kind'])}"
        " }"
    )


def _render_tuple3(row: tuple[str, str, str]) -> str:
    left, right, result = row
    return f"({_lean_str(left)}, {_lean_str(right)}, {_lean_str(result)})"


def _lean_pair_list(rows: list[tuple[str, str]]) -> str:
    return "[" + ", ".join(f"({_lean_str(left)}, {_lean_str(right)})" for left, right in rows) + "]"


def _lean_list(values: list[str]) -> str:
    return "[" + ", ".join(_lean_str(_ascii(value)) for value in values) + "]"


def _lean_str(value: str) -> str:
    return json.dumps(_ascii(value), ensure_ascii=True)


def _ascii(value: Any) -> str:
    text = str(value)
    return "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in text)


def _indent(lines: list[str], prefix: str) -> list[str]:
    return [prefix + line for line in lines]


def _optional_lean_check(lean_path: Path, *, run_lean_if_available: bool) -> dict[str, Any]:
    lean_bin = shutil.which("lean")
    if not run_lean_if_available or not lean_bin:
        return {
            "available": bool(lean_bin),
            "attempted": False,
            "passed": False,
            "reason": "lean_unavailable" if not lean_bin else "disabled",
        }
    try:
        result = subprocess.run(
            [lean_bin, str(lean_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "attempted": True,
            "passed": False,
            "returncode": None,
            "reason": "timeout",
            "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
        }
    return {
        "available": True,
        "attempted": True,
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export finite category certificates to Lean-style text.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="finite_category_lean_export_20260612")
    parser.add_argument("--lean-out", default=str(DEFAULT_LEAN_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-lean-check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_finite_category_lean_export_payload(
        root=root,
        eval_id=args.eval_id,
        lean_out=Path(args.lean_out),
        run_lean_if_available=not args.skip_lean_check,
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
                "artifacts": payload["artifacts"],
                "external_check": payload["external_check"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
