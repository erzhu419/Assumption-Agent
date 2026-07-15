"""One-shot offline runner for the frozen synthetic SC-100 shadow corpus.

The module is deliberately an orchestration boundary.  Candidate generation is
completed and joined before latent gold is loaded or any oracle is started.
Production candidate code is imported lazily so importing this module cannot
open or otherwise inspect the successor implementation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence

from ..models import stable_hash
from .sc100_shadow_gold_adapter_v1 import AdaptedShadowRecord, load_shadow_gold_jsonl
from .sc100_shadow_oracle_qualification_v1 import _run_fixture


RUNNER_VERSION = "sc100-synthetic-shadow-v1"
PREREGISTRATION_SCHEMA = "sc100-synthetic-shadow-preregistration-v1"
GENERATION_PARALLELISM = 24
ORACLE_PARALLELISM = 18
EXPECTED_CASE_IDS = frozenset(
    [*(f"S{i:02d}" for i in range(1, 13)),
     *(f"C{i:02d}" for i in range(1, 7)),
     *(f"N{i:02d}" for i in range(1, 7))]
)
EXPECTED_COUNTS = {"required_positive": 12, "coverage_probe": 6, "true_negative": 6}


class SC100SyntheticShadowError(RuntimeError):
    """Raised when the preregistered one-shot contract cannot be audited."""


Operator = Callable[..., Mapping[str, Any]]
Oracle = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    prompt_path: Path
    prompt_sha256: str
    output_path: Path


@dataclass
class GenerationRow:
    case_id: str
    prompt_sha256: str
    output_path: Path
    output_sha256: str | None
    receipt: Mapping[str, Any] | None
    receipt_hash_valid: bool
    codes: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SC100SyntheticShadowError(f"JSON object required: {path}")
    return value


def _pop_hash_valid(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        return False
    body = dict(value)
    body.pop(field, None)
    return stable_hash(body) == declared


def _operator_receipt_hash_valid(value: Mapping[str, Any]) -> bool:
    """Validate the frozen operator's pop-then-hash receipt convention."""

    declared = value.get("receipt_hash")
    if not isinstance(declared, str) or len(declared) != 64:
        return False
    body = dict(value)
    body.pop("receipt_hash", None)
    return stable_hash(body) == declared


def _resolve_under(project: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SC100SyntheticShadowError("relative path is missing")
    candidate = (project / relative).resolve()
    try:
        candidate.relative_to(project)
    except ValueError as exc:
        raise PermissionError("path escapes project root") from exc
    return candidate


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json_0600(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    if not _pop_hash_valid(payload, field):
        raise SC100SyntheticShadowError(f"{field} mismatch")
    return str(payload[field])


def _verify_file_bindings(project: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    rows = manifest.get("file_bindings")
    if not isinstance(rows, list) or not rows:
        raise SC100SyntheticShadowError("file bindings are missing")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SC100SyntheticShadowError("malformed file binding")
        relative, expected = row.get("path"), row.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in result
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise SC100SyntheticShadowError("invalid file binding")
        path = _resolve_under(project, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise SC100SyntheticShadowError(f"bound file mismatch: {relative}")
        result[relative] = expected
    return result


def _require_bound_path(
    project: Path, manifest: Mapping[str, Any], bindings: Mapping[str, str], key: str
) -> Path:
    relative = manifest.get(key)
    if not isinstance(relative, str) or relative not in bindings:
        raise SC100SyntheticShadowError(f"{key} is not file-bound")
    return _resolve_under(project, relative)


def _verify_corpus_spec(spec_path: Path) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    spec = _read_json(spec_path)
    self_hash = spec.get("corpus_self_hash")
    if not isinstance(self_hash, Mapping) or self_hash.get("algorithm") != "sha256":
        raise SC100SyntheticShadowError("corpus self hash is missing")
    declared = self_hash.get("value")
    body = json.loads(json.dumps(spec))
    body["corpus_self_hash"]["value"] = "0" * 64
    raw = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if declared != hashlib.sha256(raw).hexdigest():
        raise SC100SyntheticShadowError("corpus self hash mismatch")

    order = spec.get("case_order")
    seed = spec.get("seed")
    if (
        not isinstance(order, list)
        or any(not isinstance(case_id, str) for case_id in order)
        or len(order) != len(set(order))
        or set(order) != EXPECTED_CASE_IDS
        or not isinstance(seed, str)
    ):
        raise SC100SyntheticShadowError("invalid corpus case order")
    actual = sorted(
        order,
        key=lambda case_id: hashlib.sha256(
            seed.encode("utf-8") + b"\x00" + case_id.encode("utf-8")
        ).digest(),
    )
    if actual != order:
        raise SC100SyntheticShadowError("corpus case order does not use a real NUL")

    payload = spec.get("payload_sha256")
    if not isinstance(payload, Mapping) or not payload:
        raise SC100SyntheticShadowError("corpus payload bindings are missing")
    payload_hashes: dict[str, str] = {}
    for relative, expected in payload.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise SC100SyntheticShadowError("invalid corpus payload binding")
        path = (spec_path.parent / relative).resolve()
        try:
            path.relative_to(spec_path.parent.resolve())
        except ValueError as exc:
            raise PermissionError("corpus payload escapes corpus root") from exc
        if not path.is_file() or _sha256(path) != expected:
            raise SC100SyntheticShadowError(f"corpus payload mismatch: {relative}")
        payload_hashes[relative] = expected
    expected_files = {
        path.relative_to(spec_path.parent).as_posix()
        for path in spec_path.parent.rglob("*")
        if path.is_file() and path != spec_path
    }
    if set(payload_hashes) != expected_files:
        raise SC100SyntheticShadowError("corpus payload file set mismatch")
    return spec, list(order), payload_hashes


def _formal_paths(project: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    raw = manifest.get("formal_paths")
    if not isinstance(raw, Mapping):
        raise SC100SyntheticShadowError("formal paths are missing")
    paths = {
        key: _resolve_under(project, raw.get(key))
        for key in ("root", "report", "decision_lock", "outputs")
    }
    if any(path != paths["root"] and paths["root"] not in path.parents for path in paths.values()):
        raise PermissionError("formal paths do not share the formal root")
    if paths["report"].parent != paths["root"] or paths["decision_lock"].parent != paths["root"]:
        raise PermissionError("report and decision lock must be direct children of formal root")
    return paths


def _production_operator(*, instruction: str, blank_pdf: Path, output_pdf: Path) -> Mapping[str, Any]:
    module = importlib.import_module("assumption_agent.benchmarks.sc100_typed_operator_v2")
    execute = getattr(module, "execute")
    return execute(instruction=instruction, blank_pdf=blank_pdf, output_pdf=output_pdf)


def _verify_runtime_image(runtime: Mapping[str, Any]) -> None:
    inspected = subprocess.run(
        ["docker", "image", "inspect", str(runtime.get("image_id"))],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if inspected.returncode != 0:
        raise SC100SyntheticShadowError("frozen oracle image unavailable")
    try:
        payload = json.loads(inspected.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise SC100SyntheticShadowError("frozen oracle image inspect malformed") from exc
    if payload.get("Id") != runtime.get("image_id"):
        raise SC100SyntheticShadowError("frozen oracle image ID drift")


def _verify_generation_runtime(binding: object) -> None:
    if not isinstance(binding, Mapping):
        raise SC100SyntheticShadowError("generation runtime binding missing")
    try:
        import fitz
    except ImportError as exc:
        raise SC100SyntheticShadowError("frozen PyMuPDF runtime unavailable") from exc
    pdftotext = Path(str(binding.get("pdftotext_path", ""))).resolve()
    if (
        platform.python_version() != binding.get("python_version")
        or str(fitz.__version__) != binding.get("pymupdf_version")
        or not pdftotext.is_file()
        or _sha256(pdftotext) != binding.get("pdftotext_sha256")
    ):
        raise SC100SyntheticShadowError("generation runtime drift")
    completed = subprocess.run(
        [str(pdftotext), "-v"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    version_text = completed.stdout + completed.stderr
    if (
        completed.returncode != 0
        or str(binding.get("pdftotext_version", "")) not in version_text
    ):
        raise SC100SyntheticShadowError("pdftotext runtime drift")


def _run_generation(
    case: GenerationCase, *, blank_pdf: Path, blank_sha256: str, operator: Operator
) -> GenerationRow:
    codes: list[str] = []
    receipt: Mapping[str, Any] | None = None
    case.output_path.parent.mkdir(parents=True, exist_ok=False, mode=0o700)
    instruction = case.prompt_path.read_text(encoding="utf-8")
    if _sha256(case.prompt_path) != case.prompt_sha256:
        codes.append("prompt_hash_drift")
    try:
        value = operator(
            instruction=instruction,
            blank_pdf=blank_pdf,
            output_pdf=case.output_path,
        )
        if isinstance(value, Mapping):
            receipt = dict(value)
        else:
            codes.append("operator_receipt_not_object")
    except Exception as exc:  # The one-shot report records only a hash, never text.
        codes.extend(("operator_exception", f"operator_exception_type:{type(exc).__name__}"))
    del instruction

    files = sorted(path for path in case.output_path.parent.rglob("*") if path.is_file())
    output_hash: str | None = None
    if case.output_path.is_file() and not case.output_path.is_symlink():
        output_hash = _sha256(case.output_path)
    if _sha256(blank_pdf) != blank_sha256:
        codes.append("source_sha256_changed")
    receipt_hash_valid = bool(receipt and _operator_receipt_hash_valid(receipt))
    if receipt is not None and not receipt_hash_valid:
        codes.append("operator_receipt_hash_invalid")
    if len(files) > 1 or (files and files != [case.output_path]):
        codes.append("unexpected_candidate_files")
    return GenerationRow(
        case.case_id,
        case.prompt_sha256,
        case.output_path,
        output_hash,
        receipt,
        receipt_hash_valid,
        sorted(set(codes)),
    )


def _validate_generation(
    row: GenerationRow,
    gold: AdaptedShadowRecord,
    blank_sha256: str,
    operator_version: str,
) -> list[str]:
    codes = list(row.codes)
    receipt = row.receipt or {}
    if not row.receipt_hash_valid:
        codes.append("operator_receipt_hash_invalid")
    if receipt.get("operator_version") != operator_version:
        codes.append("operator_version_mismatch")
    if receipt.get("input_sha256") != blank_sha256:
        codes.append("operator_input_binding_mismatch")
    if receipt.get("instruction_sha256") != row.prompt_sha256:
        codes.append("operator_instruction_binding_mismatch")
    if receipt.get("source_unchanged") is not True:
        codes.append("operator_source_unchanged_not_true")
    if receipt.get("partial_output_created") is not False:
        codes.append("operator_partial_flag_not_false")
    if receipt.get("raw_case_text_persisted") is not False:
        codes.append("operator_raw_text_flag_not_false")
    if gold.case_class == "true_negative":
        if receipt.get("action") != "reject":
            codes.append("negative_action_not_reject")
        if receipt.get("reason_code") != gold.rejection_reason:
            codes.append("negative_reason_mismatch")
        if receipt.get("output_pdf") is not None:
            codes.append("negative_receipt_output_not_null")
        if row.output_sha256 is not None:
            codes.append("negative_output_created")
            if row.output_sha256 == blank_sha256:
                codes.append("negative_blank_copy_created")
            else:
                codes.append("negative_partial_output_created")
    else:
        if receipt.get("action") != "fill":
            codes.append("fill_action_not_fill")
        if row.output_sha256 is None:
            codes.append("fill_pdf_missing")
        elif row.output_sha256 == blank_sha256:
            codes.append("fill_is_blank_copy")
        else:
            try:
                if row.output_path.read_bytes()[:5] != b"%PDF-":
                    codes.append("fill_not_pdf")
            except OSError:
                codes.append("fill_pdf_unreadable")
        claimed = receipt.get("output_sha256")
        if claimed != row.output_sha256:
            codes.append("fill_receipt_output_hash_mismatch")
        if receipt.get("mutation_count") != 30:
            codes.append("fill_mutation_count_mismatch")
        if receipt.get("atomic_publish") is not True:
            codes.append("fill_atomic_publish_not_true")
        if receipt.get("temporary_cleanup_verified") is not True:
            codes.append("fill_temporary_cleanup_not_verified")
    return sorted(set(codes))


def _redact_operator_receipt(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    # The operator receipt is already content-redacted: it contains hashes,
    # counts, booleans and enum codes, never instruction text or extracted facts.
    return dict(receipt)


def _oracle_receipt_valid(
    receipt: Mapping[str, Any] | None,
    *, blank_sha256: str,
    filled_sha256: str | None,
    semantic_gold: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    codes: list[str] = []
    if receipt is None:
        return False, ["oracle_receipt_missing"]
    if not _pop_hash_valid(receipt, "receipt_sha256"):
        codes.append("oracle_receipt_hash_invalid")
    failures = receipt.get("failure_codes")
    if receipt.get("qualified") is not True or failures != []:
        codes.append("oracle_not_qualified")
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        codes.append("oracle_bindings_missing")
    else:
        if bindings.get("blank_sha256") != blank_sha256:
            codes.append("oracle_blank_binding_mismatch")
        if bindings.get("filled_sha256") != filled_sha256:
            codes.append("oracle_filled_binding_mismatch")
        if bindings.get("semantic_gold_sha256") != stable_hash(semantic_gold):
            codes.append("oracle_gold_binding_mismatch")
    return not codes, sorted(set(codes))


def _run_oracle(
    *,
    project: Path,
    manifest_hash: str,
    runtime: Mapping[str, Any],
    oracle_path: Path,
    blank_path: Path,
    row: GenerationRow,
    gold: AdaptedShadowRecord,
    oracle: Oracle | None,
) -> dict[str, Any]:
    assert gold.oracle_gold is not None
    if (
        row.output_sha256 is None
        or not row.output_path.is_file()
        or row.output_path.is_symlink()
    ):
        return {"receipt": None, "execution_codes": ["oracle_skipped_missing_fill"]}
    if oracle is not None:
        try:
            receipt = oracle(
                blank_pdf=blank_path,
                filled_pdf=row.output_path,
                semantic_gold=gold.oracle_gold,
            )
            return {"receipt": dict(receipt) if isinstance(receipt, Mapping) else None}
        except Exception as exc:
            return {
                "receipt": None,
                "execution_codes": ["oracle_exception", f"oracle_exception_type:{type(exc).__name__}"],
            }
    fixture = {
        "fixture_id": row.case_id,
        "kind": "positive_canary",
        "mutation_class": None,
        "filled_path": row.output_path,
        "filled_sha256": row.output_sha256,
        "semantic_gold": gold.oracle_gold,
        "semantic_gold_sha256": stable_hash(gold.oracle_gold),
        "expected_qualified": True,
        "must_include_failure_codes": (),
    }
    result = _run_fixture(
        project=project,
        manifest_hash=manifest_hash,
        runtime=runtime,
        oracle_path=oracle_path,
        blank_path=blank_path,
        fixture=fixture,
    )
    return {
        "receipt": result.get("receipt"),
        "execution_codes": [
            code
            for code, failed in (
                ("oracle_container_exit", result.get("container_exit") != 0),
                ("oracle_parse_error", result.get("parse_error") is True),
                ("oracle_cleanup_failed", result.get("container_cleanup_verified") is not True),
                ("oracle_network_not_none", result.get("container_network") != "none"),
            )
            if failed
        ],
    }


def _validate_corpus_shape(
    spec: Mapping[str, Any], order: Sequence[str], adapted: Sequence[AdaptedShadowRecord]
) -> dict[str, AdaptedShadowRecord]:
    by_id = {row.case_id: row for row in adapted}
    if len(by_id) != len(adapted) or set(by_id) != set(order) or set(order) != EXPECTED_CASE_IDS:
        raise SC100SyntheticShadowError("corpus/gold case ID closure mismatch")
    counts = {
        cohort: sum(row.case_class == cohort for row in adapted)
        for cohort in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS:
        raise SC100SyntheticShadowError("corpus cohort counts mismatch")
    cohorts = spec.get("cohorts")
    if not isinstance(cohorts, Mapping):
        raise SC100SyntheticShadowError("corpus cohorts missing")
    for cohort, count in EXPECTED_COUNTS.items():
        value = cohorts.get(cohort)
        if not isinstance(value, Mapping) or value.get("count") != count:
            raise SC100SyntheticShadowError("corpus cohort binding mismatch")
    return by_id


def _build_report(
    *,
    manifest_hash: str,
    blank_sha256: str,
    order: Sequence[str],
    generation: Mapping[str, GenerationRow],
    gold_by_id: Mapping[str, AdaptedShadowRecord],
    oracle_rows: Mapping[str, Mapping[str, Any]],
    candidate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "required_positive": [], "coverage_probe": [], "true_negative": []
    }
    required_passed = negative_passed = coverage_qualified = 0
    for case_id in order:
        generated = generation[case_id]
        gold = gold_by_id[case_id]
        generation_codes = _validate_generation(
            generated,
            gold,
            blank_sha256,
            str(candidate_binding["operator_version"]),
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "prompt_sha256": generated.prompt_sha256,
            "output_sha256": generated.output_sha256,
            "operator_receipt": _redact_operator_receipt(generated.receipt),
            "generation_codes": generation_codes,
        }
        if gold.case_class != "true_negative":
            oracle_result = oracle_rows.get(case_id, {})
            receipt = oracle_result.get("receipt")
            valid, oracle_codes = _oracle_receipt_valid(
                receipt if isinstance(receipt, Mapping) else None,
                blank_sha256=blank_sha256,
                filled_sha256=generated.output_sha256,
                semantic_gold=gold.oracle_gold or {},
            )
            oracle_codes.extend(oracle_result.get("execution_codes", ()))
            exact = not generation_codes and valid and not oracle_result.get("execution_codes")
            row.update(
                {
                    "oracle_receipt": receipt,
                    "oracle_codes": sorted(set(oracle_codes)),
                    "result_code": (
                        "qualified" if exact else
                        "coverage_starved" if gold.case_class == "coverage_probe" else
                        "not_qualified"
                    ),
                }
            )
            if gold.case_class == "required_positive":
                required_passed += int(exact)
            else:
                coverage_qualified += int(exact)
        else:
            exact = not generation_codes
            negative_passed += int(exact)
            row["result_code"] = "exact_reject_no_write" if exact else "negative_protocol_failed"
        grouped[gold.case_class].append(row)

    hard_pass = required_passed == 12 and negative_passed == 6
    report: dict[str, Any] = {
        "report_version": RUNNER_VERSION,
        "preregistration_manifest_hash": manifest_hash,
        "formal_decision_ordinal": 1,
        "formal_decision_budget": 1,
        "evaluation_class": "offline_synthetic_feasibility_only",
        "generation_parallelism": GENERATION_PARALLELISM,
        "oracle_parallelism": ORACLE_PARALLELISM,
        "all_generation_joined_before_any_oracle": True,
        "corpus_case_order_hash": stable_hash(list(order)),
        "blank_sha256": blank_sha256,
        "candidate_id": candidate_binding["candidate_id"],
        "operator_version": candidate_binding["operator_version"],
        "required_positive": grouped["required_positive"],
        "coverage_probe": grouped["coverage_probe"],
        "true_negative": grouped["true_negative"],
        "counts": {
            "required_positive_qualified": required_passed,
            "required_positive_required": 12,
            "coverage_probe_qualified": coverage_qualified,
            "coverage_probe_total": 6,
            "coverage_probe_coverage_starved": 6 - coverage_qualified,
            "true_negative_exact": negative_passed,
            "true_negative_required": 6,
        },
        "coverage_probe_affects_hard_decision": False,
        "coverage_probe_post_hoc_relabeling_allowed": False,
        "synthetic_feasibility_passed": hard_pass,
        "model_calls": 0,
        "ruoli_calls": 0,
        "online_judge_calls": 0,
        "official_test_calls": 0,
        "claim_boundary": {
            "task_utility_authorized": False,
            "development_authorized": False,
            "incumbent_authorized": False,
            "promotion_authorized": False,
            "validation_authorized": False,
            "benchmark_authorized": False,
            "sealed_test_authorized": False,
            "real_world_generalization_authorized": False,
            "general_sc100_or_legal_correctness_authorized": False,
            "causal_self_evolution_benefit_authorized": False,
        },
    }
    report["decision_hash"] = stable_hash(
        {
            "manifest_hash": manifest_hash,
            "candidate_id": candidate_binding["candidate_id"],
            "required_positive_qualified": required_passed,
            "required_positive_required": 12,
            "true_negative_exact": negative_passed,
            "true_negative_required": 6,
            "synthetic_feasibility_passed": hard_pass,
        }
    )
    report["report_hash"] = stable_hash(report)  # New pop-field convention.
    return report


def _preflight(
    project: Path, manifest: Mapping[str, Any]
) -> tuple[
    str,
    dict[str, Path],
    Path,
    Path,
    Path,
    dict[str, Any],
    list[str],
    dict[str, str],
    dict[str, Any],
]:
    if manifest.get("schema") != PREREGISTRATION_SCHEMA:
        raise SC100SyntheticShadowError("preregistration schema mismatch")
    if manifest.get("formal_decision_budget") != 1:
        raise SC100SyntheticShadowError("formal decision budget must equal one")
    manifest_hash = _verify_self_hash(manifest, "manifest_hash")
    bindings = _verify_file_bindings(project, manifest)
    spec_path = _require_bound_path(project, manifest, bindings, "corpus_spec_path")
    gold_path = _require_bound_path(project, manifest, bindings, "gold_path")
    blank_path = _require_bound_path(project, manifest, bindings, "blank_pdf_path")
    oracle_path = _require_bound_path(project, manifest, bindings, "oracle_source_path")
    qualification_path = _require_bound_path(
        project, manifest, bindings, "oracle_qualification_result_path"
    )
    candidate = manifest.get("candidate_binding")
    if not isinstance(candidate, Mapping):
        raise SC100SyntheticShadowError("candidate binding missing")
    candidate_body = dict(candidate)
    candidate_id = candidate_body.pop("candidate_id", None)
    for path_key, hash_key in (
        ("source_path", "source_sha256"),
        ("test_path", "test_sha256"),
    ):
        relative = candidate.get(path_key)
        expected = candidate.get(hash_key)
        if (
            not isinstance(relative, str)
            or bindings.get(relative) != expected
            or not isinstance(expected, str)
        ):
            raise SC100SyntheticShadowError("candidate file binding mismatch")
    if (
        not isinstance(candidate.get("operator_version"), str)
        or candidate_id != stable_hash(candidate_body)
    ):
        raise SC100SyntheticShadowError("candidate ID mismatch")
    spec, order, payload = _verify_corpus_spec(spec_path)
    if gold_path != (spec_path.parent / "gold.jsonl").resolve():
        raise SC100SyntheticShadowError("gold path is not corpus-bound")
    if payload.get("gold.jsonl") != _sha256(gold_path):
        raise SC100SyntheticShadowError("gold payload binding mismatch")
    if _sha256(blank_path) != manifest.get("blank_sha256"):
        raise SC100SyntheticShadowError("public blank binding mismatch")
    qualification = _read_json(qualification_path)
    if not _pop_hash_valid(qualification, "result_hash"):
        raise SC100SyntheticShadowError("oracle qualification result hash mismatch")
    status = qualification.get("qualification")
    if not isinstance(status, Mapping) or status.get("oracle_ready_for_frozen_measurement") is not True:
        raise SC100SyntheticShadowError("oracle is not qualified for frozen measurement")
    corpus_binding = qualification.get("corpus_binding")
    boundary = qualification.get("boundary")
    if (
        qualification.get("oracle_source_sha256") != _sha256(oracle_path)
        or not isinstance(corpus_binding, Mapping)
        or corpus_binding.get("corpus_self_hash")
        != spec.get("corpus_self_hash", {}).get("value")
        or corpus_binding.get("corpus_spec_file_sha256") != _sha256(spec_path)
        or corpus_binding.get("gold_file_sha256") != _sha256(gold_path)
        or not isinstance(boundary, Mapping)
        or boundary.get("may_measure_one_frozen_synthetic_shadow") is not True
        or boundary.get("oracle_change_after_first_shadow_outcome_allowed") is not False
        or status.get("model_calls") != 0
        or status.get("ruoli_calls") != 0
        or status.get("online_judge_calls") != 0
    ):
        raise SC100SyntheticShadowError("oracle qualification boundary mismatch")
    runtime = manifest.get("runtime_binding")
    if not isinstance(runtime, Mapping) or not str(runtime.get("image_id", "")).startswith("sha256:"):
        raise SC100SyntheticShadowError("immutable runtime binding missing")
    return (
        manifest_hash,
        _formal_paths(project, manifest),
        spec_path,
        gold_path,
        blank_path,
        dict(runtime),
        order,
        payload,
        dict(candidate),
    )


def run_sc100_synthetic_shadow(
    *,
    project_root: str | Path,
    preregistration_path: str | Path,
    operator: Operator | None = None,
    oracle: Oracle | None = None,
) -> dict[str, Any]:
    """Consume the single formal decision and return a redacted report."""

    project = Path(project_root).expanduser().resolve(strict=True)
    manifest = _read_json(Path(preregistration_path).expanduser().resolve(strict=True))
    (
        manifest_hash,
        paths,
        spec_path,
        gold_path,
        blank_path,
        runtime,
        order,
        payload,
        candidate_binding,
    ) = _preflight(project, manifest)
    if operator is None:
        _verify_generation_runtime(manifest.get("generation_runtime_binding"))
    if oracle is None:
        _verify_runtime_image(runtime)
    oracle_path = _resolve_under(project, manifest["oracle_source_path"])
    if paths["root"].exists():
        raise FileExistsError("formal synthetic shadow root already exists")
    paths["root"].mkdir(parents=True, exist_ok=False, mode=0o700)
    paths["outputs"].mkdir(parents=True, exist_ok=False, mode=0o700)
    _write_json_exclusive(
        paths["decision_lock"],
        {
            "lock_version": "sc100-synthetic-shadow-lock-v1",
            "state": "reserved",
            "formal_decision_ordinal": 1,
            "formal_decision_budget": 1,
            "preregistration_manifest_hash": manifest_hash,
            "candidate_id": candidate_binding["candidate_id"],
            "retry_authorized": False,
        },
    )

    blank_sha256 = _sha256(blank_path)
    cases = [
        GenerationCase(
            case_id,
            (spec_path.parent / f"prompts/{case_id}.md").resolve(),
            payload[f"prompts/{case_id}.md"],
            paths["outputs"] / case_id / "candidate.pdf",
        )
        for case_id in order
    ]
    generation: dict[str, GenerationRow] = {}
    candidate = operator or _production_operator
    with ThreadPoolExecutor(max_workers=GENERATION_PARALLELISM) as pool:
        futures = {
            pool.submit(
                _run_generation,
                case,
                blank_pdf=blank_path,
                blank_sha256=blank_sha256,
                operator=candidate,
            ): case.case_id
            for case in cases
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                generation[case_id] = future.result()
            except Exception as exc:
                case = next(item for item in cases if item.case_id == case_id)
                generation[case_id] = GenerationRow(
                    case_id=case_id,
                    prompt_sha256=case.prompt_sha256,
                    output_path=case.output_path,
                    output_sha256=(
                        _sha256(case.output_path) if case.output_path.is_file() else None
                    ),
                    receipt=None,
                    receipt_hash_valid=False,
                    codes=[
                        "generation_worker_exception",
                        f"generation_worker_exception_type:{type(exc).__name__}",
                    ],
                )
    if len(generation) != len(order) or _sha256(blank_path) != blank_sha256:
        raise SC100SyntheticShadowError("candidate generation barrier failed")

    # Latent gold is intentionally first loaded after every candidate has joined.
    adapted = load_shadow_gold_jsonl(gold_path)
    spec = _read_json(spec_path)
    gold_by_id = _validate_corpus_shape(spec, order, adapted)
    oracle_rows: dict[str, Mapping[str, Any]] = {}
    fill_ids = [case_id for case_id in order if gold_by_id[case_id].case_class != "true_negative"]
    with ThreadPoolExecutor(max_workers=ORACLE_PARALLELISM) as pool:
        futures = {
            pool.submit(
                _run_oracle,
                project=project,
                manifest_hash=manifest_hash,
                runtime=runtime,
                oracle_path=oracle_path,
                blank_path=blank_path,
                row=generation[case_id],
                gold=gold_by_id[case_id],
                oracle=oracle,
            ): case_id
            for case_id in fill_ids
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                oracle_rows[case_id] = future.result()
            except Exception as exc:
                oracle_rows[case_id] = {
                    "receipt": None,
                    "execution_codes": [
                        "oracle_worker_exception",
                        f"oracle_worker_exception_type:{type(exc).__name__}",
                    ],
                }

    report = _build_report(
        manifest_hash=manifest_hash,
        blank_sha256=blank_sha256,
        order=order,
        generation=generation,
        gold_by_id=gold_by_id,
        oracle_rows=oracle_rows,
        candidate_binding=candidate_binding,
    )
    _write_json_exclusive(paths["report"], report)
    _replace_json_0600(
        paths["decision_lock"],
        {
            "lock_version": "sc100-synthetic-shadow-lock-v1",
            "state": "completed",
            "formal_decision_ordinal": 1,
            "formal_decision_budget": 1,
            "preregistration_manifest_hash": manifest_hash,
            "candidate_id": candidate_binding["candidate_id"],
            "decision_hash": report["decision_hash"],
            "report_hash": report["report_hash"],
            "synthetic_feasibility_passed": report["synthetic_feasibility_passed"],
            "retry_authorized": False,
        },
    )
    return report


def verify_existing_sc100_synthetic_shadow(
    *, project_root: str | Path, preregistration_path: str | Path
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    manifest = _read_json(Path(preregistration_path).expanduser().resolve(strict=True))
    manifest_hash, paths, *_, candidate_binding = _preflight(project, manifest)
    report = _read_json(paths["report"])
    report_hash = _verify_self_hash(report, "report_hash")
    lock = _read_json(paths["decision_lock"])
    if (
        stat.S_IMODE(paths["decision_lock"].stat().st_mode) != 0o600
        or stat.S_IMODE(paths["report"].stat().st_mode) != 0o600
    ):
        raise SC100SyntheticShadowError("formal JSON mode is not 0600")
    if (
        lock.get("state") != "completed"
        or lock.get("formal_decision_ordinal") != 1
        or lock.get("formal_decision_budget") != 1
        or lock.get("preregistration_manifest_hash") != manifest_hash
        or lock.get("candidate_id") != candidate_binding.get("candidate_id")
        or lock.get("report_hash") != report_hash
        or lock.get("decision_hash") != report.get("decision_hash")
        or lock.get("retry_authorized") is not False
        or report.get("all_generation_joined_before_any_oracle") is not True
        or report.get("generation_parallelism") != GENERATION_PARALLELISM
        or report.get("oracle_parallelism") != ORACLE_PARALLELISM
        or report.get("model_calls") != 0
        or report.get("ruoli_calls") != 0
        or report.get("online_judge_calls") != 0
    ):
        raise SC100SyntheticShadowError("decision lock does not bind report")
    for cohort in ("required_positive", "coverage_probe", "true_negative"):
        rows = report.get(cohort)
        if not isinstance(rows, list):
            raise SC100SyntheticShadowError("report rows malformed")
        for row in rows:
            if not isinstance(row, Mapping):
                raise SC100SyntheticShadowError("report row malformed")
            case_id = row.get("case_id")
            if not isinstance(case_id, str):
                raise SC100SyntheticShadowError("report case ID malformed")
            output = paths["outputs"] / case_id / "candidate.pdf"
            expected_output_hash = row.get("output_sha256")
            if cohort == "true_negative":
                if output.exists() or expected_output_hash is not None:
                    raise SC100SyntheticShadowError("negative output exists")
            elif expected_output_hash is None:
                if output.exists():
                    raise SC100SyntheticShadowError("unbound failed output exists")
            elif (
                not isinstance(expected_output_hash, str)
                or not output.is_file()
                or _sha256(output) != expected_output_hash
            ):
                raise SC100SyntheticShadowError("filled output hash mismatch")
            operator_receipt = row.get("operator_receipt")
            if operator_receipt is not None and (
                not isinstance(operator_receipt, Mapping)
                or not _operator_receipt_hash_valid(operator_receipt)
                or operator_receipt.get("operator_version")
                != candidate_binding.get("operator_version")
            ):
                raise SC100SyntheticShadowError("operator receipt hash mismatch")
            receipt = row.get("oracle_receipt")
            if receipt is not None and (
                not isinstance(receipt, Mapping) or not _pop_hash_valid(receipt, "receipt_sha256")
            ):
                raise SC100SyntheticShadowError("oracle receipt hash mismatch")
    return report


run_synthetic_shadow = run_sc100_synthetic_shadow
verify_existing_synthetic_shadow = verify_existing_sc100_synthetic_shadow


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    function = (
        verify_existing_sc100_synthetic_shadow
        if args.verify_existing
        else run_sc100_synthetic_shadow
    )
    report = function(project_root=args.project_root, preregistration_path=args.preregistration)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
