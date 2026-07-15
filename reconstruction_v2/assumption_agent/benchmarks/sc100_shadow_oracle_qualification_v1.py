"""Offline, one-decision conformance qualification for the SC-100 shadow oracle.

This runner never opens a shadow prompt or candidate output.  It checks the
measurement instrument against frozen consumed-TRAIN canaries and mutants in
parallel, with Docker networking disabled, before any successor is frozen.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from ..models import stable_hash


QUALIFICATION_VERSION = "sc100-shadow-oracle-qualification-v1"


class SC100OracleQualificationError(RuntimeError):
    """Raised when the frozen conformance contract cannot be audited."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SC100OracleQualificationError(f"JSON object required: {path}")
    return payload


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = payload.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        raise SC100OracleQualificationError(f"missing {field}")
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise SC100OracleQualificationError(f"{field} mismatch")
    return declared


def _resolve_under(project: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SC100OracleQualificationError("relative path is missing")
    candidate = (project / relative).resolve()
    try:
        candidate.relative_to(project)
    except ValueError as exc:
        raise PermissionError("path escapes project root") from exc
    return candidate


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_bound_files(project: Path, manifest: Mapping[str, Any]) -> None:
    bindings = manifest.get("file_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise SC100OracleQualificationError("file bindings are missing")
    seen: set[str] = set()
    for row in bindings:
        if not isinstance(row, Mapping):
            raise SC100OracleQualificationError("malformed file binding")
        relative = row.get("path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or relative in seen:
            raise SC100OracleQualificationError("invalid file binding path")
        seen.add(relative)
        path = _resolve_under(project, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise SC100OracleQualificationError(f"bound file mismatch: {relative}")


def _formal_paths(project: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    formal = manifest.get("formal_paths")
    if not isinstance(formal, Mapping):
        raise SC100OracleQualificationError("formal paths are missing")
    paths = {
        key: _resolve_under(project, formal.get(key))
        for key in ("root", "report", "decision_lock")
    }
    if paths["report"].parent != paths["root"] or paths["decision_lock"].parent != paths["root"]:
        raise PermissionError("formal paths are not rooted together")
    return paths


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current: dict[str, Any] = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise SC100OracleQualificationError(f"invalid gold patch path: {dotted}")
        current = child
    if parts[-1] not in current:
        raise SC100OracleQualificationError(f"gold patch target missing: {dotted}")
    current[parts[-1]] = value


def _resolved_fixtures(
    project: Path, specification: Mapping[str, Any]
) -> tuple[Path, list[dict[str, Any]]]:
    if specification.get("schema") != "sc100-shadow-oracle-qualification-fixtures-v1":
        raise SC100OracleQualificationError("fixture schema mismatch")
    provenance = specification.get("provenance")
    if not isinstance(provenance, Mapping) or any(
        provenance.get(key) is not False
        for key in ("shadow_outcomes_accessed", "candidate_score_computed")
    ) or provenance.get("shadow_case_count") != 0:
        raise PermissionError("fixture pack crosses the shadow boundary")
    blank = specification.get("blank")
    if not isinstance(blank, Mapping):
        raise SC100OracleQualificationError("blank binding missing")
    blank_path = _resolve_under(project, blank.get("path"))
    if _sha256(blank_path) != blank.get("sha256"):
        raise SC100OracleQualificationError("blank binding mismatch")

    raw_rows = specification.get("fixtures")
    if not isinstance(raw_rows, list) or len(raw_rows) != 7:
        raise SC100OracleQualificationError("fixture count mismatch")
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("fixture_id"), str):
            raise SC100OracleQualificationError("fixture row malformed")
        fixture_id = str(raw["fixture_id"])
        if fixture_id in raw_by_id:
            raise SC100OracleQualificationError("duplicate fixture ID")
        raw_by_id[fixture_id] = raw

    resolved: list[dict[str, Any]] = []
    resolved_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        fixture_id = str(raw["fixture_id"])
        pdf_binding = raw.get("filled_pdf")
        if pdf_binding is None:
            parent_id = raw.get("filled_pdf_ref")
            parent = resolved_by_id.get(str(parent_id))
            if parent is None:
                raise SC100OracleQualificationError("unresolved filled PDF reference")
            filled_path = parent["filled_path"]
            filled_sha256 = parent["filled_sha256"]
        else:
            if not isinstance(pdf_binding, Mapping):
                raise SC100OracleQualificationError("filled PDF binding malformed")
            filled_path = _resolve_under(project, pdf_binding.get("path"))
            filled_sha256 = str(pdf_binding.get("sha256") or "")
            if _sha256(filled_path) != filled_sha256:
                raise SC100OracleQualificationError(f"filled PDF drift: {fixture_id}")

        if "semantic_gold" in raw:
            gold = deepcopy(raw["semantic_gold"])
        else:
            parent_id = raw.get("semantic_gold_ref")
            parent = resolved_by_id.get(str(parent_id))
            if parent is None:
                raise SC100OracleQualificationError("unresolved semantic gold reference")
            gold = deepcopy(parent["semantic_gold"])
        if not isinstance(gold, dict):
            raise SC100OracleQualificationError("semantic gold must be an object")
        patch = raw.get("semantic_gold_patch", {})
        if not isinstance(patch, Mapping):
            raise SC100OracleQualificationError("semantic gold patch malformed")
        for dotted, value in patch.items():
            if not isinstance(dotted, str):
                raise SC100OracleQualificationError("semantic gold patch key malformed")
            _set_dotted(gold, dotted, value)

        must_include = raw.get("must_include_failure_codes")
        if not isinstance(must_include, list) or any(not isinstance(code, str) for code in must_include):
            raise SC100OracleQualificationError("failure-code expectation malformed")
        row = {
            "fixture_id": fixture_id,
            "kind": raw.get("kind"),
            "mutation_class": raw.get("mutation_class"),
            "filled_path": filled_path,
            "filled_sha256": filled_sha256,
            "semantic_gold": gold,
            "semantic_gold_sha256": stable_hash(gold),
            "expected_qualified": raw.get("expected_qualified"),
            "must_include_failure_codes": tuple(must_include),
        }
        if row["kind"] not in {"positive_canary", "mutant"}:
            raise SC100OracleQualificationError("fixture kind malformed")
        if type(row["expected_qualified"]) is not bool:
            raise SC100OracleQualificationError("fixture decision malformed")
        resolved.append(row)
        resolved_by_id[fixture_id] = row

    if sum(row["kind"] == "positive_canary" for row in resolved) != 2:
        raise SC100OracleQualificationError("positive canary count mismatch")
    if sum(row["kind"] == "mutant" for row in resolved) != 5:
        raise SC100OracleQualificationError("mutant count mismatch")
    return blank_path, resolved


_CONTAINER_PROGRAM = r"""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("sc100_shadow_oracle", "/oracle/sc100_shadow_oracle_v1.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with open("/fixtures/gold.json", "r", encoding="utf-8") as handle:
    gold = json.load(handle)
receipt = module.qualify_sc100_shadow(
    "/fixtures/blank.pdf",
    "/fixtures/filled.pdf",
    gold,
    pdftotext_bin="/usr/bin/pdftotext",
    pdftoppm_bin="/usr/bin/pdftoppm",
    render_dpi=144,
)
print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
""".strip()


def _run_fixture(
    *,
    project: Path,
    manifest_hash: str,
    runtime: Mapping[str, Any],
    oracle_path: Path,
    blank_path: Path,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    fixture_id = str(fixture["fixture_id"])
    container_name = f"sc100-oracle-q-{manifest_hash[:10]}-{fixture_id.lower().replace('_', '-')[:32]}"
    with tempfile.TemporaryDirectory(prefix="sc100-oracle-gold-") as directory:
        gold_path = Path(directory) / "gold.json"
        gold_path.write_text(
            json.dumps(fixture["semantic_gold"], sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        command = [
            "docker", "run", "--rm", "--name", container_name,
            "--pull", "never", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=256m",
            "-e", "PYTHONNOUSERSITE=1", "-e", "PIP_NO_INDEX=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{oracle_path}:/oracle/sc100_shadow_oracle_v1.py:ro",
            "-v", f"{blank_path}:/fixtures/blank.pdf:ro",
            "-v", f"{fixture['filled_path']}:/fixtures/filled.pdf:ro",
            "-v", f"{gold_path}:/fixtures/gold.json:ro",
            str(runtime["image_id"]), "python3", "-c", _CONTAINER_PROGRAM,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(runtime.get("fixture_timeout_seconds", 180)),
        )
    lingering = subprocess.run(
        ["docker", "container", "inspect", container_name],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    receipt: dict[str, Any] | None = None
    parse_error = False
    if completed.returncode == 0:
        try:
            value = json.loads(completed.stdout)
            if isinstance(value, dict):
                receipt = value
            else:
                parse_error = True
        except json.JSONDecodeError:
            parse_error = True
    failures = tuple(receipt.get("failure_codes", ())) if receipt else ()
    must_include = tuple(fixture["must_include_failure_codes"])
    receipt_hash_valid = False
    if receipt and isinstance(receipt.get("receipt_sha256"), str):
        body = dict(receipt)
        declared = body.pop("receipt_sha256")
        receipt_hash_valid = stable_hash(body) == declared
    qualified = receipt.get("qualified") if receipt else None
    expectation_met = bool(
        completed.returncode == 0
        and not parse_error
        and lingering.returncode != 0
        and receipt_hash_valid
        and qualified is fixture["expected_qualified"]
        and all(code in failures for code in must_include)
        and receipt.get("bindings", {}).get("filled_sha256") == fixture["filled_sha256"]
        and receipt.get("bindings", {}).get("semantic_gold_sha256")
        == fixture["semantic_gold_sha256"]
        and receipt.get("runtime", {}).get("pypdf") == runtime["pypdf_version"]
        and receipt.get("runtime", {}).get("pillow") == runtime["pillow_version"]
        and runtime["poppler_version"] in receipt.get("runtime", {}).get("pdftotext", "")
        and runtime["poppler_version"] in receipt.get("runtime", {}).get("pdftoppm", "")
    )
    return {
        "fixture_id": fixture_id,
        "kind": fixture["kind"],
        "mutation_class": fixture.get("mutation_class"),
        "filled_sha256": fixture["filled_sha256"],
        "semantic_gold_sha256": fixture["semantic_gold_sha256"],
        "expected_qualified": fixture["expected_qualified"],
        "must_include_failure_codes": list(must_include),
        "container_exit": completed.returncode,
        "container_network": "none",
        "container_cleanup_verified": lingering.returncode != 0,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "receipt": receipt,
        "receipt_hash_valid": receipt_hash_valid,
        "parse_error": parse_error,
        "expectation_met": expectation_met,
    }


def _preflight(
    project: Path, manifest: Mapping[str, Any]
) -> tuple[Path, list[dict[str, Any]], Path]:
    if manifest.get("qualification_version") != QUALIFICATION_VERSION:
        raise SC100OracleQualificationError("qualification version mismatch")
    if manifest.get("qualification_attempt_budget") != 1:
        raise SC100OracleQualificationError("attempt budget must equal one")
    scope = manifest.get("scope")
    if not isinstance(scope, Mapping) or scope.get("instrument_conformance_only") is not True:
        raise PermissionError("qualification scope mismatch")
    if scope.get("shadow_case_count") != 0 or scope.get("shadow_outcome_accessed") is not False:
        raise PermissionError("shadow boundary is not closed")
    _verify_bound_files(project, manifest)

    runtime = manifest.get("runtime_binding")
    if not isinstance(runtime, Mapping):
        raise SC100OracleQualificationError("runtime binding missing")
    inspected = subprocess.run(
        ["docker", "image", "inspect", str(runtime.get("image_id"))],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if inspected.returncode != 0:
        raise SC100OracleQualificationError("frozen image unavailable")
    image_payload = json.loads(inspected.stdout)[0]
    if image_payload.get("Id") != runtime.get("image_id"):
        raise SC100OracleQualificationError("image ID drift")

    fixture_path = _resolve_under(project, manifest.get("fixture_specification_path"))
    if _sha256(fixture_path) != manifest.get("fixture_specification_sha256"):
        raise SC100OracleQualificationError("fixture specification drift")
    blank_path, fixtures = _resolved_fixtures(project, _read_json(fixture_path))
    oracle_path = _resolve_under(project, manifest.get("oracle_source_path"))
    if _sha256(oracle_path) != manifest.get("oracle_source_sha256"):
        raise SC100OracleQualificationError("oracle source drift")
    return blank_path, fixtures, oracle_path


def run_oracle_qualification(
    *, project_root: str | Path, preregistration_path: str | Path
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    manifest = _read_json(Path(preregistration_path).expanduser().resolve(strict=True))
    manifest_hash = _verify_self_hash(manifest, "manifest_hash")
    blank_path, fixtures, oracle_path = _preflight(project, manifest)
    paths = _formal_paths(project, manifest)
    if paths["root"].exists():
        raise FileExistsError("formal oracle qualification root already exists")
    paths["root"].mkdir(parents=True, exist_ok=False)
    runtime = manifest["runtime_binding"]
    fixture_set_hash = stable_hash(
        [
            {
                key: row[key]
                for key in (
                    "fixture_id", "kind", "filled_sha256", "semantic_gold_sha256",
                    "expected_qualified", "must_include_failure_codes",
                )
            }
            for row in fixtures
        ]
    )
    _write_json_exclusive(
        paths["decision_lock"],
        {
            "lock_version": "sc100-shadow-oracle-qualification-lock-v1",
            "state": "reserved",
            "qualification_ordinal": 1,
            "preregistration_manifest_hash": manifest_hash,
            "oracle_source_sha256": manifest["oracle_source_sha256"],
            "fixture_set_hash": fixture_set_hash,
            "runtime_binding_hash": stable_hash(runtime),
            "retry_authorized": False,
        },
    )

    rows_by_id: dict[str, dict[str, Any]] = {}
    workers = len(fixtures)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_fixture,
                project=project,
                manifest_hash=manifest_hash,
                runtime=runtime,
                oracle_path=oracle_path,
                blank_path=blank_path,
                fixture=row,
            ): row
            for row in fixtures
        }
        for future in as_completed(futures):
            fixture = futures[future]
            try:
                rows_by_id[str(fixture["fixture_id"])] = future.result()
            except Exception as exc:
                rows_by_id[str(fixture["fixture_id"])] = {
                    "fixture_id": fixture["fixture_id"],
                    "kind": fixture["kind"],
                    "expectation_met": False,
                    "execution_error_type": type(exc).__name__,
                    "execution_error_hash": stable_hash({"error": str(exc)}),
                }
    rows = [rows_by_id[str(fixture["fixture_id"])] for fixture in fixtures]
    positive_rows = [row for row in rows if row["kind"] == "positive_canary"]
    mutant_rows = [row for row in rows if row["kind"] == "mutant"]
    positives_passed = sum(bool(row.get("expectation_met")) for row in positive_rows)
    mutants_rejected = sum(bool(row.get("expectation_met")) for row in mutant_rows)
    ready = positives_passed == 2 and mutants_rejected == 5
    receipt_set_hash = stable_hash(
        [
            {
                "fixture_id": row["fixture_id"],
                "receipt_sha256": (row.get("receipt") or {}).get("receipt_sha256"),
                "expectation_met": row.get("expectation_met"),
            }
            for row in rows
        ]
    )
    decision_hash = stable_hash(
        {
            "manifest_hash": manifest_hash,
            "oracle_source_sha256": manifest["oracle_source_sha256"],
            "runtime_binding_hash": stable_hash(runtime),
            "fixture_set_hash": fixture_set_hash,
            "fixture_receipt_set_hash": receipt_set_hash,
            "oracle_ready_for_frozen_measurement": ready,
        }
    )
    report: dict[str, Any] = {
        "report_version": QUALIFICATION_VERSION,
        "preregistration_manifest_hash": manifest_hash,
        "qualification_ordinal": 1,
        "qualification_attempt_budget": 1,
        "evaluation_class": "offline_instrument_conformance_only",
        "docker": {
            "image_id": runtime["image_id"],
            "container_network": "none",
            "pull_policy": "never",
            "read_only_root": True,
            "cap_drop_all": True,
            "no_new_privileges": True,
            "parallelism": workers,
            "runtime_binding_hash": stable_hash(runtime),
        },
        "fixture_set_hash": fixture_set_hash,
        "fixture_receipt_set_hash": receipt_set_hash,
        "rows": rows,
        "positive_canaries_passed": positives_passed,
        "positive_canaries_required": 2,
        "mutants_rejected": mutants_rejected,
        "mutants_required": 5,
        "oracle_ready_for_frozen_measurement": ready,
        "shadow_case_count": 0,
        "shadow_outcome_accessed": False,
        "candidate_bound": False,
        "candidate_score_computed": False,
        "agent_calls": 0,
        "model_calls": 0,
        "ruoli_calls": 0,
        "online_judge_calls": 0,
        "task_utility_authorized": False,
        "development_authorized": False,
        "incumbent_authorized": False,
        "promotion_authorized": False,
        "validation_authorized": False,
        "sealed_test_authorized": False,
        "decision_hash": decision_hash,
    }
    report["report_hash"] = stable_hash(report)
    _write_json_exclusive(paths["report"], report)
    _replace_json(
        paths["decision_lock"],
        {
            "lock_version": "sc100-shadow-oracle-qualification-lock-v1",
            "state": "completed",
            "qualification_ordinal": 1,
            "preregistration_manifest_hash": manifest_hash,
            "oracle_source_sha256": manifest["oracle_source_sha256"],
            "fixture_set_hash": fixture_set_hash,
            "runtime_binding_hash": stable_hash(runtime),
            "fixture_receipt_set_hash": receipt_set_hash,
            "decision_hash": decision_hash,
            "report_hash": report["report_hash"],
            "oracle_ready_for_frozen_measurement": ready,
            "retry_authorized": False,
        },
    )
    return report


def verify_existing_oracle_qualification(
    *, project_root: str | Path, preregistration_path: str | Path
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    manifest = _read_json(Path(preregistration_path).expanduser().resolve(strict=True))
    manifest_hash = _verify_self_hash(manifest, "manifest_hash")
    _preflight(project, manifest)
    paths = _formal_paths(project, manifest)
    report = _read_json(paths["report"])
    report_hash = _verify_self_hash(report, "report_hash")
    lock = _read_json(paths["decision_lock"])
    if (
        lock.get("state") != "completed"
        or lock.get("qualification_ordinal") != 1
        or lock.get("preregistration_manifest_hash") != manifest_hash
        or lock.get("report_hash") != report_hash
        or lock.get("decision_hash") != report.get("decision_hash")
        or lock.get("oracle_ready_for_frozen_measurement")
        is not report.get("oracle_ready_for_frozen_measurement")
        or lock.get("retry_authorized") is not False
    ):
        raise SC100OracleQualificationError("decision lock does not bind report")
    for row in report.get("rows", ()):  # Receipt uses the same pop-then-hash convention.
        receipt = row.get("receipt") if isinstance(row, Mapping) else None
        if receipt:
            _verify_self_hash(receipt, "receipt_sha256")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_existing:
        report = verify_existing_oracle_qualification(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
        )
    else:
        report = run_oracle_qualification(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
