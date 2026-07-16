from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contract import TASK_CONTRACT, payload_hash
from .train_export import verify_train_view


FORMATION_VERSION = "noaa_gsod_train_only_finite_typed_formation_v1"
DSL_VERSION = "finite_typed_relational_csv_dsl_v1"
OPERATOR_VERSION = "noaa_gsod_typed_relational_operator_v1"
MAX_CANDIDATES = 4096
MAX_SEMANTIC_NODES = 8
FOLD_POLICY = "anonymous_station_modulo_4_v1"
SELECTION_POLICY = (
    "invalid_then_harm_then_negative_exact_recovery_then_program_length_then_hash_v1"
)


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_program_parameters() -> Mapping[str, Any]:
    normalization = TASK_CONTRACT["normalization"]["PRCP"]
    relational = tuple(str(row) for row in TASK_CONTRACT["relational_program"])
    row_scope = str(TASK_CONTRACT["input"]["row_scope"])
    output_mean = str(TASK_CONTRACT["output"]["value"]["mean_daily_precip_mm"])
    if "belongs to 2020" not in row_scope:
        raise ValueError("unsupported frozen year contract")
    if not any("mean_daily_precip_in = sum(PRCP) / count(valid PRCP)" in row for row in relational):
        raise ValueError("unsupported frozen aggregation contract")
    if not any("argmax mean_daily_precip_in" in row for row in relational):
        raise ValueError("unsupported frozen extreme contract")
    if not any("exact tie choose the earliest month" in row for row in relational):
        raise ValueError("unsupported frozen tie contract")
    if not any("multiplying by 25.4" in row for row in relational):
        raise ValueError("unsupported frozen unit contract")
    if not any("ROUND_HALF_UP" in row for row in relational):
        raise ValueError("unsupported frozen rounding contract")
    if "exactly two decimals" not in output_mean:
        raise ValueError("unsupported frozen decimal-place contract")
    return {
        "missing_tokens": tuple(str(row) for row in normalization["missing_tokens"]),
        "year": 2020,
        "aggregation": "mean",
        "extreme": "argmax",
        "tie_break": "earliest",
        "unit_factor": "25.4",
        "rounding": "ROUND_HALF_UP",
        "decimal_places": 2,
    }


_CONTRACT_PARAMETERS = _contract_program_parameters()


@dataclass(frozen=True)
class TypedRelationalProgram:
    missing_tokens: tuple[str, ...]
    year: int
    aggregation: str
    extreme: str
    tie_break: str
    unit_factor: str
    rounding: str
    decimal_places: int
    input_columns: tuple[str, ...] = ("STATION", "DATE", "PRCP")
    output_fields: tuple[str, ...] = (
        "mean_daily_precip_mm",
        "month",
        "valid_day_count",
    )
    dsl_version: str = DSL_VERSION

    @property
    def semantic_nodes(self) -> tuple[Mapping[str, Any], ...]:
        return (
            {"op": "normalize_missing", "column": "PRCP", "tokens": list(self.missing_tokens)},
            {"op": "filter_year", "column": "DATE", "year": self.year},
            {"op": "derive_month", "column": "DATE", "format": "%m"},
            {"op": "group_aggregate", "key": "month", "value": "PRCP", "measures": ["sum", "count", self.aggregation]},
            {"op": self.extreme, "value": self.aggregation, "tie_break": self.tie_break},
            {"op": "unit_convert", "factor": self.unit_factor},
            {"op": "decimal_round", "mode": self.rounding, "places": self.decimal_places},
        )

    @property
    def program_length(self) -> int:
        # The fixed rank's program-length field also expresses distance from
        # the public task contract.  This resolves TRAIN-behavior aliases by a
        # declared semantic source instead of silently choosing an arbitrary
        # hash representative.
        return len(self.semantic_nodes) + self.contract_deviation_count

    @property
    def contract_deviation_count(self) -> int:
        actual = {
            "missing_tokens": self.missing_tokens,
            "year": self.year,
            "aggregation": self.aggregation,
            "extreme": self.extreme,
            "tie_break": self.tie_break,
            "unit_factor": self.unit_factor,
            "rounding": self.rounding,
            "decimal_places": self.decimal_places,
        }
        return sum(
            actual[key] != value for key, value in _CONTRACT_PARAMETERS.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "missing_tokens": list(self.missing_tokens),
            "input_columns": list(self.input_columns),
            "output_fields": list(self.output_fields),
            "semantic_nodes": [dict(row) for row in self.semantic_nodes],
            "fixed_envelope": {
                "read": "RFC4180_csv_dict_rows",
                "serialize": "canonical_json_sorted_keys_compact_no_nan",
            },
        }

    @property
    def program_hash(self) -> str:
        return _hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TypedRelationalProgram":
        return cls(
            missing_tokens=tuple(str(row) for row in value["missing_tokens"]),
            year=int(value["year"]),
            aggregation=str(value["aggregation"]),
            extreme=str(value["extreme"]),
            tie_break=str(value["tie_break"]),
            unit_factor=str(value["unit_factor"]),
            rounding=str(value["rounding"]),
            decimal_places=int(value["decimal_places"]),
            input_columns=tuple(str(row) for row in value.get("input_columns", ("STATION", "DATE", "PRCP"))),
            output_fields=tuple(str(row) for row in value.get("output_fields", ("mean_daily_precip_mm", "month", "valid_day_count"))),
            dsl_version=str(value.get("dsl_version") or DSL_VERSION),
        )

    def type_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.dsl_version != DSL_VERSION:
            issues.append("dsl_version")
        if len(self.semantic_nodes) > MAX_SEMANTIC_NODES:
            issues.append("semantic_node_budget")
        if not self.missing_tokens or any(not isinstance(row, str) for row in self.missing_tokens):
            issues.append("missing_tokens")
        if self.aggregation not in {"sum", "count", "mean"}:
            issues.append("aggregation")
        if self.extreme not in {"argmax", "argmin"}:
            issues.append("extreme")
        if self.tie_break not in {"earliest", "latest"}:
            issues.append("tie_break")
        try:
            factor = Decimal(self.unit_factor)
            if factor <= 0:
                issues.append("unit_factor")
        except InvalidOperation:
            issues.append("unit_factor")
        if self.rounding not in {"ROUND_HALF_UP", "ROUND_HALF_EVEN"}:
            issues.append("rounding")
        if self.decimal_places not in {1, 2}:
            issues.append("decimal_places")
        if self.input_columns != ("STATION", "DATE", "PRCP"):
            issues.append("input_columns")
        if self.output_fields != (
            "mean_daily_precip_mm",
            "month",
            "valid_day_count",
        ):
            issues.append("output_fields")
        return tuple(sorted(set(issues)))


@dataclass(frozen=True)
class CandidateAssessment:
    program: TypedRelationalProgram
    invalid_count: int
    harm_count: int
    exact_recovery_count: int
    behavior_hash: str
    output_hashes: tuple[str, ...]

    @property
    def rank(self) -> tuple[int, int, int, int, str]:
        return (
            self.invalid_count,
            self.harm_count,
            -self.exact_recovery_count,
            self.program.program_length,
            self.program.program_hash,
        )


@dataclass(frozen=True)
class FormationResult:
    status: str
    program: TypedRelationalProgram | None
    receipt: Mapping[str, Any]


def enumerate_programs() -> Iterable[TypedRelationalProgram]:
    dimensions = (
        (("",), ("99.99",), ("", "99.99")),
        (2019, 2020, 2021),
        ("sum", "count", "mean"),
        ("argmax", "argmin"),
        ("earliest", "latest"),
        ("1", "25.4"),
        ("ROUND_HALF_UP", "ROUND_HALF_EVEN"),
        (1, 2),
    )
    for values in itertools.product(*dimensions):
        yield TypedRelationalProgram(*values)


def execute_frozen_operator(
    program: TypedRelationalProgram,
    input_csv: str | Path,
) -> bytes:
    issues = program.type_issues()
    if issues:
        raise ValueError(f"ill-typed relational program: {issues}")
    groups: dict[str, list[Decimal]] = {}
    with Path(input_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != program.input_columns:
            raise ValueError("input schema mismatch")
        for row in reader:
            date_text = str(row["DATE"]).strip()
            try:
                parsed_date = calendar_date.fromisoformat(date_text)
            except ValueError as exc:
                raise ValueError("invalid DATE") from exc
            if parsed_date.year != program.year:
                continue
            month_number = parsed_date.month
            token = str(row["PRCP"]).strip()
            if token in program.missing_tokens:
                continue
            try:
                amount = Decimal(token)
            except InvalidOperation as exc:
                raise ValueError("invalid PRCP") from exc
            if not amount.is_finite() or amount < 0:
                raise ValueError("invalid PRCP")
            groups.setdefault(f"{month_number:02d}", []).append(amount)
    if not groups:
        raise ValueError("no valid grouped rows")

    scored: list[tuple[Decimal, str, int]] = []
    for month, values in groups.items():
        total = sum(values, Decimal("0"))
        count = len(values)
        if program.aggregation == "sum":
            score = total
        elif program.aggregation == "count":
            score = Decimal(count)
        else:
            score = total / Decimal(count)
        scored.append((score, month, count))
    reverse_extreme = program.extreme == "argmax"
    best_value = (max if reverse_extreme else min)(row[0] for row in scored)
    tied = [row for row in scored if row[0] == best_value]
    selected = (min if program.tie_break == "earliest" else max)(
        tied, key=lambda row: row[1]
    )
    selected_value, month, valid_count = selected
    converted = selected_value * Decimal(program.unit_factor)
    quantum = Decimal(1).scaleb(-program.decimal_places)
    rounding = ROUND_HALF_UP if program.rounding == "ROUND_HALF_UP" else ROUND_HALF_EVEN
    formatted = format(converted.quantize(quantum, rounding=rounding), f".{program.decimal_places}f")
    payload = {
        "mean_daily_precip_mm": formatted,
        "month": month,
        "valid_day_count": valid_count,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_oracle(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _assess(
    program: TypedRelationalProgram,
    items: Sequence[Mapping[str, Any]],
    root: Path,
) -> CandidateAssessment:
    invalid = 0
    harm = 0
    exact = 0
    output_hashes: list[str] = []
    for item in items:
        try:
            output = execute_frozen_operator(program, root / str(item["input_relative_path"]))
            output_hashes.append(hashlib.sha256(output).hexdigest())
            if output == _canonical_oracle(item["oracle_consensus"]):
                exact += 1
            else:
                harm += 1
        except (OSError, ValueError, ArithmeticError):
            invalid += 1
            output_hashes.append(_hash({"invalid": True}))
    return CandidateAssessment(
        program=program,
        invalid_count=invalid,
        harm_count=harm,
        exact_recovery_count=exact,
        behavior_hash=_hash({"ordered_output_hashes": output_hashes}),
        output_hashes=tuple(output_hashes),
    )


def _validate_train_view(view: Mapping[str, Any], root: Path) -> tuple[Mapping[str, Any], ...]:
    verified = verify_train_view(view, train_view_root=root)
    if verified.get("task_contract") != TASK_CONTRACT:
        raise ValueError("TRAIN task contract mismatch")
    if verified.get("task_contract_hash") != payload_hash(TASK_CONTRACT):
        raise ValueError("TRAIN task contract hash mismatch")
    return tuple(verified["items"])


def _folds(items: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(index for index in range(len(items)) if index % 4 == fold) for fold in range(4))


def form_typed_relational_candidate(
    train_view_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> FormationResult:
    path = Path(train_view_path)
    root = path.parent.resolve()
    view = json.loads(path.read_text(encoding="utf-8"))
    items = _validate_train_view(view, root)
    programs = tuple(enumerate_programs())
    if len(programs) > MAX_CANDIDATES:
        raise ValueError("finite candidate budget exceeded")
    assessments = tuple(_assess(program, items, root) for program in programs)

    by_behavior: dict[str, CandidateAssessment] = {}
    for row in sorted(assessments, key=lambda candidate: candidate.rank):
        by_behavior.setdefault(row.behavior_hash, row)
    deduplicated = tuple(sorted(by_behavior.values(), key=lambda row: row.rank))
    winner = deduplicated[0]
    winner_aliases = tuple(
        row for row in assessments if row.behavior_hash == winner.behavior_hash
    )
    contract_conformant_exact = tuple(
        row
        for row in assessments
        if row.invalid_count == 0
        and row.harm_count == 0
        and row.exact_recovery_count == len(items)
        and row.program.contract_deviation_count == 0
    )

    fold_receipts: list[dict[str, Any]] = []
    crossfit_program_hashes: list[str] = []
    for validation_indices in _folds(items):
        validation_set = set(validation_indices)
        train_items = tuple(row for index, row in enumerate(items) if index not in validation_set)
        fold_assessments = tuple(_assess(program, train_items, root) for program in programs)
        fold_by_behavior: dict[str, CandidateAssessment] = {}
        for row in sorted(fold_assessments, key=lambda candidate: candidate.rank):
            fold_by_behavior.setdefault(row.behavior_hash, row)
        fold_winner = min(fold_by_behavior.values(), key=lambda row: row.rank)
        held_out = tuple(items[index] for index in validation_indices)
        held_out_assessment = _assess(fold_winner.program, held_out, root)
        crossfit_program_hashes.append(fold_winner.program.program_hash)
        fold_receipts.append(
            {
                "fold_index": len(fold_receipts),
                "train_count": len(train_items),
                "station_out_count": len(held_out),
                "selected_program_hash": fold_winner.program.program_hash,
                "held_out_invalid_count": held_out_assessment.invalid_count,
                "held_out_harm_count": held_out_assessment.harm_count,
                "held_out_exact_recovery_count": held_out_assessment.exact_recovery_count,
                "held_out_item_set_hash": _hash(sorted(str(row["train_item_hash"]) for row in held_out)),
            }
        )

    exact_unique_behavior = (
        winner.invalid_count == 0
        and winner.harm_count == 0
        and winner.exact_recovery_count == len(items)
    )
    crossfit_exact = all(
        row["held_out_invalid_count"] == 0
        and row["held_out_harm_count"] == 0
        and row["held_out_exact_recovery_count"] == row["station_out_count"]
        for row in fold_receipts
    )
    crossfit_stable = len(set(crossfit_program_hashes)) == 1 and crossfit_program_hashes[0] == winner.program.program_hash
    contract_resolved = (
        winner.program.contract_deviation_count == 0
        and len(contract_conformant_exact) == 1
    )
    formed = (
        exact_unique_behavior
        and contract_resolved
        and crossfit_exact
        and crossfit_stable
    )
    status = "formed_unique_exact_crossfit" if formed else "representation_or_identifiability_failure"

    receipt_body: dict[str, Any] = {
        "formation_version": FORMATION_VERSION,
        "study_id": str(view["study_id"]),
        "status": status,
        "offline_contract": {
            "partition": "train",
            "model_calls": 0,
            "network_calls": 0,
            "online_judge_calls": 0,
            "development_or_sealed_accessed": False,
        },
        "source_receipt": {
            "train_view_hash": str(view["train_view_hash"]),
            "task_contract_hash": str(view["task_contract_hash"]),
            "train_item_set_hash": _hash(sorted(str(row["train_item_hash"]) for row in items)),
            "train_item_count": len(items),
            "raw_content_persisted": False,
        },
        "search_receipt": {
            "candidate_count": len(programs),
            "candidate_budget": MAX_CANDIDATES,
            "type_valid_count": sum(not row.program.type_issues() for row in assessments),
            "behavior_unique_count": len(deduplicated),
            "behavior_hash_dedup": True,
            "maximum_semantic_nodes": max(len(row.program.semantic_nodes) for row in assessments),
            "semantic_node_budget": MAX_SEMANTIC_NODES,
            "selection_policy": SELECTION_POLICY,
            "contract_hash": payload_hash(TASK_CONTRACT),
            "contract_deviation_in_program_length": True,
        },
        "crossfit_receipt": {
            "policy": FOLD_POLICY,
            "folds": fold_receipts,
            "all_station_out_exact": crossfit_exact,
            "selected_program_stable": crossfit_stable,
        },
        "selection_receipt": {
            "selected_program_hash": winner.program.program_hash if formed else None,
            "selected_behavior_hash": winner.behavior_hash if formed else None,
            "invalid_count": winner.invalid_count,
            "harm_count": winner.harm_count,
            "exact_recovery_count": winner.exact_recovery_count,
            "program_length": winner.program.program_length,
            "semantic_node_count": len(winner.program.semantic_nodes),
            "contract_deviation_count": winner.program.contract_deviation_count,
            "exact_behavior_alias_class_size": len(winner_aliases),
            "exact_behavior_alias_program_set_hash": _hash(
                sorted(row.program.program_hash for row in winner_aliases)
            ),
            "contract_conformant_exact_candidate_count": len(contract_conformant_exact),
            "contract_derived_resolution": contract_resolved,
            "unique_after_contract_resolution": formed,
        },
        "claim_boundary": {
            "train_only_formation": True,
            "performance_claim": False,
            "development_run_authorized": False,
            "sealed_run_authorized": False,
        },
        "raw_content_persisted": False,
    }
    receipt = {**receipt_body, "receipt_hash": _hash(receipt_body)}
    selected = winner.program if formed else None
    result = FormationResult(status=status, program=selected, receipt=receipt)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "formation.receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if selected is not None:
            program_payload = selected.to_dict()
            envelope = {
                "operator_version": OPERATOR_VERSION,
                "program": program_payload,
                "program_hash": selected.program_hash,
                "formation_receipt_hash": receipt["receipt_hash"],
                "raw_content_persisted": False,
            }
            (destination / "frozen_program.json").write_text(
                json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    return result


def load_formation_receipt(path: str | Path) -> Mapping[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    declared = receipt.pop("receipt_hash", None)
    if declared != _hash(receipt):
        raise ValueError("formation receipt hash mismatch")
    if receipt.get("raw_content_persisted") is not False:
        raise ValueError("unsafe formation receipt")
    return {**receipt, "receipt_hash": declared}


def load_frozen_program(
    path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> TypedRelationalProgram:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_envelope_keys = {
        "operator_version",
        "program",
        "program_hash",
        "formation_receipt_hash",
        "raw_content_persisted",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected_envelope_keys:
        raise ValueError("frozen program envelope schema mismatch")
    if envelope.get("operator_version") != OPERATOR_VERSION:
        raise ValueError("frozen program operator version mismatch")
    if envelope.get("raw_content_persisted") is not False:
        raise ValueError("unsafe frozen program")
    payload = envelope.get("program")
    if not isinstance(payload, Mapping):
        raise ValueError("frozen program payload is malformed")
    try:
        program = TypedRelationalProgram.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen program payload is malformed") from exc
    if dict(payload) != program.to_dict():
        raise ValueError("frozen program canonical payload mismatch")
    if envelope.get("program_hash") != program.program_hash:
        raise ValueError("frozen program hash mismatch")
    formation_hash = envelope.get("formation_receipt_hash")
    if (
        not isinstance(formation_hash, str)
        or len(formation_hash) != 64
        or any(character not in "0123456789abcdef" for character in formation_hash)
    ):
        raise ValueError("formation receipt binding is malformed")
    if receipt_path is not None:
        receipt = load_formation_receipt(receipt_path)
        if receipt.get("receipt_hash") != formation_hash:
            raise ValueError("formation receipt binding mismatch")
        selection = receipt.get("selection_receipt")
        if (
            not isinstance(selection, Mapping)
            or selection.get("selected_program_hash") != program.program_hash
        ):
            raise ValueError("formation receipt selected program mismatch")
    return program
