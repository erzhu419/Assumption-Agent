from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from ..models import stable_hash
from ..splits import SplitManifest
from ..typed_operator_grammar import (
    freeze_typed_recipe_selection_snapshot,
    freeze_typed_selection_snapshot_ledger,
)
from .typed_portable_integration import (
    FrozenTypedSelectionLedger,
    _project_portable_graph,
    _projection_row,
    _verify_historical_feasibility,
    build_implementation_file_binding,
    reconstruct_current_full_graph_material,
)


HISTORICAL_TYPED_SELECTION_PROJECTION_VERSION = (
    "historical_portable_typed_selection_projection_v2"
)
HISTORICAL_TYPED_INTEGRATION_COMMIT = (
    "a66ecaf2c07102cb86f3e95f8a1017b9658be287"
)
HISTORICAL_TYPED_SOURCE_RUN_COMMIT = (
    "ffdf8064ae9cd3e03b079ff6de001f3923b6bdeb"
)
HISTORICAL_TYPED_PREREGISTRATION_SHA256 = (
    "6656ae3e31e9d852b26e23820776d7e515f0c23d4d61522323c384f38b0a61d9"
)
HISTORICAL_TYPED_PREREGISTRATION_HASH = (
    "f6b240f78c27951e14f82b08b6226abc47e995edf0089f5ec4dee5d69c16117c"
)
HISTORICAL_TYPED_RESULT_RECEIPT_SHA256 = (
    "613ad0dfcd9c06b6a47722f91c2caf2532e74d9e40b4e3577ed0c8db42f778d0"
)
HISTORICAL_TYPED_RESULT_RECEIPT_HASH = (
    "afb04a536cfaee93f5549b0875dc40b1e255e4a0fbaed14561d34c7b09db34cf"
)
HISTORICAL_TYPED_REPORT_SHA256 = (
    "f2cb0921fb7a01afff9086729b9396a08e12d338bcde9b86350933a8dc05556b"
)
HISTORICAL_TYPED_EVENTS_SHA256 = (
    "86f6954298aa09ce45819bee5df8ef2a681b917185005fc409b65503ee5fa4fd"
)
HISTORICAL_TYPED_DECISION_LOCK_SHA256 = (
    "348dd9e675c1717ac6a81a941f84590eff1a66e8013670295bcb52ae42d6c6cf"
)
HISTORICAL_TYPED_SOURCE_PROTOCOL_SHA256 = (
    "fe982586f8de7beb10928e4187d2f0b347f89b2c780540a9bb7ff2de4618d929"
)
HISTORICAL_PROJECTED_LEDGER_HASH = (
    "d560903a5df0da0a464b3636ef2f80bd86cba3f5230de53f5da6f3acc4597bbf"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HistoricalTypedSelectionProjectionError(PermissionError):
    """A historical typed selection source failed compatibility projection."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalTypedSelectionProjectionError(
            f"{label} is not a regular file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalTypedSelectionProjectionError(
            f"{label} is not readable JSON"
        ) from exc
    if not isinstance(value, dict):
        raise HistoricalTypedSelectionProjectionError(
            f"{label} is not an object"
        )
    return value


def _project_file(project_root: Path, relative: str) -> Path:
    candidate = project_root.joinpath(*Path(relative).parts)
    if candidate.is_symlink():
        raise HistoricalTypedSelectionProjectionError(
            "historical typed artifact is a symlink"
        )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(project_root)
    except (FileNotFoundError, ValueError) as exc:
        raise HistoricalTypedSelectionProjectionError(
            "historical typed artifact escaped the project root"
        ) from exc
    return resolved


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not _SHA256.fullmatch(expected) or _sha256_file(path) != expected:
        raise HistoricalTypedSelectionProjectionError(
            f"historical {label} file hash drifted"
        )


def _verify_legacy_implementation_blobs(
    *,
    project_root: Path,
    preregistration: Mapping[str, Any],
) -> str:
    rows = preregistration.get("implementation_files")
    if not isinstance(rows, list) or not rows:
        raise HistoricalTypedSelectionProjectionError(
            "historical implementation ledger is missing"
        )
    try:
        repo_root = Path(
            subprocess.run(
                ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        ).resolve(strict=True)
        project_prefix = project_root.relative_to(repo_root).as_posix()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise HistoricalTypedSelectionProjectionError(
            "historical implementation git authority is unavailable"
        ) from exc
    canonical_rows: list[dict[str, str]] = []
    for raw in rows:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"path", "sha256"}
            or not isinstance(raw.get("path"), str)
            or not _SHA256.fullmatch(str(raw.get("sha256") or ""))
        ):
            raise HistoricalTypedSelectionProjectionError(
                "historical implementation ledger row is malformed"
            )
        relative = Path(raw["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise HistoricalTypedSelectionProjectionError(
                "historical implementation path escaped the project"
            )
        object_path = (
            f"{HISTORICAL_TYPED_INTEGRATION_COMMIT}:"
            f"{project_prefix}/{relative.as_posix()}"
        )
        try:
            content = subprocess.run(
                ["git", "-C", str(repo_root), "show", object_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise HistoricalTypedSelectionProjectionError(
                "historical implementation blob is unavailable"
            ) from exc
        if _sha256_bytes(content) != raw["sha256"]:
            raise HistoricalTypedSelectionProjectionError(
                "historical implementation blob hash drifted"
            )
        canonical_rows.append(
            {"path": relative.as_posix(), "sha256": raw["sha256"]}
        )
    if canonical_rows != sorted(canonical_rows, key=lambda row: row["path"]):
        raise HistoricalTypedSelectionProjectionError(
            "historical implementation ledger is not canonical"
        )
    implementation_set_hash = stable_hash({"files": canonical_rows})
    if implementation_set_hash != preregistration.get(
        "expected_implementation_file_set_hash"
    ):
        raise HistoricalTypedSelectionProjectionError(
            "historical implementation set hash drifted"
        )
    return implementation_set_hash


@dataclass(frozen=True)
class HistoricalTypedSelectionSourceReceiptV2:
    ledger: FrozenTypedSelectionLedger = field(compare=False, repr=False)
    legacy_implementation_set_hash: str
    current_implementation_set_hash: str
    compatibility_adapter_file_sha256: str
    historical_binding_hash: str
    source_train_receipt_hash: str
    trial_evidence_hash: str
    full_graph_set_hash: str
    projected_graph_set_hash: str
    projected_model_catalog_set_hash: str
    projected_snapshot_ledger_hash: str

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "projection_policy": HISTORICAL_TYPED_SELECTION_PROJECTION_VERSION,
            "historical_integration_commit": (
                HISTORICAL_TYPED_INTEGRATION_COMMIT
            ),
            "historical_source_run_commit": HISTORICAL_TYPED_SOURCE_RUN_COMMIT,
            "historical_preregistration_file_sha256": (
                HISTORICAL_TYPED_PREREGISTRATION_SHA256
            ),
            "historical_preregistration_hash": (
                HISTORICAL_TYPED_PREREGISTRATION_HASH
            ),
            "historical_result_receipt_file_sha256": (
                HISTORICAL_TYPED_RESULT_RECEIPT_SHA256
            ),
            "historical_result_receipt_hash": (
                HISTORICAL_TYPED_RESULT_RECEIPT_HASH
            ),
            "historical_report_file_sha256": HISTORICAL_TYPED_REPORT_SHA256,
            "historical_events_file_sha256": HISTORICAL_TYPED_EVENTS_SHA256,
            "historical_decision_lock_file_sha256": (
                HISTORICAL_TYPED_DECISION_LOCK_SHA256
            ),
            "historical_source_protocol_file_sha256": (
                HISTORICAL_TYPED_SOURCE_PROTOCOL_SHA256
            ),
            "legacy_implementation_set_hash": (
                self.legacy_implementation_set_hash
            ),
            "current_implementation_set_hash": (
                self.current_implementation_set_hash
            ),
            "compatibility_adapter_file_sha256": (
                self.compatibility_adapter_file_sha256
            ),
            "historical_binding_hash": self.historical_binding_hash,
            "source_train_receipt_hash": self.source_train_receipt_hash,
            "trial_evidence_hash": self.trial_evidence_hash,
            "full_graph_set_hash": self.full_graph_set_hash,
            "projected_graph_set_hash": self.projected_graph_set_hash,
            "projected_model_catalog_set_hash": (
                self.projected_model_catalog_set_hash
            ),
            "projected_snapshot_ledger_hash": (
                self.projected_snapshot_ledger_hash
            ),
            "snapshot_count": len(self.ledger.snapshots),
            "train_observation_count": len(self.ledger.evidence.residuals),
            "historical_files_verified": True,
            "historical_git_blobs_verified": True,
            "current_graphs_reconstructed": True,
            "projection_rows_exact": True,
            "legacy_ledger_hash_reproduced": True,
            "diagnostic_only": True,
            "freeze_authority_granted": False,
            "development_execution_authorized": False,
            "model_calls": 0,
            "task_backend_calls": 0,
            "evaluator_calls": 0,
            "validation_accessed": False,
            "test_accessed": False,
            "raw_content_persisted": False,
            "secret_value_persisted": False,
        }

    def verify(self) -> None:
        hash_values = (
            self.legacy_implementation_set_hash,
            self.current_implementation_set_hash,
            self.compatibility_adapter_file_sha256,
            self.historical_binding_hash,
            self.source_train_receipt_hash,
            self.trial_evidence_hash,
            self.full_graph_set_hash,
            self.projected_graph_set_hash,
            self.projected_model_catalog_set_hash,
            self.projected_snapshot_ledger_hash,
        )
        if (
            any(not _SHA256.fullmatch(value) for value in hash_values)
            or self.projected_snapshot_ledger_hash
            != HISTORICAL_PROJECTED_LEDGER_HASH
            or self.ledger.production_snapshot_ledger.ledger_hash
            != self.projected_snapshot_ledger_hash
            or self.ledger.graph_set_hash != self.projected_graph_set_hash
            or self.ledger.model_catalog_set_hash
            != self.projected_model_catalog_set_hash
            or self.ledger.trial_evidence_hash != self.trial_evidence_hash
            or self.ledger.evidence.source_train_receipt_hash
            != self.source_train_receipt_hash
            or self.ledger.freeze_authorization is not None
            or len(self.ledger.snapshots) != 3
            or self.ledger.production_snapshot_ledger.validate()
            or any(snapshot.validate() for snapshot in self.ledger.snapshots)
        ):
            raise HistoricalTypedSelectionProjectionError(
                "historical typed selection source receipt drifted"
            )


def load_historical_portable_typed_selection_projection_v2(
    *,
    project_root: Path,
) -> HistoricalTypedSelectionSourceReceiptV2:
    """Reconstruct the v3.15 projected ledger under current source.

    The legacy integration's current-file hash is intentionally not reused as
    execution authority.  Its exact bytes and historical git blobs are audited,
    the TRAIN graphs are reconstructed locally, and the old projected ledger
    hash must reproduce exactly.  The returned ledger remains diagnostic-only.
    """

    root = project_root.resolve(strict=True)
    preregistration_path = _project_file(
        root,
        "manifests/skilllearn_typed_portable_integration_v1.json",
    )
    result_receipt_path = _project_file(
        root,
        "manifests/skilllearn_typed_portable_integration_result_v1.json",
    )
    report_path = _project_file(
        root,
        "artifacts/typed_portable_integration_v1_v315_train/"
        "typed_portable_integration.report.json",
    )
    events_path = _project_file(
        root,
        "artifacts/typed_portable_integration_v1_v315_train/"
        "typed_portable_integration.events.jsonl",
    )
    decision_lock_path = _project_file(
        root,
        "artifacts/typed_portable_integration_v1_v315_train/"
        "typed_portable_integration.decision.lock.json",
    )
    source_protocol_path = _project_file(
        root,
        "artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_"
        "outer6_model1_actiondelta01/protocol_lock.json",
    )
    for path, expected, label in (
        (
            preregistration_path,
            HISTORICAL_TYPED_PREREGISTRATION_SHA256,
            "typed preregistration",
        ),
        (
            result_receipt_path,
            HISTORICAL_TYPED_RESULT_RECEIPT_SHA256,
            "typed result receipt",
        ),
        (report_path, HISTORICAL_TYPED_REPORT_SHA256, "typed report"),
        (events_path, HISTORICAL_TYPED_EVENTS_SHA256, "typed events"),
        (
            decision_lock_path,
            HISTORICAL_TYPED_DECISION_LOCK_SHA256,
            "typed decision lock",
        ),
        (
            source_protocol_path,
            HISTORICAL_TYPED_SOURCE_PROTOCOL_SHA256,
            "typed source protocol",
        ),
    ):
        _verify_hash(path, expected, label)

    preregistration = _read_json(
        preregistration_path,
        "historical typed preregistration",
    )
    result_receipt = _read_json(
        result_receipt_path,
        "historical typed result receipt",
    )
    report = _read_json(report_path, "historical typed report")
    decision_lock = _read_json(
        decision_lock_path,
        "historical typed decision lock",
    )
    source_protocol = _read_json(
        source_protocol_path,
        "historical typed source protocol",
    )
    report_without_hash = dict(report)
    report_hash = report_without_hash.pop("report_hash", None)
    source_protocol_without_hash = dict(source_protocol)
    source_protocol_lock_hash = source_protocol_without_hash.pop(
        "lock_hash",
        None,
    )
    canonical_artifacts = result_receipt.get("canonical_artifacts")
    projection = preregistration.get("portable_projection")
    result_projection = result_receipt.get("portable_projection")
    if (
        stable_hash(preregistration)
        != HISTORICAL_TYPED_PREREGISTRATION_HASH
        or stable_hash(result_receipt) != HISTORICAL_TYPED_RESULT_RECEIPT_HASH
        or not isinstance(canonical_artifacts, dict)
        or canonical_artifacts.get("report", {}).get("sha256")
        != HISTORICAL_TYPED_REPORT_SHA256
        or canonical_artifacts.get("events", {}).get("sha256")
        != HISTORICAL_TYPED_EVENTS_SHA256
        or canonical_artifacts.get("decision_lock", {}).get("sha256")
        != HISTORICAL_TYPED_DECISION_LOCK_SHA256
        or stable_hash(report_without_hash) != report_hash
        or result_receipt.get("report_hash") != report_hash
        or result_receipt.get("integration_passed") is not True
        or result_receipt.get("exact_replay_verified") is not True
        or decision_lock.get("state") != "completed"
        or decision_lock.get("report_hash") != report_hash
        or decision_lock.get("decision_hash")
        != result_receipt.get("decision_hash")
        or not isinstance(projection, dict)
        or not isinstance(result_projection, dict)
        or projection.get("ledger_hash") != HISTORICAL_PROJECTED_LEDGER_HASH
        or result_projection.get("snapshot_ledger_hash")
        != HISTORICAL_PROJECTED_LEDGER_HASH
        or source_protocol_lock_hash
        != stable_hash(source_protocol_without_hash)
        or source_protocol.get("git", {}).get("commit")
        != HISTORICAL_TYPED_SOURCE_RUN_COMMIT
        or source_protocol.get("git", {}).get("scoped_dirty") is not False
    ):
        raise HistoricalTypedSelectionProjectionError(
            "historical typed authority drifted"
        )

    legacy_implementation_set_hash = _verify_legacy_implementation_blobs(
        project_root=root,
        preregistration=preregistration,
    )
    historical = _verify_historical_feasibility(
        preregistration,
        preregistration_path=preregistration_path,
    )
    commitments = dict(preregistration["full_graph_commitments"])
    commitments.update(
        {
            "manifest_hash": preregistration["manifest_hash"],
            "source_train_receipt_hash": preregistration[
                "source_train_receipt_hash"
            ],
        }
    )
    material = reconstruct_current_full_graph_material(
        root=_project_file(root, preregistration["benchmark_root"]),
        manifest_path=_project_file(root, preregistration["manifest"]),
        source_run_root=(root / preregistration["source_run_root"]).resolve(
            strict=True
        ),
        source_train_receipt=_project_file(
            root,
            preregistration["source_train_receipt"],
        ),
        commitments=commitments,
    )
    projected_graphs = tuple(
        _project_portable_graph(graph) for graph in material.full_graphs
    )
    projection_rows = tuple(
        _projection_row(full, projected)
        for full, projected in zip(
            material.full_graphs,
            projected_graphs,
            strict=True,
        )
    )
    if list(projection_rows) != projection.get("rows"):
        raise HistoricalTypedSelectionProjectionError(
            "current portable graph projection drifted"
        )
    snapshots = tuple(
        freeze_typed_recipe_selection_snapshot(graph)
        for graph in projected_graphs
    )
    graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": snapshot.graph.target_family_hash,
                    "graph_hash": snapshot.expected_graph_hash,
                    "availability_error_hash": None,
                }
                for snapshot in snapshots
            ]
        }
    )
    model_catalog_set_hash = stable_hash(
        {
            "catalog_hashes": [
                snapshot.expected_model_catalog_hash
                for snapshot in snapshots
            ]
        }
    )
    manifest = SplitManifest.read(
        _project_file(root, preregistration["manifest"])
    )
    production_ledger = freeze_typed_selection_snapshot_ledger(
        snapshots,
        feasibility_preregistration_hash=historical.preregistration_hash,
        feasibility_result_receipt_sha256=(
            historical.result_receipt_file_sha256
        ),
        feasibility_decision_hash=historical.decision_hash,
        feasibility_report_hash=historical.report_hash,
        manifest_hash=manifest.manifest_hash,
        source_train_receipt_hash=material.evidence.source_train_receipt_hash,
        expected_graph_set_hash=graph_set_hash,
        expected_model_catalog_set_hash=model_catalog_set_hash,
        expected_target_family_hashes=tuple(
            graph.target_family_hash for graph in projected_graphs
        ),
    )
    if production_ledger.ledger_hash != HISTORICAL_PROJECTED_LEDGER_HASH:
        raise HistoricalTypedSelectionProjectionError(
            "historical projected ledger hash did not reproduce"
        )
    frozen = FrozenTypedSelectionLedger(
        evidence=material.evidence,
        trials=material.trials,
        snapshots=snapshots,
        production_snapshot_ledger=production_ledger,
        upstream_binding_hash=historical.binding_hash,
        trial_evidence_hash=material.trial_evidence_hash,
        graph_set_hash=graph_set_hash,
        model_catalog_set_hash=model_catalog_set_hash,
    )

    current_binding = build_implementation_file_binding(
        preregistration_path
    )
    current_rows = list(current_binding["implementation_files"])
    adapter_relative = Path(__file__).resolve().relative_to(root).as_posix()
    adapter_sha256 = _sha256_file(Path(__file__).resolve())
    current_rows.append(
        {"path": adapter_relative, "sha256": adapter_sha256}
    )
    current_rows.sort(key=lambda row: row["path"])
    receipt = HistoricalTypedSelectionSourceReceiptV2(
        ledger=frozen,
        legacy_implementation_set_hash=legacy_implementation_set_hash,
        current_implementation_set_hash=stable_hash(
            {"files": current_rows}
        ),
        compatibility_adapter_file_sha256=adapter_sha256,
        historical_binding_hash=historical.binding_hash,
        source_train_receipt_hash=material.evidence.source_train_receipt_hash,
        trial_evidence_hash=material.trial_evidence_hash,
        full_graph_set_hash=material.full_graph_set_hash,
        projected_graph_set_hash=graph_set_hash,
        projected_model_catalog_set_hash=model_catalog_set_hash,
        projected_snapshot_ledger_hash=production_ledger.ledger_hash,
    )
    receipt.verify()
    return receipt
