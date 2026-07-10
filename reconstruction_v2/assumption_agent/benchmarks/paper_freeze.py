from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import HypothesisProgram, HypothesisStatus, SplitName, stable_hash
from ..splits import SplitManifest
from .paper_controls import ControlSource, control_config_hash, source_tree_hash
from .paper_protocol import PaperProtocol, _code_fingerprint, _git_state
from .skilllearn_compiler import SkillLearnProgramCompiler
from .skilllearnbench import SkillLearnBenchAdapter


@dataclass(frozen=True)
class FrozenArchive:
    archive_hash: str
    incumbent_id: str | None
    evaluator_epoch: str
    active_programs: tuple[HypothesisProgram, ...]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_hash": self.archive_hash,
            "incumbent_id": self.incumbent_id,
            "evaluator_epoch": self.evaluator_epoch,
            "active_hypothesis_ids": [row.id for row in self.active_programs],
            "active_program_hashes": [row.payload_hash for row in self.active_programs],
            "content_hash": self.content_hash,
        }


def freeze_paper_workspace(
    *,
    protocol: PaperProtocol,
    protocol_lock: Mapping[str, Any],
    manifest: SplitManifest,
    benchmark_root: str | Path,
    project_root: str | Path,
    recursive_report_path: str | Path,
    recursive_archive_path: str | Path,
    no_recursive_report_path: str | Path,
    no_recursive_archive_path: str | Path,
    controls_output_root: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    benchmark = Path(benchmark_root).resolve()
    _validate_protocol_lock(protocol, protocol_lock, manifest, project)
    recursive_report = _read_mapping(recursive_report_path, "recursive development report")
    no_recursive_report = _read_mapping(
        no_recursive_report_path,
        "no-recursive development report",
    )
    evaluator_epoch = f"skilllearn-eval-{manifest.manifest_hash[:12]}"
    _validate_development_report(
        recursive_report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=True,
    )
    _validate_development_report(
        no_recursive_report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=False,
    )
    recursive_archive = read_frozen_archive(
        recursive_archive_path,
        expected_evaluator_epoch=evaluator_epoch,
        expected_report=recursive_report,
    )
    no_recursive_archive = read_frozen_archive(
        no_recursive_archive_path,
        expected_evaluator_epoch=evaluator_epoch,
        expected_report=no_recursive_report,
    )
    controls_root = Path(controls_output_root).resolve()
    if controls_root.exists():
        raise FileExistsError("paper control output must not already exist")
    controls_root.mkdir(parents=True)
    adapter = SkillLearnBenchAdapter(benchmark)
    items = adapter.discover()
    static_program = HypothesisProgram.from_dict(
        _read_mapping(project / "baselines" / "static_generic_program.json", "static program")
    )
    if static_program.validate() or static_program.status is not HypothesisStatus.PROMOTED:
        raise ValueError("static paper control is not a valid promoted program")
    archives = {
        "promoted_v2": recursive_archive,
        "v2_no_recursive_repair": no_recursive_archive,
    }
    control_sets: dict[str, Any] = {}
    for split in (SplitName.VALIDATION, SplitName.TEST):
        controls = _compile_control_set(
            protocol=protocol,
            manifest=manifest,
            items=items,
            project_root=project,
            output_root=controls_root / split.value,
            split=split,
            static_program=static_program,
            archives=archives,
        )
        control_sets[split.value] = {
            "controls": [
                {
                    "id": row.id,
                    "root": str(row.root) if row.root else None,
                    "source_hash": source_tree_hash(row.root) if row.root else None,
                }
                for row in sorted(controls, key=lambda value: value.id)
            ],
            "config_hash": control_config_hash(controls),
            "target_item_set_hash": stable_hash(
                {"item_ids": sorted(manifest.ids_for(split))}
            ),
        }
    receipt = {
        "receipt_version": "paper_freeze_receipt_v1",
        "frozen": True,
        "protocol_id": protocol.id,
        "protocol_hash": protocol.protocol_hash,
        "protocol_lock_hash": protocol_lock.get("lock_hash"),
        "manifest_hash": manifest.manifest_hash,
        "manifest_role": _manifest_role(protocol_lock, manifest),
        "evaluator_epoch": evaluator_epoch,
        "recursive_archive": recursive_archive.to_dict(),
        "no_recursive_archive": no_recursive_archive.to_dict(),
        "recursive_report_hash": _file_content_hash(recursive_report_path),
        "no_recursive_report_hash": _file_content_hash(no_recursive_report_path),
        "control_sets": control_sets,
        "code_fingerprint": protocol_lock.get("code_fingerprint"),
        "git_commit": dict(protocol_lock.get("git") or {}).get("commit"),
        "selected_candidate_available": bool(recursive_archive.active_programs),
        "test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt


def read_frozen_archive(
    path: str | Path,
    *,
    expected_evaluator_epoch: str,
    expected_report: Mapping[str, Any],
) -> FrozenArchive:
    source = Path(path)
    payload = _read_mapping(source, "policy archive")
    hypotheses_payload = payload.get("hypotheses")
    nodes_payload = payload.get("nodes")
    scores_payload = payload.get("score_records")
    if not isinstance(hypotheses_payload, Mapping):
        raise ValueError("archive hypotheses are malformed")
    if not isinstance(nodes_payload, Mapping):
        raise ValueError("archive nodes are malformed")
    if not isinstance(scores_payload, Mapping):
        raise ValueError("archive score records are malformed")
    hypotheses: dict[str, HypothesisProgram] = {}
    for key, row in hypotheses_payload.items():
        if not isinstance(row, Mapping):
            raise ValueError("archive hypothesis row is malformed")
        program = HypothesisProgram.from_dict(row)
        if program.id != key or program.validate():
            raise ValueError("archive hypothesis identity or contract is invalid")
        if program.evaluator_epoch != expected_evaluator_epoch:
            raise ValueError("archive hypothesis evaluator epoch mismatch")
        hypotheses[str(key)] = program
    for key, row in nodes_payload.items():
        if not isinstance(row, Mapping) or row.get("id") != key:
            raise ValueError("archive node identity is invalid")
        active = {str(value) for value in row.get("active_hypothesis_ids", [])}
        if not active <= set(hypotheses):
            raise ValueError("archive node references an unknown hypothesis")
        if row.get("evaluator_epoch_id") != expected_evaluator_epoch:
            raise ValueError("archive node evaluator epoch mismatch")
    incumbent_id = str(payload["incumbent_id"]) if payload.get("incumbent_id") else None
    active_programs: tuple[HypothesisProgram, ...] = ()
    incumbent_rows = [
        str(key) for key, row in nodes_payload.items()
        if isinstance(row, Mapping) and row.get("status") == "incumbent"
    ]
    if incumbent_id is None:
        if incumbent_rows:
            raise ValueError("archive has an incumbent node but no incumbent ID")
    else:
        incumbent = nodes_payload.get(incumbent_id)
        if not isinstance(incumbent, Mapping) or incumbent.get("status") != "incumbent":
            raise ValueError("archive incumbent node is missing or not incumbent")
        if incumbent_rows != [incumbent_id]:
            raise ValueError("archive must have exactly one declared incumbent")
        active_programs = tuple(
            hypotheses[str(hypothesis_id)]
            for hypothesis_id in incumbent.get("active_hypothesis_ids", [])
        )
        if any(row.status is not HypothesisStatus.PROMOTED for row in active_programs):
            raise ValueError("archive incumbent contains a non-promoted hypothesis")
    calculated_hash = stable_hash(
        {
            "hypotheses": {
                key: value.payload_hash for key, value in sorted(hypotheses.items())
            },
            "nodes": {
                str(key): stable_hash(dict(value))
                for key, value in sorted(nodes_payload.items())
                if isinstance(value, Mapping)
            },
            "scores": {
                str(key): dict(value)
                for key, value in sorted(scores_payload.items())
                if isinstance(value, Mapping)
            },
            "incumbent_id": incumbent_id,
        }
    )
    if calculated_hash != payload.get("archive_hash"):
        raise ValueError("archive content hash mismatch")
    if expected_report.get("archive_hash") != calculated_hash:
        raise ValueError("development report and archive hash differ")
    generation = expected_report.get("generation")
    generations = expected_report.get("generations")
    generation_rows = (
        [row for row in generations if isinstance(row, Mapping)]
        if isinstance(generations, list)
        else ([generation] if isinstance(generation, Mapping) else [])
    )
    any_promoted = any(bool(row.get("promoted")) for row in generation_rows)
    if any_promoted != bool(active_programs):
        raise ValueError("development promotion history and archive incumbent differ")
    return FrozenArchive(
        archive_hash=calculated_hash,
        incumbent_id=incumbent_id,
        evaluator_epoch=expected_evaluator_epoch,
        active_programs=active_programs,
        content_hash=_file_content_hash(source),
    )


def _compile_control_set(
    *,
    protocol: PaperProtocol,
    manifest: SplitManifest,
    items: Sequence[Any],
    project_root: Path,
    output_root: Path,
    split: SplitName,
    static_program: HypothesisProgram,
    archives: Mapping[str, FrozenArchive],
) -> tuple[ControlSource, ...]:
    compiler = SkillLearnProgramCompiler()
    controls: list[ControlSource] = []
    for control in protocol.payload["controls"]:
        control_id = str(control["id"])
        source = str(control["source"])
        if source == "none":
            controls.append(ControlSource(control_id, None))
            continue
        if source == "baselines/static_generic_program.json":
            programs = (static_program,)
        elif source == "frozen_archive_incumbent":
            programs = archives["promoted_v2"].active_programs
        elif source == "no_recursive_archive_incumbent":
            programs = archives["v2_no_recursive_repair"].active_programs
        else:
            root = (project_root / source).resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"external control source is missing: {control_id}")
            controls.append(ControlSource(control_id, root))
            continue
        result = compiler.compile(
            programs=programs,
            items=items,
            split_manifest=manifest,
            output_root=output_root,
            method_name=control_id,
            allowed_statuses={HypothesisStatus.PROMOTED},
            target_item_ids=manifest.ids_for(split),
            target_split=split.value,
            trace_id=f"paper-freeze:{split.value}:{control_id}",
        )
        controls.append(ControlSource(control_id, result.output_root.resolve()))
    return tuple(controls)


def _validate_protocol_lock(
    protocol: PaperProtocol,
    lock: Mapping[str, Any],
    manifest: SplitManifest,
    project_root: Path,
) -> None:
    if lock.get("claim_eligible") is not True:
        raise PermissionError("paper freeze requires a claim-eligible protocol lock")
    if lock.get("protocol_hash") != protocol.protocol_hash:
        raise PermissionError("paper freeze protocol lock mismatch")
    if manifest.manifest_hash not in {
        lock.get("primary_manifest_hash"),
        lock.get("secondary_manifest_hash"),
    }:
        raise PermissionError("paper freeze manifest mismatch")
    if lock.get("code_fingerprint") != _code_fingerprint(project_root):
        raise PermissionError("paper code changed after protocol lock")
    git_state = _git_state(project_root)
    locked_git = dict(lock.get("git") or {})
    if git_state.get("scoped_dirty") or git_state.get("commit") != locked_git.get("commit"):
        raise PermissionError("paper source tree changed after protocol lock")


def _validate_development_report(
    report: Mapping[str, Any],
    *,
    protocol: PaperProtocol,
    manifest: SplitManifest,
    recursive_validation_enabled: bool,
) -> None:
    if report.get("mode") != "execute" or report.get("executed") is not True:
        raise ValueError("paper freeze requires an executed development report")
    if report.get("test_content_accessed") is not False:
        raise PermissionError("development report accessed sealed test content")
    preflight = report.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("blockers"):
        raise ValueError("development report has preflight blockers")
    plan = report.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("development report plan is missing")
    secondary = manifest.protocol == "family_out"
    phase_name = "family_out_development" if secondary else "development"
    development = protocol.payload["phases"][phase_name]
    expected = {
        "manifest_hash": manifest.manifest_hash,
        "train_count": int(development["train_count"]),
        "validation_count": int(development["validation_count"]),
        "model": protocol.payload["model"],
        "trial_provider_mode": protocol.payload["trial_provider_mode"],
        "max_steps": int(protocol.payload["max_steps"]),
        "recursive_validation_enabled": recursive_validation_enabled,
        "max_generations": int(protocol.payload["evolution"]["max_generations"]),
        "max_consecutive_non_promotions": int(
            protocol.payload["evolution"]["max_consecutive_non_promotions"]
        ),
        "proposal_candidates_per_generation": int(
            protocol.payload["evolution"]["proposal_candidates_per_generation"]
        ),
        "test_content_accessed": False,
    }
    registry_isolation = protocol.payload["execution"].get(
        "runner_agent_registry_isolation"
    )
    if registry_isolation:
        expected["runner_agent_registry_isolation"] = registry_isolation
    prewarm_version = protocol.payload["execution"].get("development_prewarm")
    if prewarm_version:
        expected["development_prewarm_version"] = prewarm_version
        expected["prewarm_passed"] = True
    timeout_policy = protocol.payload["execution"].get("trial_timeout_policy")
    if timeout_policy:
        expected["trial_timeout_policy"] = timeout_policy
    for field in (
        "provider_failure_policy",
        "ephemeral_auth_cleanup",
        "training_evidence_policy",
    ):
        value = protocol.payload["execution"].get(field)
        if value:
            expected[field] = value
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"development report plan mismatch: {key}")
    if prewarm_version and not str(plan.get("prewarm_receipt_hash") or ""):
        raise ValueError("development report has no prewarm receipt provenance")
    generation = report.get("generation")
    generations = report.get("generations")
    if not isinstance(generation, Mapping) or not isinstance(generations, list) or not generations:
        raise ValueError("development generation summary is missing")
    if int(report.get("generation_count") or 0) != len(generations):
        raise ValueError("development generation count mismatch")
    if len(generations) > int(protocol.payload["evolution"]["max_generations"]):
        raise ValueError("development exceeded the frozen generation budget")
    if not recursive_validation_enabled and any(
        int(row.get("recursive_depth") or 0) != 0
        for row in generations
        if isinstance(row, Mapping)
    ):
        raise ValueError("no-recursive control unexpectedly used recursive repair")


def _manifest_role(lock: Mapping[str, Any], manifest: SplitManifest) -> str:
    if manifest.manifest_hash == lock.get("primary_manifest_hash"):
        return "primary_instance_holdout"
    if manifest.manifest_hash == lock.get("secondary_manifest_hash"):
        return "secondary_family_out"
    raise PermissionError("manifest is not part of the frozen paper protocol")


def _read_mapping(path: str | Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _file_content_hash(path: str | Path) -> str:
    return stable_hash({"bytes": Path(path).read_bytes().hex()})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze two development archives and compile immutable paper controls."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--recursive-report", type=Path, required=True)
    parser.add_argument("--recursive-archive", type=Path, required=True)
    parser.add_argument("--no-recursive-report", type=Path, required=True)
    parser.add_argument("--no-recursive-archive", type=Path, required=True)
    parser.add_argument("--controls-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = PaperProtocol.read(args.protocol)
    lock = _read_mapping(args.protocol_lock, "protocol lock")
    manifest = SplitManifest.read(args.manifest)
    receipt = freeze_paper_workspace(
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
        benchmark_root=args.benchmark_root,
        project_root=args.project_root,
        recursive_report_path=args.recursive_report,
        recursive_archive_path=args.recursive_archive,
        no_recursive_report_path=args.no_recursive_report,
        no_recursive_archive_path=args.no_recursive_archive,
        controls_output_root=args.controls_out,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "receipt_hash": receipt["receipt_hash"],
                "selected_candidate_available": receipt["selected_candidate_available"],
                "test_content_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
