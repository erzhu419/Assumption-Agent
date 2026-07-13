from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import Event, EventSink, NullEventSink
from ..models import ActionNode, HypothesisProgram, HypothesisStatus, stable_hash
from ..splits import BenchmarkItem, SplitManifest
from ..validation import backend_action_contract_issues


SKILL_ROUTING_VERSION = "per_item_trigger_routing_v2"
LEGACY_SKILL_ACTION_LOWERING_VERSION = "skilllearn_prompt_directive_lowering_v1"
SKILL_ACTION_LOWERING_VERSION = "skilllearn_prompt_directive_lowering_v2"
SKILL_FALLBACK_SEMANTICS_VERSION = "baseline_on_nonactivation_only_v1"
SKILLLEARN_ALLOWED_ACTION_OPERATIONS = frozenset(
    {"execute_step", "check_condition", "produce_artifact", "request_evidence"}
)
NO_SKILL_TREATMENT_HASH = stable_hash(
    {
        "routing_version": SKILL_ROUTING_VERSION,
        "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
        "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
        "skill_contents": [],
    }
)


@dataclass(frozen=True)
class LoweredSkillAction:
    action_id: str
    semantics: str
    instruction: str

    def to_dict(self) -> dict[str, str]:
        return {
            "action_id": self.action_id,
            "semantics": self.semantics,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class SkillCompileResult:
    output_root: Path
    skill_paths: tuple[Path, ...]
    family_count: int
    hypothesis_ids: tuple[str, ...]
    manifest_hash: str
    item_sources: Mapping[str, Path]
    program_set_hash: str
    treatment_hash: str
    item_treatment_hashes: Mapping[str, str]

    def source_for(self, item_id: str) -> Path | None:
        return self.item_sources.get(stable_hash({"item_id": item_id}))

    def treatment_hash_for(self, item_id: str) -> str:
        return self.item_treatment_hashes.get(
            stable_hash({"item_id": item_id}),
            NO_SKILL_TREATMENT_HASH,
        )


class SkillLearnProgramCompiler:
    """Compile promoted programs into SkillLearnBench-compatible SKILL.md files."""

    def __init__(self, *, event_sink: EventSink | None = None) -> None:
        self.event_sink = event_sink or NullEventSink()

    def compile(
        self,
        *,
        programs: Sequence[HypothesisProgram],
        items: Sequence[BenchmarkItem],
        split_manifest: SplitManifest,
        output_root: str | Path,
        method_name: str = "assumption-agent-v2",
        allowed_statuses: set[HypothesisStatus] | None = None,
        target_item_ids: Sequence[str] | None = None,
        target_split: str = "train",
        trace_id: str = "skill_compile",
    ) -> SkillCompileResult:
        allowed = allowed_statuses or {HypothesisStatus.PROMOTED}
        destination = Path(output_root) / method_name
        target_ids = set(target_item_ids or split_manifest.train_ids)
        known_manifest_ids = {
            *split_manifest.train_ids,
            *split_manifest.validation_ids,
            *split_manifest.test_ids,
        }
        if not target_ids <= known_manifest_ids:
            raise ValueError("compiler target IDs are outside the split manifest")
        split_ids = {
            "train": set(split_manifest.train_ids),
            "validation": set(split_manifest.validation_ids),
            "test": set(split_manifest.test_ids),
        }
        if target_split not in split_ids:
            raise ValueError("compiler target split must be train, validation, or test")
        if not target_ids <= split_ids[target_split]:
            raise PermissionError("compiler target IDs do not belong to the declared target split")
        target_items = [item for item in items if item.id in target_ids]
        if len(target_items) != len(target_ids):
            raise ValueError("split manifest references missing SkillLearnBench target items")
        program_rows: list[
            tuple[
                HypothesisProgram,
                str,
                tuple[LoweredSkillAction, ...],
                str,
                str,
            ]
        ] = []
        seen_program_ids: set[str] = set()
        for program in sorted(programs, key=lambda row: row.id):
            if program.status not in allowed:
                continue
            validation_issues = program.validate()
            if validation_issues:
                raise ValueError(
                    "SkillLearn compiler received an invalid program: "
                    f"{program.id}: {validation_issues}"
                )
            if program.id in seen_program_ids:
                raise ValueError("SkillLearn compiler program IDs must be unique")
            seen_program_ids.add(program.id)
            lowered_actions = _lower_skilllearn_program(program)
            skill_name = _slug(program.id)
            skill_text = _render_skill(program, skill_name, lowered_actions)
            treatment_hash = skilllearn_program_treatment_hash(
                program,
                lowered_actions=lowered_actions,
                rendered_skill=skill_text,
            )
            program_rows.append(
                (
                    program,
                    skill_name,
                    lowered_actions,
                    skill_text,
                    treatment_hash,
                )
            )

        program_set_hash = stable_hash(
            {
                "routing_version": SKILL_ROUTING_VERSION,
                "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
                "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
                "program_treatment_hashes": sorted(
                    row[4] for row in program_rows
                ),
            }
        )
        rendered_skills: dict[str, tuple[HypothesisProgram, str, str]] = {}
        used_hypotheses: set[str] = set()
        families: set[str] = set()
        routed_item_hashes: set[str] = set()
        action_lowering_hashes: dict[str, str] = {
            program.id: stable_hash(
                [
                    {
                        "semantics": row.semantics,
                        "instruction": row.instruction,
                    }
                    for row in lowered_actions
                ]
            )
            for program, _, lowered_actions, _, _ in program_rows
        }
        program_treatment_hashes = {
            program.id: treatment_hash
            for program, _, _, _, treatment_hash in program_rows
        }
        item_skill_content_hashes: dict[str, list[str]] = {
            item.id_hash: [] for item in target_items
        }
        for program, skill_name, _, skill_text, _ in program_rows:
            matched_items = sorted(
                (
                    item
                    for item in target_items
                    if program.matches({**dict(item.features), "family": item.family})
                ),
                key=lambda item: item.id_hash,
            )
            for item in matched_items:
                item_hash = item.id_hash
                relative_path = str(
                    Path("items") / item_hash / skill_name / "SKILL.md"
                )
                if relative_path in rendered_skills:
                    raise ValueError(
                        "SkillLearn compiler produced colliding skill paths: "
                        f"{relative_path}"
                    )
                content_hash = stable_hash({"content": skill_text})
                rendered_skills[relative_path] = (
                    program,
                    skill_text,
                    content_hash,
                )
                item_skill_content_hashes[item_hash].append(content_hash)
                routed_item_hashes.add(item_hash)
                used_hypotheses.add(program.id)
                families.add(item.family)

        item_treatment_hashes = {
            item_hash: (
                stable_hash(
                    {
                        "routing_version": SKILL_ROUTING_VERSION,
                        "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
                        "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
                        "skill_content_hashes": sorted(content_hashes),
                    }
                )
                if content_hashes
                else NO_SKILL_TREATMENT_HASH
            )
            for item_hash, content_hashes in sorted(
                item_skill_content_hashes.items()
            )
        }
        treatment_hash = stable_hash(
            {
                "program_set_hash": program_set_hash,
                "item_treatment_hashes": item_treatment_hashes,
            }
        )
        skill_content_hashes = {
            relative_path: row[2]
            for relative_path, row in sorted(rendered_skills.items())
        }
        compile_manifest = {
            "method_name": method_name,
            "routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "external_verifier_exposed_to_agent": False,
            "action_lowering_hashes": action_lowering_hashes,
            "program_treatment_hashes": program_treatment_hashes,
            "program_set_hash": program_set_hash,
            "skill_content_hashes": skill_content_hashes,
            "item_treatment_hashes": item_treatment_hashes,
            "treatment_hash": treatment_hash,
            "skill_paths": sorted(rendered_skills),
            "item_routes": {
                item.id_hash: (
                    str(Path("items") / item.id_hash)
                    if item.id_hash in routed_item_hashes
                    else None
                )
                for item in sorted(target_items, key=lambda row: row.id_hash)
            },
            "family_count": len(families),
            "hypothesis_ids": sorted(used_hypotheses),
            "split_manifest_hash": split_manifest.manifest_hash,
            "source_split": "train",
            "target_split": target_split,
            "target_item_set_hash": stable_hash({"item_ids": sorted(target_ids)}),
            "test_content_accessed": False,
            "raw_content_persisted": False,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-",
                dir=destination.parent,
            )
        )
        try:
            for relative_path, (_, skill_text, _) in sorted(
                rendered_skills.items()
            ):
                skill_path = staging / relative_path
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(skill_text, encoding="utf-8")
            (staging / "compile_manifest.json").write_text(
                json.dumps(compile_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _atomic_replace_directory(staging, destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

        skill_paths = tuple(
            destination / relative_path for relative_path in sorted(rendered_skills)
        )
        item_sources = {
            item_hash: destination / "items" / item_hash
            for item_hash in sorted(routed_item_hashes)
        }
        items_by_hash = {item.id_hash: item for item in target_items}
        for relative_path, (program, _, content_hash) in sorted(
            rendered_skills.items()
        ):
            item_hash = Path(relative_path).parts[1]
            item = items_by_hash[item_hash]
            skill_path = destination / relative_path
            self.event_sink.emit(
                Event(
                    event="skilllearn_skill_compiled",
                    stage="benchmark.skilllearn.compile",
                    trace_id=trace_id,
                    payload={
                        "hypothesis_id": program.id,
                        "hypothesis_hash": program.payload_hash,
                        "program_treatment_hash": (
                            program_treatment_hashes[program.id]
                        ),
                        "program_set_hash": program_set_hash,
                        "item_id_hash": item_hash,
                        "item_treatment_hash": item_treatment_hashes[item_hash],
                        "family_hash": stable_hash({"family": item.family}),
                        "skill_path_hash": stable_hash({"path": str(skill_path)}),
                        "skill_content_hash": content_hash,
                        "split_manifest_hash": split_manifest.manifest_hash,
                        "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
                        "action_lowering_hash": action_lowering_hashes[program.id],
                        "external_verifier_exposed_to_agent": False,
                        "source_split": "train",
                        "target_split": target_split,
                    },
                )
            )
        return SkillCompileResult(
            output_root=destination,
            skill_paths=skill_paths,
            family_count=len(families),
            hypothesis_ids=tuple(sorted(used_hypotheses)),
            manifest_hash=stable_hash(compile_manifest),
            item_sources=dict(item_sources),
            program_set_hash=program_set_hash,
            treatment_hash=treatment_hash,
            item_treatment_hashes=item_treatment_hashes,
        )


def _render_skill(
    program: HypothesisProgram,
    skill_name: str,
    lowered_actions: Sequence[LoweredSkillAction],
) -> str:
    description = program.statement.replace("\n", " ").strip()
    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {json.dumps(description, ensure_ascii=True)}",
        "---",
        f"# {description}",
        "",
        "## Activation",
        "",
    ]
    lines.extend(_render_trigger(program))
    lines.extend(["", "## Procedure", ""])
    for index, action in enumerate(lowered_actions, start=1):
        label = (
            "Agent-local self-check"
            if action.semantics == "agent_local_self_check"
            else "Agent instruction"
        )
        lines.append(f"{index}. **{label}:** {action.instruction}")
    lines.extend(
        [
            "",
            "## Evaluation boundary",
            "",
            "- Only the task-local instructions and evidence available during this run may be used.",
            "- The benchmark verifier runs after the agent exits and is not callable from this skill.",
            "",
            "## Fallback",
            "",
            "The frozen router omits this skill when activation does not match. Once this skill is injected, the benchmark does not replace the result with a post-hoc baseline output.",
            "",
        ]
    )
    return "\n".join(lines)


def _lower_skilllearn_program(
    program: HypothesisProgram,
) -> tuple[LoweredSkillAction, ...]:
    contract_issues = backend_action_contract_issues(
        program,
        allowed_operations=SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
        external_evidence_is_hidden=True,
    )
    if contract_issues:
        summary = (
            "contains operations without a backend lowering"
            if any(
                issue.startswith("unsupported_action_operation:")
                for issue in contract_issues
            )
            else "references hidden external evaluation evidence"
        )
        raise ValueError(
            f"SkillLearn action graph {summary}: "
            f"{list(contract_issues)}"
        )
    lowered: list[LoweredSkillAction] = []
    for action in _ordered_actions(program.action_graph):
        value = _display_action_value(action.value).strip()
        target = action.target.strip()
        if action.operation == "execute_step":
            instruction = (
                f"Execute the task step `{target}`: {value}"
                if value
                else f"Execute the task step `{target}`."
            )
            semantics = "prompt_directive"
        elif action.operation == "produce_artifact":
            detail = f": {value}" if value else "."
            instruction = f"Produce the requested artifact `{target}`{detail}"
            semantics = "prompt_directive"
        elif action.operation == "request_evidence":
            detail = f": {value}" if value else "."
            instruction = (
                f"Gather task-local evidence `{target}`{detail} Do not request "
                "policy-off/on results or the hidden benchmark verifier."
            )
            semantics = "prompt_directive"
        else:
            detail = f"`{target}`: {value}" if value else target
            instruction = f"Before completion, check locally that {detail}"
            if not instruction.endswith((".", "!", "?")):
                instruction += "."
            semantics = "agent_local_self_check"
        lowered.append(
            LoweredSkillAction(
                action_id=action.id,
                semantics=semantics,
                instruction=instruction,
            )
        )
    return tuple(lowered)


def skilllearn_program_treatment_hash(
    program: HypothesisProgram,
    *,
    lowered_actions: Sequence[LoweredSkillAction] | None = None,
    rendered_skill: str | None = None,
) -> str:
    """Hash only the external treatment that can reach the SkillLearn agent."""

    lowered = tuple(lowered_actions or _lower_skilllearn_program(program))
    skill_text = rendered_skill or _render_skill(
        program,
        _slug(program.id),
        lowered,
    )
    return stable_hash(
        {
            "routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "external_verifier_exposed_to_agent": False,
            "rendered_skill_hash": stable_hash({"content": skill_text}),
        }
    )


def skilllearn_program_set_treatment_hash(
    programs: Sequence[HypothesisProgram],
) -> str:
    return stable_hash(
        {
            "routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "program_treatment_hashes": sorted(
                skilllearn_program_treatment_hash(program)
                for program in programs
            ),
        }
    )


def _atomic_replace_directory(staging: Path, destination: Path) -> None:
    """Publish a complete compiler tree without merging stale files into it."""

    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(
            f".{destination.name}.backup-{uuid.uuid4().hex}"
        )
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None and backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _render_trigger(program: HypothesisProgram) -> list[str]:
    rows: list[str] = []
    for group_name, predicates in (
        ("Require all", program.trigger.all_of),
        ("Require any", program.trigger.any_of),
        ("Exclude", (*program.trigger.none_of, *program.anti_trigger.all_of, *program.anti_trigger.any_of)),
    ):
        for predicate in predicates:
            rows.append(f"- {group_name}: `{predicate.key}` `{predicate.op}` `{_display_value(predicate.value)}`")
    return rows or ["- Apply only when the structured task router selects this program."]


def _ordered_actions(actions: tuple[ActionNode, ...]) -> tuple[ActionNode, ...]:
    by_id = {action.id: action for action in actions}
    pending = {action.id: set(action.depends_on) for action in actions}
    ordered: list[ActionNode] = []
    while pending:
        ready = sorted(action_id for action_id, dependencies in pending.items() if not dependencies)
        if not ready:
            raise ValueError("cannot compile a cyclic action graph")
        for action_id in ready:
            ordered.append(by_id[action_id])
            pending.pop(action_id)
            for dependencies in pending.values():
                dependencies.discard(action_id)
    return tuple(ordered)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "hypothesis-program"


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _display_action_value(value: Any, *, humanize_identifiers: bool = False) -> str:
    """Render structured action values as deterministic, agent-readable prose."""

    if value is None:
        return ""
    if isinstance(value, str):
        if humanize_identifiers and re.fullmatch(r"[A-Za-z0-9_]+", value):
            return value.replace("_", " ")
        return value
    if isinstance(value, Mapping):
        return "; ".join(
            f"{_humanize_action_identifier(str(key))}: "
            f"{_display_action_value(value[key], humanize_identifiers=True)}"
            for key in sorted(value, key=lambda row: str(row))
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ", ".join(
            _display_action_value(row, humanize_identifiers=True)
            for row in value
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _humanize_action_identifier(value: str) -> str:
    return value.replace("_", " ")
