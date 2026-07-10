from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import Event, EventSink, NullEventSink
from ..models import ActionNode, HypothesisProgram, HypothesisStatus, stable_hash
from ..splits import BenchmarkItem, SplitManifest


SKILL_ROUTING_VERSION = "per_item_trigger_routing_v1"


@dataclass(frozen=True)
class SkillCompileResult:
    output_root: Path
    skill_paths: tuple[Path, ...]
    family_count: int
    hypothesis_ids: tuple[str, ...]
    manifest_hash: str
    item_sources: Mapping[str, Path]

    def source_for(self, item_id: str) -> Path | None:
        return self.item_sources.get(stable_hash({"item_id": item_id}))


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
        skill_paths: list[Path] = []
        used_hypotheses: set[str] = set()
        families: set[str] = set()
        item_sources: dict[str, Path] = {}
        for program in sorted(programs, key=lambda row: row.id):
            if program.status not in allowed:
                continue
            if program.validate():
                continue
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
                skill_name = _slug(program.id)
                item_source = destination / "items" / item_hash
                skill_dir = item_source / skill_name
                skill_dir.mkdir(parents=True, exist_ok=True)
                skill_path = skill_dir / "SKILL.md"
                skill_path.write_text(_render_skill(program, skill_name), encoding="utf-8")
                skill_paths.append(skill_path)
                item_sources[item_hash] = item_source
                used_hypotheses.add(program.id)
                families.add(item.family)
                self.event_sink.emit(
                    Event(
                        event="skilllearn_skill_compiled",
                        stage="benchmark.skilllearn.compile",
                        trace_id=trace_id,
                        payload={
                            "hypothesis_id": program.id,
                            "hypothesis_hash": program.payload_hash,
                            "item_id_hash": item_hash,
                            "family_hash": stable_hash({"family": item.family}),
                            "skill_path_hash": stable_hash({"path": str(skill_path)}),
                            "split_manifest_hash": split_manifest.manifest_hash,
                            "source_split": "train",
                            "target_split": target_split,
                        },
                    )
                )
        compile_manifest = {
            "method_name": method_name,
            "routing_version": SKILL_ROUTING_VERSION,
            "skill_paths": [str(path.relative_to(destination)) for path in skill_paths],
            "item_routes": {
                item.id_hash: (
                    str(item_sources[item.id_hash].relative_to(destination))
                    if item.id_hash in item_sources
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
        (destination / "compile_manifest.json").write_text(
            json.dumps(compile_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return SkillCompileResult(
            output_root=destination,
            skill_paths=tuple(skill_paths),
            family_count=len(families),
            hypothesis_ids=tuple(sorted(used_hypotheses)),
            manifest_hash=stable_hash(compile_manifest),
            item_sources=dict(item_sources),
        )


def _render_skill(program: HypothesisProgram, skill_name: str) -> str:
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
    for index, action in enumerate(_ordered_actions(program.action_graph), start=1):
        value = _display_value(action.value)
        instruction = f"{action.operation} `{action.target}`"
        if value:
            instruction += f": {value}"
        lines.append(f"{index}. {instruction}")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            *[f"- {check}" for check in program.verifier.checks],
        ]
    )
    if program.verifier.required_evidence:
        lines.extend(["", "Required evidence:"])
        lines.extend(f"- {row}" for row in program.verifier.required_evidence)
    lines.extend(
        [
            "",
            "## Expected Effect",
            "",
            f"- Metric: `{program.expected_effect.metric}`",
            f"- Minimum held-out delta: `{program.expected_effect.minimum_delta}`",
            f"- Maximum harm rate: `{program.expected_effect.maximum_harm_rate}`",
            "",
            "## Fallback",
            "",
            "Preserve the baseline workflow whenever the activation conditions or verification checks do not pass.",
            "",
        ]
    )
    return "\n".join(lines)


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
