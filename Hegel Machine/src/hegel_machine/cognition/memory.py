"""Executable knowledge, parameterized abstractions and empirical scope repair."""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field, replace
from itertools import islice
import json
from typing import Any, Sequence

from .abstractions import (AbstractTemplate, extract_templates, fit_template,
                           fitting_shape, screen_template, trajectory_losses)
from .language import Definition, Program
from .models import Candidate, Observation, SearchBudget, TaskSpec
from .condition_learning import learn_condition, prediction_matches
from .knowledge_returns import transfer_condition, return_priority, return_scope_matches


def _object_schema(spec):
    return json.loads(spec.context["object_schema"]) if "object_schema" in spec.context else None


def _row_payload(row: Observation) -> dict[str, Any]:
    return {"features": dict(row.features), "target": row.target,
            "history": [dict(previous) for previous in row.history]}


def _as_observation(example: Observation | dict[str, Any]) -> Observation:
    if isinstance(example, Observation):
        return example
    features = example.get("features", example.get("observables", example))
    return Observation({name: float(value) for name, value in features.items()
                        if isinstance(value, (float, int))},
                       float(example.get("target", 0.0)),
                       tuple(dict(row) for row in example.get("history", [])))


def _read(row: Observation, name: str, lag: int) -> float | None:
    source = row.features if lag == 0 else row.history[-lag] if len(row.history) >= lag else {}
    return float(source[name]) if name in source else None


@dataclass
class EmpiricalScope:
    task_id: str
    system_id: str | None = None
    support: list[dict[str, Any]] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    observed_rows: int = 0
    fallback: float = 0.0
    verified: bool = True
    predicate: dict[str, Any] | None = None
    support_examples: list[dict[str, Any]] = field(default_factory=list)
    counter_examples: list[dict[str, Any]] = field(default_factory=list)
    exception_program: dict[str, Any] | None = None

    @classmethod
    def measured(cls, spec: TaskSpec, program: Program,
                 observations: Sequence[Observation]) -> "EmpiricalScope":
        support = []
        for name, lag in sorted(program.observed_variables()):
            values = [value for row in observations if (value := _read(row, name, lag)) is not None]
            if values:
                support.append({"name": name, "lag": lag, "lower": min(values), "upper": max(values)})
        return cls(spec.task_id, spec.context.get("system_id"), support=support,
                   observed_rows=len(observations),
                   fallback=sum(row.target for row in observations) / len(observations) if observations else 0.0,
                   support_examples=[_row_payload(row) for row in observations if prediction_matches(program, row)])

    def matches_conditions(self, row: Observation) -> bool:
        if self.predicate is not None:
            predicate = Program.from_dict(self.predicate)
            if len(row.history) < predicate.minimum_history_depth or not predicate.required_features().issubset(row.features):
                return False
            if not predicate.predict(row):
                return False
        for axis in self.conditions:
            value = _read(row, axis["name"], axis["lag"])
            if value is None:
                return False
            if axis.get("lower") is not None and value < axis["lower"]:
                return False
            if axis.get("upper") is not None and value >= axis["upper"]:
                return False
        return True

    def supports(self, row: Observation) -> bool:
        return bool(self.verified and self.observed_rows and self.matches_conditions(row) and all(
            (value := _read(row, axis["name"], axis["lag"])) is not None
            and axis["lower"] <= value <= axis["upper"] for axis in self.support))

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "system_id": self.system_id,
                "support": [dict(axis) for axis in self.support],
                "conditions": [dict(condition) for condition in self.conditions],
                "observed_rows": self.observed_rows, "fallback": self.fallback,
                "verified": self.verified, "predicate": deepcopy(self.predicate),
                "support_examples": deepcopy(self.support_examples),
                "counter_examples": deepcopy(self.counter_examples),
                "exception_program": deepcopy(self.exception_program)}


@dataclass
class Concept:
    concept_id: str
    program: Program
    risk_quantile: float
    learned_on: str
    operation: str
    status: str = "available"
    successful_tasks: list[str] = field(default_factory=list)
    excluded_tasks: list[str] = field(default_factory=list)
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    generation: int = 0
    scopes: list[EmpiricalScope] = field(default_factory=list)
    reuse_history: list[dict[str, Any]] = field(default_factory=list)
    production_methods: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**deepcopy(self.__dict__), "program": self.program.to_dict(),
                "scopes": [scope.to_dict() for scope in self.scopes]}


def _guard(program: Program, scope: EmpiricalScope | None) -> Program:
    if scope is None or (not scope.conditions and scope.predicate is None):
        return program
    condition = Program.from_dict(scope.predicate) if scope.predicate is not None else None
    for axis in scope.conditions:
        variable = Program("feature", name=axis["name"]) if axis["lag"] == 0 else Program("lag", name=axis["name"], lag=axis["lag"])
        checks = []
        if axis.get("lower") is not None:
            checks.append(Program("ge", children=(variable, Program("const", value=axis["lower"]))))
        if axis.get("upper") is not None:
            checks.append(Program("lt", children=(variable, Program("const", value=axis["upper"]))))
        for check in checks:
            condition = check if condition is None else Program("and", children=(condition, check))
    return Program("if", children=(condition, program, Program("const", value=scope.fallback))) if condition else program


def _called_definitions(program: Program) -> list[Definition]:
    found: dict[str, Definition] = {}

    def visit(node):
        if node.op == "call" and node.definition.definition_id not in found:
            found[node.definition.definition_id] = node.definition
            visit(node.definition.body)
        for child in node.children:
            visit(child)

    visit(program)
    return list(found.values())


class KnowledgeLibrary:
    def __init__(self, concepts: list[Concept] | None = None,
                 abstracts: list[AbstractTemplate] | None = None,
                 *, next_concept_index: int | None = None,
                 deleted_ids: list[str] | None = None,
                 research_experience: dict[str, dict[str, float]] | None = None,
                 action_experience: dict | None = None,
                 definitions: list[dict] | None = None,
                 experiments: list[dict] | None = None, methods: dict | None = None,
                 evolution: dict | None = None, structural_priors: list[dict] | None = None,
                 research_opportunities: dict | None = None,
                 measurements: list[dict] | None = None, meta_priors: list[dict] | None = None,
                 instrument_models: list[dict] | None = None,
                 object_models: list[dict] | None = None,
                 program_experiment_methods: list[dict] | None = None,
                 research_drafts: list[dict] | None = None,
                 transfer_serial: int | None = None,
                 condition_coverages: list[dict] | None = None,
                 research_workflows: dict | None = None):
        self.concepts = list(concepts or [])
        self.abstracts = list(abstracts) if abstracts is not None else [
            template for concept in self.concepts for template in extract_templates(concept.program, concept.concept_id)]
        highest = max((int(concept.concept_id.removeprefix("concept_")) for concept in self.concepts
                       if concept.concept_id.removeprefix("concept_").isdigit()), default=0)
        self.next_concept_index = next_concept_index if next_concept_index is not None else highest + 1
        self.deleted_ids = list(deleted_ids or [])
        self.research_experience = deepcopy(research_experience or {})
        self.action_experience = deepcopy(action_experience or {})
        from .experiment_knowledge import ExperimentKnowledge
        from .research_methods import ResearchMethods
        from .growth_methods import ResearchMethodBook
        self.experiments = ExperimentKnowledge(experiments or [])
        self.methods = ResearchMethods(methods)
        self.research_workflows = ResearchMethodBook.from_dict(research_workflows or {})
        from .measurement_acquisition import MeasurementAcquisitionMemory
        self.measurements = MeasurementAcquisitionMemory(self, measurements or [])
        from .research_opportunities import ResearchOpportunities
        self.research_opportunities = ResearchOpportunities(research_opportunities)
        from .knowledge_evolution import KnowledgeEvolution
        self.evolution = KnowledgeEvolution.from_dict(evolution)
        self.structural_priors = deepcopy(structural_priors or [])
        self.meta_priors = deepcopy(meta_priors or [])
        self.research_drafts = deepcopy(research_drafts or [])
        self.condition_coverages = deepcopy(condition_coverages or [])
        self.instrument_models = deepcopy(instrument_models or [])
        self.object_models = deepcopy(object_models or [])
        from .program_experiment_methods import ProgramExperimentMethods
        self.program_experiment_methods = ProgramExperimentMethods(program_experiment_methods)
        self.last_search: dict[str, Any] = {}
        self.last_callable_selection: dict[str, Any] = {}
        self._pending_transfers: dict[str, dict[str, Any]] = {}
        self._search_serial = 0
        # Visit identifiers must remain unique after a library is restored.
        # Older payloads do not carry a counter, so recover the largest
        # persisted ``transfer:N`` identifier from source histories.
        persisted_serials = []
        for concept in self.concepts:
            for item in concept.reuse_history:
                identifier = item.get("episode_id")
                if isinstance(identifier, str) and identifier.startswith("transfer:"):
                    suffix = identifier.removeprefix("transfer:")
                    if suffix.isdigit():
                        persisted_serials.append(int(suffix))
        self._transfer_serial = (int(transfer_serial) if transfer_serial is not None
                                 else max(persisted_serials, default=0))
        self._transfer_episode: dict[str, Any] | None = None
        self.definitions: dict[str, Definition] = {}
        self.definition_bindings: dict[str, dict] = {}
        self._call_templates = self._definition_templates()
        if definitions is None:
            self._refresh_definitions()
        else:
            for entry in definitions:
                definition = Definition.from_dict(entry["definition"])
                self.definitions[definition.definition_id] = definition
                self.definition_bindings[definition.definition_id] = {
                    "template_id": entry["template_id"],
                    "arguments": tuple(Program.from_dict(argument) for argument in entry["arguments"]),
                }

    def _definition_templates(self) -> dict[str, AbstractTemplate]:
        """Expose interval guards to calls without adding refit competitors."""
        templates = {template.template_id: template for template in self.abstracts}
        for concept in self.concepts:
            for scope in concept.scopes:
                if not scope.conditions or scope.predicate is not None:
                    continue
                prefix = f"{concept.concept_id}:condition:{scope.task_id}:supported"
                template = extract_templates(_guard(concept.program, scope), prefix)[0]
                template.source_concept_id = concept.concept_id
                template.parents = [concept.concept_id]
                templates[template.template_id] = template
        return templates

    def _refresh_definitions(self) -> None:
        sources = {concept.concept_id: concept for concept in self.concepts}
        self._call_templates = self._definition_templates()
        existing = set(self._call_templates)
        for definition_id, binding in list(self.definition_bindings.items()):
            if binding["template_id"] not in existing:
                del self.definition_bindings[definition_id]
                del self.definitions[definition_id]
        previous_ids = {binding["template_id"]: definition_id
                        for definition_id, binding in self.definition_bindings.items()}
        for template in self._call_templates.values():
            source = sources.get(template.source_concept_id)
            if source is None or template.pattern.size <= 1 or ":repair:" in template.template_id:
                continue
            pattern = template.pattern
            if template.template_id == f"{source.concept_id}:abstract:0":
                # Keep the learned program's calls and their dependencies in the
                # whole-concept definition. Refit templates remain expanded.
                pattern = source.program.remap_features(dict(zip(template.source_features, template.roles)))
            leaves: list[Program] = []

            def parameterize(node):
                if node.op in {"feature", "lag", "available"}:
                    if node not in leaves:
                        leaves.append(node)
                    return Program("arg", name=f"argument_{leaves.index(node)}")
                return Program(node.op, node.value, node.name, node.lag,
                               tuple(parameterize(child) for child in node.children),
                               definition=node.definition)

            body = parameterize(pattern)
            if not leaves:
                continue
            definition_id = f"{template.template_id}:definition"
            previous_id = previous_ids.get(template.template_id)
            if previous_id is not None:
                previous = self.definitions[previous_id]
                if previous.body == body and previous.arguments == tuple(f"argument_{index}" for index in range(len(leaves))):
                    definition_id = previous_id
                else:
                    revision = 2 if previous_id == definition_id else int(previous_id.rsplit(":", 1)[1]) + 1
                    definition_id = f"{definition_id}:{revision}"
                    del self.definitions[previous_id]
                    del self.definition_bindings[previous_id]
            definition = Definition(definition_id, tuple(f"argument_{index}" for index in range(len(leaves))),
                                    body, source_concept_id=source.concept_id)
            self.definitions[definition_id] = definition
            self.definition_bindings[definition_id] = {"template_id": template.template_id,
                                                       "arguments": tuple(leaves)}

    def _calls_available(self, program: Program, spec: TaskSpec) -> bool:
        definitions = _called_definitions(program)
        if not definitions:
            return True
        sources = {concept.concept_id: concept for concept in self.concepts}
        templates = self._call_templates
        for definition in definitions:
            source_id = definition.source_concept_id
            source = sources.get(source_id)
            if source_id is not None and (source is None or source.status == "revoked" or source_id in self.deleted_ids):
                return False
            binding = self.definition_bindings.get(definition.definition_id)
            template = templates.get(binding["template_id"]) if binding else None
            if template and template.status == "revoked":
                return False
            if source and binding is None:
                local = self._scope_for(source, spec)
                if local and (local.conditions or local.predicate):
                    return False
            if source and template and template.kind == "theory":
                local = self._scope_for(source, spec)
                if local and (local.conditions or local.predicate):
                    prefix = f"{source.concept_id}:condition:{local.task_id}:supported:"
                    if not template.template_id.startswith(prefix):
                        return False
        return True

    def callable_atoms(self, spec: TaskSpec, *, structural_transfer: bool = True,
                       limit: int = 8) -> list[Program]:
        """Bind learned computations as atoms for subsequent program search."""
        sources = {concept.concept_id: concept for concept in self.concepts}
        templates = self._call_templates
        eligible = []
        for definition_id, definition in self.definitions.items():
            binding = self.definition_bindings[definition_id]
            template = templates.get(binding["template_id"])
            source = sources.get(definition.source_concept_id)
            if (template is None or source is None or source.status == "revoked" or template.status == "revoked"
                    or source.risk_quantile != spec.risk_quantile or spec.task_id in source.excluded_tasks):
                continue
            if not self._calls_available(source.program, spec):
                continue
            eligible.append((template, definition, binding, source))
        eligible.sort(key=lambda item: (item[0].kind != "theory", -len(set(item[3].successful_tasks)),
                                        item[0].pattern.size, item[0].template_id))
        schema = _object_schema(spec)
        queues = [(definition, binding, iter(template.mappings(spec.feature_names,
                    structural_transfer=structural_transfer, object_schema=schema)))
                  for template, definition, binding, _ in eligible]
        atoms = []
        while queues and len(atoms) < limit:
            remaining = []
            for definition, binding, mappings in queues:
                mapping = next(mappings, None)
                if mapping is None:
                    continue
                remaining.append((definition, binding, mappings))
                call = Program("call", name=definition.definition_id,
                               children=tuple(argument.remap_features(mapping) for argument in binding["arguments"]),
                               definition=definition)
                if self._calls_available(call, spec) and call not in atoms:
                    atoms.append(call)
                    if len(atoms) >= limit:
                        break
            queues = remaining
        return atoms

    def _selection_calls(self, spec: TaskSpec, *, structural_transfer: bool):
        from .concept_selection import is_raw_affine

        sources = {concept.concept_id: concept for concept in self.concepts}
        ordinary_templates = {template.template_id for template in self.abstracts}
        groups = {}
        schema = _object_schema(spec)
        for definition_id, definition in self.definitions.items():
            binding = self.definition_bindings[definition_id]
            template = self._call_templates.get(binding["template_id"])
            source = sources.get(definition.source_concept_id)
            if (template is None or source is None or source.status == "revoked" or template.status == "revoked"
                    or source.risk_quantile != spec.risk_quantile or spec.task_id in source.excluded_tasks
                    or not self._calls_available(source.program, spec) or is_raw_affine(template.pattern)):
                continue
            if template.template_id not in ordinary_templates:
                # A task's observed interval is a condition for that task, not
                # another universal computation to rank in every future task.
                local = self._scope_for(source, spec)
                if local is None or not template.template_id.startswith(
                        f"{source.concept_id}:condition:{local.task_id}:supported:"):
                    continue
            shape = fitting_shape(template, {})
            group = groups.setdefault(shape, {})
            group.setdefault(source.concept_id, deque()).append((
                definition, binding, iter(template.mappings(spec.feature_names,
                    structural_transfer=structural_transfer, object_schema=schema))))
        active = {shape: deque(group.values()) for shape, group in groups.items()}
        while active:
            for shape in list(active):
                sources_queue = active[shape]
                records = sources_queue.popleft()
                definition, binding, mappings = records.popleft()
                mapping = next(mappings, None)
                if mapping is not None:
                    records.append((definition, binding, mappings))
                if records:
                    sources_queue.append(records)
                if not sources_queue:
                    del active[shape]
                if mapping is None:
                    continue
                call = Program("call", name=definition.definition_id,
                               children=tuple(argument.remap_features(mapping) for argument in binding["arguments"]),
                               definition=definition)
                if self._calls_available(call, spec):
                    yield call

    def select_callable_atoms(self, spec: TaskSpec, observations: Sequence[Observation], budget: SearchBudget,
                              *, limit: int = 4, structural_transfer: bool = True,
                              baseline_program: Program | None = None) -> list[Program]:
        from .concept_selection import select_atoms

        atoms, self.last_callable_selection = select_atoms(
            self._selection_calls(spec, structural_transfer=structural_transfer), spec,
            list(observations), budget, limit=limit, baseline_program=baseline_program)
        return atoms

    @staticmethod
    def _scope_for(concept: Concept, spec: TaskSpec) -> EmpiricalScope | None:
        exact = next((scope for scope in concept.scopes if scope.task_id == spec.task_id), None)
        if exact is not None:
            return exact
        system_id = spec.context.get("system_id")
        return next((scope for scope in concept.scopes if system_id and scope.system_id == system_id), None)

    @staticmethod
    def _transfer_target(spec: TaskSpec) -> dict:
        """Identify this public target and observation view, not a task family."""
        return {"task_id": spec.task_id, "feature_names": list(spec.feature_names),
                "feature_bounds": {name: list(bounds) for name, bounds in spec.feature_bounds.items()},
                "risk_quantile": spec.risk_quantile,
                "decision_costs": list(spec.decision_costs) if spec.decision_costs is not None else None,
                "context": deepcopy(spec.context)}

    def draft_transfer_refuted(self, spec: TaskSpec, program: Program, *, fallback=None) -> bool:
        """A failed draft binding leaves its reusable procedure available."""
        target = self._transfer_target(spec)
        records = [item for draft in self.research_drafts for item in draft.get("outcomes", [])
                   if item.get("kind") == "transfer_counterexample" and item["target"] == target]
        if not records:
            return False
        canonical = program.expanded().simplified().to_dict()
        return any(item["program"] == canonical and (not program.minimum_history_depth
            or item.get("fallback") == fallback) for item in records)

    def transfer_refuted(self, spec: TaskSpec, program: Program, *, fallback=None) -> bool:
        """A target's counterexample withdraws its complete bound program only."""
        if self.draft_transfer_refuted(spec, program, fallback=fallback):
            return True
        target = self._transfer_target(spec)
        records = [record for concept in self.concepts for record in concept.reuse_history
                   if record.get("kind") == "transfer_counterexample" and record["target"] == target]
        excluded = [concept for concept in self.concepts if spec.task_id in concept.excluded_tasks]
        if not records and not excluded:
            return False
        canonical = program.expanded().simplified().to_dict()
        return (any(record["program"] == canonical for record in records)
                or any(concept.program.expanded().simplified().to_dict() == canonical for concept in excluded))

    def record_transfer_counterexample(self, spec: TaskSpec, program: Program | Candidate, source_ids, *,
                                      evidence: dict, probe_id: str, source_draft_ids=(),
                                      fallback=None) -> dict | None:
        """Record a measured failure without blaming its reusable components.

        Deterministic outcomes may refute directly. Once response replication
        participates, only its repeatable error can withdraw a program. Source
        associations identify actual consumption; they are not causal credit.
        A fitted Candidate retains consumed parents after calls are expanded.
        """
        replication = evidence.get("noise_replication")
        if (evidence.get("status") != "counterexample" or (replication is not None
                and (not replication.get("repeatable_error") or replication.get("noise_supported")))):
            return None
        consumed = set()
        if isinstance(program, Candidate):
            consumed.update(program.parents)
            if program.memory_id is not None:
                consumed.add(program.memory_id)
            program = program.program
        canonical = program.expanded().simplified().to_dict()
        called = {definition.source_concept_id for definition in _called_definitions(program)}
        sources = [concept for concept in self.concepts if concept.concept_id in source_ids
                   and (concept.concept_id in called | consumed
                        or concept.program.expanded().simplified().to_dict() == canonical)]
        drafts = [draft for draft in self.research_drafts
                  if draft["draft_id"] in source_draft_ids]
        if not sources and not drafts:
            return None
        record = {"kind": "transfer_counterexample", "target": self._transfer_target(spec),
                  "program": canonical, "probe_id": probe_id, "evidence": deepcopy(evidence)}
        recorded = []
        for source in sources:
            if any(item.get("kind") == record["kind"] and item["target"] == record["target"]
                   and item["program"] == canonical for item in source.reuse_history):
                continue
            source.reuse_history.append(deepcopy(record))
            recorded.append(source.concept_id)
        recorded_drafts = []
        for draft in drafts:
            outcomes = draft.setdefault("outcomes", [])
            if any(item.get("kind") == record["kind"] and item["target"] == record["target"]
                   and item["program"] == canonical and item.get("fallback") == fallback
                   for item in outcomes):
                continue
            outcomes.append({**deepcopy(record), "fallback": fallback})
            recorded_drafts.append(draft["draft_id"])
        return ({"action": "withdraw_target_transfer", "source_ids": recorded,
                 "source_draft_ids": recorded_drafts, "fallback": fallback, **deepcopy(record)}
                if recorded or recorded_drafts else None)

    def retrieve(self, spec: TaskSpec, *, observations: Sequence[Observation] = ()) -> list[Candidate]:
        condition = transfer_condition(spec, observations)
        eligible = [c for c in self.concepts
                    if c.status != "revoked" and spec.task_id not in c.excluded_tasks
                    and c.risk_quantile == spec.risk_quantile
                    and self._calls_available(c.program, spec)
                    and {name for name, _ in c.program.observed_variables() if name != "__target__"}.issubset(spec.feature_names)]
        eligible.sort(key=lambda c: (
            spec.task_id not in c.successful_tasks and self._scope_for(c, spec) is None,
            return_priority(c.reuse_history, condition),
            -len(set(c.successful_tasks)), len(c.counterexamples), c.program.size, c.concept_id))
        return [Candidate(c.concept_id, _guard(c.program, self._scope_for(c, spec)), "reuse",
                          "Execute a previously learned program within its measured context", (c.concept_id,), c.concept_id)
                for c in eligible if _guard(c.program, self._scope_for(c, spec)).required_features().issubset(spec.feature_names)
                and not self.transfer_refuted(spec, _guard(c.program, self._scope_for(c, spec)))]

    @staticmethod
    def _transfer_priority(template: AbstractTemplate, mapping: dict[str, str], condition=None,
                           mapping_roles=None) -> tuple:
        matched = [event for event in template.transfer_history
                   if (not event.get("condition") or event["condition"] == condition)
                   and (event.get("mapping_roles") == mapping_roles if event.get("mapping_roles") is not None
                        else not event.get("mapping") or event["mapping"] == mapping)]
        tested = [event for event in matched if event.get("evidence_status") != "proposal_not_compared"]
        successes = {event["task_id"] for event in tested if event["accepted"]}
        failures = {event["task_id"] for event in tested if not event["accepted"]}
        return (return_priority(matched, condition), -len(successes | set(template.uses)),
                len(failures), template.pattern.size, template.template_id)

    def propose(self, spec: TaskSpec, observations: Sequence[Observation], budget: SearchBudget,
                *, structural_transfer: bool = True,
                preserve_parameters: bool = True, source_concept_ids: set[str] | None = None) -> list[Candidate]:
        """Screen structures on complete training trajectories, then refit a few.

        Fitting is shared only by identical parameterized symbolic shapes in
        this call. Every emitted result retains its actual source and remains
        an adaptation proposal, including sources excluded from this context.
        Original parameters also compete when requested: startup observations
        cannot identify coefficients in a mature branch they never execute.
        """
        from .evaluation import initial_prediction, point_loss
        from .abstractions import _parameter_degree

        started = budget.spent
        condition = transfer_condition(spec, observations)
        mapping_roles = lambda mapping: {role: spec.feature_names.index(name) for role, name in mapping.items()}
        priority = lambda template, mapping: self._transfer_priority(template, mapping, condition, mapping_roles(mapping))
        work: dict[tuple, int] = {}
        preserved_work: dict[Program, dict] = {}
        self._search_serial += 1
        self.last_search = {"task_id": spec.task_id, "training_rows": len(observations),
                            "template_bindings": 0, "unique_fit_shapes": 0,
                            "screened_shapes": 0, "deep_fits": 0,
                            "shared_fit_bindings": 0, "screen_compute": 0,
                            "fit_compute": 0, "candidates": 0, "distinct_programs": 0,
                            "parameter_compute": 0, "parameter_trials": 0,
                            "preserved_parameter_candidates": 0,
                            "compute_cost": 0, "ranked_shapes": [],
                            "fit_cache_scope": "this_call_training_observations"}
        if not observations or budget.remaining < len(observations):
            return []
        sources = {concept.concept_id: concept for concept in self.concepts}
        groups: dict[tuple, list[tuple[AbstractTemplate, dict[str, str]]]] = {}
        schema = _object_schema(spec)
        # Enumerating nPk bindings before the first charged fit made wide
        # object theories effectively non-terminating. A source needs no more
        # candidate bindings than this call could evaluate on its full data.
        bindings_per_template = budget.remaining // len(observations)
        for template in self.abstracts:
            if source_concept_ids is not None and template.source_concept_id not in source_concept_ids:
                continue
            source = sources.get(template.source_concept_id)
            if source is None or source.status == "revoked" or template.status == "revoked" or source.risk_quantile != spec.risk_quantile:
                continue
            if not self._calls_available(source.program, spec):
                continue
            local = self._scope_for(source, spec)
            if local and local.predicate and template.kind == "theory" and ":condition:" not in template.template_id:
                # A scoped law cannot recover its former unrestricted form by
                # taking the parameter-transfer path. Its conditional templates
                # compete below, including a separately tested repair proposal.
                continue
            for mapping in islice(template.mappings(spec.feature_names,
                    structural_transfer=structural_transfer, object_schema=schema), bindings_per_template):
                if any(len(row.history) >= template.pattern.minimum_history_depth for row in observations):
                    groups.setdefault(fitting_shape(template, mapping), []).append((template, mapping))
        self.last_search["template_bindings"] = sum(map(len, groups.values()))
        self.last_search["unique_fit_shapes"] = len(groups)
        self.last_search["shared_fit_bindings"] = self.last_search["template_bindings"] - len(groups)
        if not groups or not budget.spend(len(observations)):
            return []
        fallback = initial_prediction(observations, spec.risk_quantile)
        baseline = sum(point_loss(fallback, row.target, spec.risk_quantile) for row in observations) / len(observations)
        for members in groups.values():
            members.sort(key=lambda member: priority(*member))

        # A fair first pass across history depths and nonlinear structures keeps
        # old insertion order from consuming all work before temporal models.
        families: dict[tuple, list[tuple]] = {}
        for shape, members in groups.items():
            template, mapping = members[0]
            family = (template.pattern.history_depth, _parameter_degree(template.pattern) > 1)
            families.setdefault(family, []).append(shape)
        for family in families:
            families[family].sort(key=lambda shape: priority(*groups[shape][0]))
        ordered = []
        while any(families.values()):
            for family in sorted(families):
                if families[family]:
                    ordered.append(families[family].pop(0))

        candidates = []
        seen: set[tuple[str, Program]] = set()
        if preserve_parameters:
            # Give different complete theories a turn before parameter aliases
            # of the same shape. No more than eight programs and one quarter
            # of the remaining fit budget are spent on preserving parameters.
            parameter_budget = SearchBudget(min(8 * len(observations), budget.remaining // 4))
            max_observed_history = max(len(row.history) for row in observations)
            shapes = sorted(ordered, key=lambda shape: (
                groups[shape][0][0].kind != "theory",
                groups[shape][0][0].pattern.history_depth <= max_observed_history))
            queues = {shape: list(groups[shape]) for shape in shapes}
            measured: dict[Program, tuple[float, ...]] = {}
            preserved = []
            while any(queues.values()) and parameter_budget.remaining >= len(observations):
                for shape in shapes:
                    if not queues[shape]:
                        continue
                    template, mapping = queues[shape].pop(0)
                    program = template.instantiate(mapping)
                    if self.transfer_refuted(spec, program):
                        continue
                    if program not in measured:
                        if not parameter_budget.spend(len(observations)):
                            break
                        measured[program] = trajectory_losses(program, observations, fallback, spec.risk_quantile)
                        work[shape] = work.get(shape, 0) + len(observations)
                        preserved_work[program] = {"shape": shape, "compute_cost": len(observations), "source_ids": []}
                    if template.source_concept_id not in preserved_work[program]["source_ids"]:
                        preserved_work[program]["source_ids"].append(template.source_concept_id)
                    losses = measured[program]
                    preserved.append((sum(losses) / len(losses), max(losses), template, mapping, program))
            budget.spend(parameter_budget.spent)
            preserved.sort(key=lambda item: (item[0], item[1], item[2].kind != "theory",
                                             priority(item[2], item[3])))
            for _, _, template, mapping, program in preserved:
                key = (template.source_concept_id, program)
                if key in seen:
                    continue
                seen.add(key)
                candidate = Candidate(
                    f"{template.template_id}:adapt:{self._search_serial}:preserved:{len(candidates)}",
                    program, "abstract_reuse",
                    f"Preserve learned parameters of {template.kind}; roles {mapping}; test on new observations",
                    (template.source_concept_id, template.template_id), template.source_concept_id)
                candidates.append(candidate)
                self._pending_transfers[candidate.candidate_id] = {
                    "template_id": template.template_id, "mapping": dict(mapping),
                    "parameter_mode": "preserved", "condition": condition, "mapping_roles": mapping_roles(mapping),
                    "work_id": (self._search_serial, fitting_shape(template, mapping))}
                if len(candidates) == 4:
                    break
            self.last_search.update(parameter_compute=parameter_budget.spent,
                                    parameter_trials=len(measured),
                                    preserved_parameter_candidates=len(candidates))
        before_screen = budget.spent
        screen_budget = SearchBudget(max(0, budget.remaining * 2 // 3))
        screened = []
        for shape in ordered:
            template, mapping = groups[shape][0]
            before = screen_budget.spent
            result = screen_template(template, mapping, observations, screen_budget, spec.risk_quantile)
            work[shape] = work.get(shape, 0) + screen_budget.spent - before
            if result is not None:
                screened.append((shape, result))
            if screen_budget.remaining < len(observations):
                break
        budget.spend(screen_budget.spent)
        self.last_search["screen_compute"] = budget.spent - before_screen + len(observations)
        self.last_search["screened_shapes"] = len(screened)
        screened.sort(key=lambda item: (item[1]["loss"], item[1]["worst_trajectory_loss"],
                                        priority(*groups[item[0]][0])))
        shortlist = list(screened[:4])
        retained_depths = {item[1]["history_depth"] for item in shortlist}
        for item in screened:
            if item[1]["history_depth"] not in retained_depths:
                shortlist.append(item)
                retained_depths.add(item[1]["history_depth"])
        self.last_search["ranked_shapes"] = [
            {"source_ids": list(dict.fromkeys(template.source_concept_id for template, _ in groups[shape])),
             "template_ids": [template.template_id for template, _ in groups[shape]],
             "mapping": groups[shape][0][1], "loss": result["loss"],
             "trajectory_losses": list(result["trajectory_losses"]),
             "history_depth": result["history_depth"], "fitted_rows": result["fitted_rows"],
             "screen_expression": result["program"].describe()}
            for shape, result in screened]

        before_fit = budget.spent
        exact_training_fit = bool(screened and screened[0][1]["loss"] <= max(1e-12, baseline * 1e-8))
        for index, (shape, result) in enumerate(shortlist):
            template, mapping = groups[shape][0]
            programs = [result["program"]]
            # An exact structural fit needs no additional parameter search.
            if not exact_training_fit and result["loss"] > max(1e-12, baseline * 1e-8):
                local_budget = SearchBudget(budget.remaining // max(1, len(shortlist) - index))
                details: dict[str, Any] = {}
                fitted = fit_template(template, mapping, observations, local_budget,
                                      spec.risk_quantile, report=details)
                budget.spend(local_budget.spent)
                work[shape] = work.get(shape, 0) + local_budget.spent
                if local_budget.spent:
                    self.last_search["deep_fits"] += 1
                if fitted:
                    ranked = sorted(fitted, key=lambda program: (details["losses"][program], program.size))
                    if ranked[0] not in programs:
                        programs.append(ranked[0])
            for origin, origin_mapping in groups[shape]:
                for program in programs:
                    if self.transfer_refuted(spec, program):
                        continue
                    key = (origin.source_concept_id, program)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = Candidate(
                        f"{origin.template_id}:adapt:{self._search_serial}:fit:{len(candidates)}", program, "abstract_reuse",
                        f"Training-trajectory fit of {origin.kind}; roles {origin_mapping}; independent testing still required",
                        (origin.source_concept_id, origin.template_id), origin.source_concept_id)
                    candidates.append(candidate)
                    self._pending_transfers[candidate.candidate_id] = {
                        "template_id": origin.template_id, "mapping": dict(origin_mapping),
                        "parameter_mode": "refitted", "condition": condition, "mapping_roles": mapping_roles(origin_mapping),
                        "work_id": (self._search_serial, shape)}
        self.last_search.update(fit_compute=budget.spent - before_fit, candidates=len(candidates),
                                distinct_programs=len({candidate.program for candidate in candidates}),
                                compute_cost=budget.spent - started)
        # Screening and execution can belong to different restored sessions.
        # Carry only this fit group's actual work with its executable offer.
        for candidate in candidates:
            origin = self._pending_transfers[candidate.candidate_id]
            shape = origin["work_id"][1]
            origin["adaptation_work"] = [{
                "work_id": f"adapt:{spec.task_id}:{self._search_serial}:{ordered.index(shape)}",
                "compute_cost": work.get(shape, 0),
                "preserved_work": [{"compute_cost": item["compute_cost"], "source_ids": list(item["source_ids"])}
                                   for item in preserved_work.values() if item["shape"] == shape],
                "source_ids": list(dict.fromkeys(template.source_concept_id for template, _ in groups[shape])),
                "template_ids": list(dict.fromkeys(template.template_id for template, _ in groups[shape]))}]
        if self._transfer_episode is not None:
            for shape, cost in work.items():
                work_id = (self._search_serial, shape)
                self._transfer_episode["work"][work_id] = {
                    "compute_cost": cost,
                    "preserved_work": [{"compute_cost": item["compute_cost"], "source_ids": list(item["source_ids"])}
                                       for item in preserved_work.values() if item["shape"] == shape],
                    "source_ids": list(dict.fromkeys(template.source_concept_id for template, _ in groups[shape])),
                    "template_ids": list(dict.fromkeys(template.template_id for template, _ in groups[shape]))}
            # Keep unmeasured proposals visible, but key them by the same
            # executable program identity used when a comparison is recorded.
            # Otherwise a later measured standalone refit would be attached to
            # a legacy source/template bucket and lose its attribution scope.
            for candidate in candidates:
                origin = self._pending_transfers[candidate.candidate_id]
                source = candidate.memory_id
                if source is None:
                    continue
                canonical = candidate.program.expanded().simplified()
                canonical_key = json.dumps(canonical.to_dict(), sort_keys=True, separators=(",", ":"))
                joint_sources = tuple(sorted(set(origin.get("joint_source_ids", [source]))))
                key = (source, origin.get("template_id"),
                       tuple(sorted(origin.get("mapping_roles", {}).items())),
                       joint_sources, canonical_key)
                attempt = self._transfer_episode["attempts"].setdefault(key, {
                    "source_id": source, "template_id": origin.get("template_id"),
                    "condition": deepcopy(origin.get("condition", condition)),
                    "mapping": dict(origin.get("mapping", {})),
                    "mapping_roles": origin.get("mapping_roles", {}),
                    "work_ids": set(), "comparisons": {},
                    "lineage_id": origin.get("lineage_id"),
                    "joint_source_ids": list(joint_sources)})
                if origin.get("work_id") in self._transfer_episode["work"]:
                    attempt["work_ids"].add(origin["work_id"])
                for work in origin.get("adaptation_work", ()):
                    if work.get("work_id") in self._transfer_episode["work"]:
                        attempt["work_ids"].add(work["work_id"])
        return candidates

    def transfer_evidence(self, candidate: Candidate) -> list[dict]:
        """Export the measured preparation of an offer, without opening a visit."""
        origin = self._pending_transfers.get(candidate.candidate_id.removeprefix("startup:"))
        if origin is None or candidate.memory_id is None:
            return []
        return [{"source_id": candidate.memory_id, **{key: deepcopy(origin[key]) for key in (
            "template_id", "mapping", "mapping_roles", "condition", "adaptation_work")}}]

    def record_transfer_result(self, candidate: Candidate, spec: TaskSpec, *, accepted: bool,
                               loss: float, baseline_loss: float, evidence_rows: int = 0,
                               comparison_compute: int = 0, comparison_id=None,
                               observations: Sequence[Observation] | None = None,
                               origin: dict | None = None) -> None:
        """Remember an actually tested adaptation without validating its source law."""
        prepared_origin = origin is not None
        origin = (origin if origin is not None else
                  self._pending_transfers.get(candidate.candidate_id.removeprefix("startup:"), {}))
        episode = self._transfer_episode
        if episode is not None and episode["task_id"] == spec.task_id and candidate.memory_id:
            source = candidate.memory_id
            template_id = origin.get("template_id")
            key = (source, template_id, tuple(sorted(origin.get("mapping_roles", {}).items())))
            canonical = candidate.program.expanded().simplified()
            joint_sources = tuple(sorted(set(origin.get("joint_source_ids", [source]))))
            # A source/template can yield several standalone executable
            # programs during one visit (for example, two parameter refits).
            # Their measured outcomes are different trials and must not be
            # averaged into one source return. The same binding also remains
            # distinct when the call participates in a joint lineage.
            canonical_key = json.dumps(canonical.to_dict(), sort_keys=True, separators=(",", ":"))
            key = (*key, joint_sources, canonical_key)
            attempt = episode["attempts"].setdefault(key, {"source_id": source,
                "template_id": template_id, "condition": deepcopy(origin.get("condition", episode["condition"])),
                "mapping": dict(origin.get("mapping", {})), "mapping_roles": origin.get("mapping_roles", {}),
                "work_ids": set(), "comparisons": {},
                "lineage_id": origin.get("lineage_id"),
                "joint_source_ids": list(origin.get("joint_source_ids", [source]))})
            if origin.get("lineage_id") is not None:
                attempt["lineage_id"] = origin["lineage_id"]
                attempt["joint_source_ids"] = list(origin.get("joint_source_ids", [source]))
            if origin.get("work_id") in episode["work"]:
                attempt["work_ids"].add(origin["work_id"])
            for work in (origin.get("adaptation_work", []) if prepared_origin else ()):
                work_id = work["work_id"]
                episode["work"].setdefault(work_id, deepcopy(work))
                attempt["work_ids"].add(work_id)
            comparison_key = (comparison_id, canonical)
            attempt["comparisons"][comparison_key] = {
                "program": candidate.program.to_dict(),
                "loss": float(loss), "baseline_loss": float(baseline_loss), "accepted": bool(accepted),
                "evidence_rows": evidence_rows, "compute_cost": comparison_compute,
                "comparison_id": comparison_id}
        for template in self.abstracts:
            if template.template_id in candidate.parents:
                event = {
                    "task_id": spec.task_id, "accepted": bool(accepted),
                    "loss": float(loss), "baseline_loss": float(baseline_loss),
                    "mapping": dict(origin.get("mapping", {})),
                    "parameter_mode": origin.get("parameter_mode", "refitted"),
                }
                if episode is not None:
                    event.update(condition=deepcopy(origin.get("condition", episode["condition"])),
                                 mapping_roles=origin.get("mapping_roles", {}))
                template.transfer_history.append(event)

    def begin_transfer_episode(self, spec: TaskSpec, observations: Sequence[Observation]) -> None:
        """Start attribution of this visit's actual transfer fits and comparisons."""
        self._transfer_serial += 1
        self._transfer_episode = {"task_id": spec.task_id, "condition": transfer_condition(spec, observations),
                                  "episode_id": f"transfer:{self._transfer_serial}",
                                  "work": {}, "attempts": {}, "production_methods": {
                                      concept.concept_id: deepcopy(concept.production_methods) for concept in self.concepts}}

    def record_knowledge_production(self, concept_id: str, spec: TaskSpec, trace: Sequence[dict]) -> list[dict]:
        """Connect an actually formed executable concept to its measured research step."""
        concept = next(item for item in self.concepts if item.concept_id == concept_id)
        canonical = concept.program.expanded().simplified()
        for step in reversed(trace):
            if (step.get("gain", 0.) <= 0 or not step.get("execution_complete", True)
                    or step.get("action") in {"calibration", "decision_calibration"}
                    or not step.get("after_program")
                    or Program.from_dict(step["after_program"]).expanded().simplified() != canonical):
                continue
            identifier = step.get("method_id") or step.get("produced_method_id")
            method = next((item for item in self.methods.methods if item["method_id"] == identifier), None)
            if method is None or step["action"] not in method["steps"]:
                continue
            production = {"method_id": identifier, "source_concept_id": concept_id,
                "action": step["action"], "profile": deepcopy(step.get("profile")),
                "source_task": spec.task_id, "gain": step["gain"],
                "compute_cost": step["compute_cost"], "observation_cost": step["observation_cost"]}
            if production in concept.production_methods:
                return []
            concept.production_methods.append(production)
            return [deepcopy(production)]
        return []

    def finish_transfer_episode(self, spec: TaskSpec, *, final_candidate: Candidate,
                                final_loss: float | None, baseline_loss: float | None,
                                compute_cost: int, observation_cost: int, validation_rows: int = 0,
                                final_status: str | None = None) -> dict:
        """Save source-local returns; the full visit bill is contextual, never attributed."""
        episode = self._transfer_episode
        self._transfer_episode = None
        if episode is None or episode["task_id"] != spec.task_id:
            return {"returns": []}
        final_program = final_candidate.program.expanded().simplified()
        retained_programs = {final_program}

        def retain_calls(program):
            if program.op == "call":
                retained_programs.add(program.expanded().simplified())
            for child in program.children:
                retain_calls(child)

        retain_calls(final_candidate.program)
        records = []
        sources = {concept.concept_id: concept for concept in self.concepts}
        templates = {template.template_id: template for template in self.abstracts}
        # Attempts in one visit are alternatives on the same paid response.
        # They must not confirm each other merely because the first record has
        # already been appended to the source history.
        prior_histories = {source_id: deepcopy(source.reuse_history)
                           for source_id, source in sources.items()}
        for attempt in episode["attempts"].values():
            comparisons = list(attempt["comparisons"].values())
            surviving = [item for item in comparisons if (
                Program.from_dict(item["program"]).expanded().simplified() in retained_programs)]
            retained = bool(surviving)
            retained_program = (Program.from_dict(surviving[0]["program"]).expanded().simplified()
                                if surviving else None)
            rows = sum(max(1, item["evidence_rows"]) for item in comparisons)
            paired_gain = sum((item["baseline_loss"] - item["loss"]) * max(1, item["evidence_rows"])
                              for item in comparisons) / rows if rows else None
            # A proposal may beat the earlier incumbent yet lose to a better
            # explanation before the visit ends. Keep that paired result but
            # do not promote it to retained downstream value.
            normalized = sum((item["baseline_loss"] - item["loss"]) /
                max(abs(item["baseline_loss"]), abs(item["loss"]), 1e-8) * max(1, item["evidence_rows"])
                for item in comparisons) / rows if rows else 0.
            work_items = [item for identifier, item in episode["work"].items() if identifier in attempt["work_ids"]]
            comparison_cost = sum(item["compute_cost"] for item in comparisons)
            # A positive paired comparison from a successor that is still
            # unresolved is useful evidence, but it has not confirmed the
            # source as an executable reusable explanation. Keep that result
            # visible while preventing it from acting like confirmed value in
            # the source scheduler. Older callers omit ``final_status``;
            # positive records retain their historical confirmed semantics.
            # A candidate surviving as the final program is not by itself
            # transferable value: the fresh comparison must improve on the
            # incumbent. Keep neutral fits visible without letting them enter
            # the confirmed-return scheduler.
            # Do not turn floating-point residue into a reusable gain or a
            # negative source signal. The scheduler's meaningful gains are
            # much larger than this numerical resolution.
            value_epsilon = 1e-12
            value_status = ("positive" if retained and normalized > value_epsilon
                            else "negative" if normalized < -value_epsilon
                            else "neutral")
            confirmation_status = ("confirmed" if value_status == "positive"
                                    and final_status in (None, "completed")
                                    else "provisional" if value_status == "positive" and retained
                                    else "neutral" if value_status == "neutral"
                                    else "rejected")
            confirmation_replicates = 0
            signature = attempt["condition"].get("response_signature") or {}
            repeat_required = (value_status == "positive" and final_status == "completed"
                               and signature.get("residual_bin", 0) > 0)
            if repeat_required:
                scope_program = retained_program or final_program
                prior = [item for item in prior_histories.get(attempt["source_id"], [])
                         if item.get("kind") == "research_return"
                         and item.get("condition") == attempt["condition"]
                         and item.get("retained") and item.get("retained_normalized_gain", 0.) > value_epsilon
                         and return_scope_matches(item, program=scope_program,
                             source_ids=attempt.get("joint_source_ids", [attempt["source_id"]]))
                         and item.get("final_status") in (None, "completed")]
                unique_prior = []
                seen_episodes = set()
                for item in prior:
                    episode_id = item.get("episode_id")
                    if episode_id in seen_episodes:
                        continue
                    seen_episodes.add(episode_id)
                    unique_prior.append(item)
                confirmation_replicates = len(unique_prior) + 1
                confirmation_status = "confirmed" if confirmation_replicates >= 2 else "provisional"
                if confirmation_status == "confirmed":
                    for item in sources[attempt["source_id"]].reuse_history:
                        if any(item == previous for previous in unique_prior):
                            item["confirmation_status"] = "confirmed"
                            item["confirmation_replicates"] = confirmation_replicates
            elif value_status == "positive" and final_status in (None, "completed"):
                confirmation_replicates = 1
            record = {"kind": "research_return", "task_id": spec.task_id,
                "source_id": attempt["source_id"], "template_id": attempt["template_id"],
                "condition": attempt["condition"], "mapping": attempt["mapping"], "mapping_roles": attempt["mapping_roles"],
                "accepted": retained, "retained": retained, "paired_gain": paired_gain,
                "final_status": final_status,
                "value_status": value_status,
                "confirmation_status": confirmation_status,
                "confirmation_replicates": confirmation_replicates,
                "evidence_status": "measured" if comparisons else "proposal_not_compared",
                "retained_normalized_gain": (normalized if value_status == "positive"
                    else 0. if value_status == "neutral" else min(0., normalized)),
                "adaptation_compute": sum(item["compute_cost"] for item in work_items),
                "comparison_compute": comparison_cost, "shared_adaptation_work": work_items,
                "comparisons": comparisons,
                "lineage_id": attempt.get("lineage_id"),
                "joint_source_ids": list(attempt.get("joint_source_ids", [attempt["source_id"]])),
                "episode_id": episode["episode_id"],
                "retained_program": retained_program.to_dict() if retained_program is not None else None,
                "visit_context": {"compute_cost": compute_cost, "observation_cost": observation_cost,
                    "final_loss": final_loss, "baseline_loss": baseline_loss, "validation_rows": validation_rows}}
            if attempt["source_id"] in sources:
                sources[attempt["source_id"]].reuse_history.append(deepcopy(record))
            if attempt["template_id"] in templates:
                templates[attempt["template_id"]].transfer_history.append(deepcopy(record))
            records.append(record)
        joint_returns = []
        grouped = {}
        for record in records:
            lineage_id = record.get("lineage_id")
            if lineage_id is not None and record.get("comparisons"):
                canonical = Program.from_dict(record["comparisons"][0]["program"]).expanded().simplified()
                grouped.setdefault((lineage_id, canonical), []).append(record)
        for (lineage_id, canonical), group in grouped.items():
            comparisons = {}
            work = {}
            for record in group:
                for comparison in record["comparisons"]:
                    key = (comparison.get("comparison_id"),
                           Program.from_dict(comparison["program"]).expanded().simplified())
                    comparisons[key] = comparison
                for item in record["shared_adaptation_work"]:
                    work[item["work_id"]] = item
            comparison_rows = list(comparisons.values())
            rows = sum(max(1, item["evidence_rows"]) for item in comparison_rows)
            gain = (sum((item["baseline_loss"] - item["loss"]) * max(1, item["evidence_rows"])
                        for item in comparison_rows) / rows if rows else None)
            normalized_gain = (sum((item["baseline_loss"] - item["loss"]) /
                max(abs(item["baseline_loss"]), abs(item["loss"]), 1e-8) * max(1, item["evidence_rows"])
                for item in comparison_rows) / rows if rows else 0.)
            joint_returns.append({"kind": "joint_research_return", "task_id": spec.task_id,
                "lineage_id": lineage_id,
                "program": canonical.to_dict(),
                "source_ids": sorted({source for record in group
                    for source in record.get("joint_source_ids", [record["source_id"]])}),
                "retained": any(record["retained"] for record in group),
                "paired_gain": gain, "retained_normalized_gain": (normalized_gain
                    if any(record["retained"] for record in group) else min(0., normalized_gain)),
                "adaptation_compute": sum(item["compute_cost"] for item in work.values()),
                "comparison_compute": sum(item["compute_cost"] for item in comparison_rows),
                "comparisons": comparison_rows,
                "source_local_records": [record["source_id"] for record in group],
                "shared_cost_accounted_once": True})
        from .knowledge_returns import producer_outcomes
        method_returns = []
        for source_id, outcome in producer_outcomes(episode, retained_programs):
            source = sources.get(source_id)
            if source is None:
                continue
            for production in episode["production_methods"].get(source_id, []):
                if production.get("kind") == "research_workflow":
                    # A reopened visit can share the producer's public task
                    # label. The episode snapshot already excludes links
                    # formed by the current consumer itself.
                    measured = self.research_workflows.observe_knowledge_transfer(production,
                        {**outcome, "episode_id": episode["episode_id"]},
                        return_history=source.reuse_history)
                else:
                    if production["source_task"] == spec.task_id:
                        continue
                    measured = self.methods.observe_knowledge_transfer(production, outcome)
                if measured is not None:
                    method_returns.append(measured)
        return {"returns": records, "joint_returns": joint_returns,
                "unique_adaptation_compute": sum(
                    item["compute_cost"] for item in episode["work"].values()),
                "method_returns": method_returns,
                "comparison_compute_is_shared_by_equal_program_origins": True,
                "episode_cost_is_context_only": True}

    def record_counterexample(self, concept_id: str, spec: TaskSpec, old_loss: float,
                              replacement_loss: float, example: Observation | dict[str, Any], *,
                              budget: SearchBudget | None = None,
                              supporting_observations: Sequence[Observation] = ()) -> dict[str, Any]:
        concept = next(c for c in self.concepts if c.concept_id == concept_id)
        row = _as_observation(example)
        local = self._scope_for(concept, spec)
        same_context = spec.task_id in concept.successful_tasks or (
            local is not None and bool(spec.context.get("system_id"))
            and local.system_id == spec.context["system_id"] and local.verified)
        inside = bool(local and local.supports(row))
        # Earlier saved contexts have no measured intervals. Preserve their
        # supported-task behavior until real rows for that context are provided.
        if same_context and local is None:
            inside = True
        event = {"task_id": spec.task_id, "concept_loss": old_loss,
                 "replacement_loss": replacement_loss, "observation": _row_payload(row),
                 "same_supported_context": same_context, "inside_observed_support": inside}
        concept.counterexamples.append(event)
        if concept.status != "revoked" and same_context and local and local.predicate and not local.matches_conditions(row):
            # A fallback error outside an already learned validity condition
            # does not refute the source law on the side where it still applies.
            return {"action": concept.status, "scope_change": "outside_learned_condition",
                    "status": concept.status, "concept_id": concept_id,
                    "condition_compute_cost": 0, **event}
        if (concept.status != "revoked" and same_context and local is not None
                and local.support_examples and spec.risk_quantile == 0.5 and spec.decision_costs is None
                and max([len(row.history), *(len(observed.history) for observed in supporting_observations)])
                    > max(len(item.get("history", ())) for item in local.support_examples)):
            # A short reset trajectory can support a law which fails only once
            # more history is available. Learn this observable condition from
            # measured successes/failures, even outside the old feature box.
            allowance = budget if budget is not None else SearchBudget(2000)
            history_start = allowance.spent
            evidence = [*[_as_observation(item) for item in local.support_examples],
                        *[_as_observation(item) for item in local.counter_examples],
                        *supporting_observations, row]
            if allowance.spend(len(evidence)):
                labels = [(observed, prediction_matches(concept.program, observed)) for observed in evidence]
                positives = [observed for observed, correct in labels if correct]
                negatives = [observed for observed, correct in labels if not correct]
                if (positives and negatives and max(len(observed.history) for observed in positives)
                        < min(len(observed.history) for observed in negatives)):
                    boundary = min(len(observed.history) for observed in negatives)
                    condition = Program("lt", children=(Program("available", lag=boundary),
                                                       Program("const", value=1.)))
                    local.predicate = condition.to_dict()
                    local.support_examples = [_row_payload(observed) for observed in positives]
                    local.counter_examples = [_row_payload(observed) for observed in negatives]
                    concept.status = "restricted"
                    self._refresh_condition_templates(concept, local)
                    event.update(condition_fit={"condition": condition.to_dict(), "status": "separated_history",
                        "support_rows": len(positives), "counterexample_rows": len(negatives),
                        "maximum_supported_history": max(len(observed.history) for observed in positives),
                        "first_counterexample_history": boundary}, condition_compute_cost=allowance.spent - history_start)
                    return {"action": "restricted", "scope_change": "restricted_history",
                            "status": concept.status, "concept_id": concept_id, **event}
            event["condition_compute_cost"] = allowance.spent - history_start
        if (concept.status != "revoked" and same_context and inside and local is not None
                and spec.risk_quantile == 0.5 and spec.decision_costs is None):
            allowance = budget if budget is not None else SearchBudget(2000)
            condition_start = allowance.spent
            positives = list(local.support_examples)
            negatives = list(local.counter_examples)
            current_rows = [*supporting_observations, row]
            classified = allowance.spend(len(current_rows))
            if classified:
                for observed in current_rows:
                    payload = _row_payload(observed)
                    bucket = positives if prediction_matches(concept.program, observed) else negatives
                    if payload not in bucket:
                        bucket.append(payload)
            local.counter_examples = negatives
            local.support_examples = positives
            learned = (learn_condition([_as_observation(item) for item in positives],
                                       [_as_observation(item) for item in negatives], allowance)
                       if classified else {"condition": None, "status": "insufficient_budget", "compute_cost": 0})
            event["condition_fit"] = learned
            event["condition_compute_cost"] = event.get("condition_compute_cost", 0) + allowance.spent - condition_start
            if learned["condition"] is not None:
                local.predicate = learned["condition"]
                # The exception estimate is only a candidate. Direct retrieval
                # retains the old fallback and claims support on the valid side.
                if allowance.spend(len(negatives)):
                    local.exception_program = Program("const", value=sum(item["target"] for item in negatives) / len(negatives)).to_dict()
                    event["condition_compute_cost"] += len(negatives)
                concept.status = "restricted"
                self._refresh_condition_templates(concept, local)
                return {"action": "restricted", "scope_change": "restricted_condition",
                        "status": concept.status, "concept_id": concept_id, **event}
        if concept.status == "revoked" or (same_context and inside):
            concept.status = "revoked"
            for template in self.abstracts:
                if template.source_concept_id == concept_id:
                    template.status = "revoked"
            action = "revoked"
        else:
            concept.status = "restricted"
            reference = local if local and local.verified else next((scope for scope in concept.scopes if scope.verified), None)
            separations = []
            if reference:
                for axis in reference.support:
                    value = _read(row, axis["name"], axis["lag"])
                    if value is None:
                        continue
                    width = max(axis["upper"] - axis["lower"], 1e-9)
                    if value < axis["lower"]:
                        separations.append(((axis["lower"] - value) / width,
                                            {"name": axis["name"], "lag": axis["lag"],
                                             "lower": (value + axis["lower"]) / 2, "upper": None}))
                    elif value > axis["upper"]:
                        separations.append(((value - axis["upper"]) / width,
                                            {"name": axis["name"], "lag": axis["lag"],
                                             "lower": None, "upper": (value + axis["upper"]) / 2}))
            if separations:
                boundary = max(separations, key=lambda item: item[0])[1]
                if local is None or local.task_id != spec.task_id:
                    local = EmpiricalScope(spec.task_id, spec.context.get("system_id"),
                                           fallback=reference.fallback, verified=False)
                    concept.scopes.append(local)
                local.conditions.append(boundary)
                event["learned_boundary"] = boundary
                self._refresh_condition_templates(concept, local)
                action = "restricted_interval"
            else:
                # Overlapping measurements in a different system provide no
                # observable separator. Keep that contextual conflict explicit.
                if spec.task_id not in concept.excluded_tasks:
                    concept.excluded_tasks.append(spec.task_id)
                action = "restricted_context"
        return {"action": concept.status, "scope_change": action,
                "status": concept.status, "concept_id": concept_id, **event}

    def _refresh_condition_templates(self, concept: Concept, scope: EmpiricalScope) -> None:
        prefix = f"{concept.concept_id}:condition:{scope.task_id}:"
        self.abstracts = [template for template in self.abstracts if not template.template_id.startswith(prefix)]
        if scope.predicate is None:
            # Interval restrictions already governed direct retrieval in 0.9.
            # Their scoped call definitions must not enlarge the old refit pool.
            self._refresh_definitions()
            return
        guarded = _guard(concept.program, scope)
        programs = [("supported", guarded)]
        if scope.exception_program is not None:
            programs.append(("repair", Program("if", children=(guarded.children[0], concept.program,
                                                               Program.from_dict(scope.exception_program)))))
        for kind, program in programs:
            template = extract_templates(program, prefix + kind)[0]
            template.source_concept_id = concept.concept_id
            template.parents = [concept.concept_id]
            self.abstracts.append(template)
        self._refresh_definitions()

    def record_use(self, concept_id: str, task_id: str, *, spec: TaskSpec | None = None,
                   observations: Sequence[Observation] | None = None) -> None:
        concept = next(c for c in self.concepts if c.concept_id == concept_id)
        if task_id not in concept.successful_tasks:
            concept.successful_tasks.append(task_id)
        concept.reuse_history.append({"task_id": task_id, "kind": "executed_program"})
        if spec is not None:
            self._record_scope(concept, spec, observations)
            local = self._scope_for(concept, spec)
            if local and (local.conditions or local.predicate):
                self._refresh_condition_templates(concept, local)

    def learn(self, candidate: Candidate, spec: TaskSpec,
              observations: Sequence[Observation] | None = None) -> tuple[str, bool]:
        candidate = replace(candidate, program=candidate.program.simplified())
        equivalent = candidate.program.expanded().simplified()
        for concept in self.concepts:
            if (concept.status != "revoked" and concept.program.expanded().simplified() == equivalent
                    and concept.risk_quantile == spec.risk_quantile):
                normalized = concept.program.simplified()
                if normalized != concept.program:
                    concept.program = normalized
                    previous = [template for template in self.abstracts if template.source_concept_id == concept.concept_id]
                    replacements = extract_templates(normalized, concept.concept_id, object_schema=_object_schema(spec))
                    for template in replacements:
                        old = next((item for item in previous if item.pattern.expanded().simplified() == template.pattern), None)
                        if old is not None:
                            template.uses = list(old.uses)
                            template.transfer_history = deepcopy(old.transfer_history)
                            template.status = old.status
                    self.abstracts = [template for template in self.abstracts if template.source_concept_id != concept.concept_id]
                    self.abstracts.extend(replacements)
                    self._refresh_definitions()
                self.record_use(concept.concept_id, spec.task_id, spec=spec, observations=observations)
                self._record_abstract_use(candidate, spec.task_id)
                return concept.concept_id, False
        concept_id = f"concept_{self.next_concept_index:04d}"
        self.next_concept_index += 1
        called_sources = [definition.source_concept_id for definition in _called_definitions(candidate.program)
                          if definition.source_concept_id is not None]
        parents = list(dict.fromkeys((*candidate.parents, *([candidate.memory_id] if candidate.memory_id else []),
                                     *called_sources)))
        generation = 1 + max((concept.generation for concept in self.concepts if concept.concept_id in parents), default=-1)
        concept = Concept(concept_id, candidate.program, spec.risk_quantile,
                          spec.task_id, candidate.operation, successful_tasks=[spec.task_id],
                          parents=parents, generation=generation)
        self._record_scope(concept, spec, observations)
        self.concepts.append(concept)
        self.abstracts.extend(extract_templates(candidate.program, concept_id, object_schema=_object_schema(spec)))
        self._refresh_definitions()
        self._record_abstract_use(candidate, spec.task_id)
        return concept_id, True

    @staticmethod
    def _record_scope(concept: Concept, spec: TaskSpec,
                      observations: Sequence[Observation] | None) -> None:
        if not observations:
            return
        applicable = KnowledgeLibrary._scope_for(concept, spec)
        supported = [row for row in observations if applicable.matches_conditions(row)] if applicable else list(observations)
        if not supported:
            return
        measured = EmpiricalScope.measured(spec, concept.program, supported)
        if applicable is not None:
            measured.conditions = [dict(condition) for condition in applicable.conditions]
            measured.predicate = deepcopy(applicable.predicate)
            measured.exception_program = deepcopy(applicable.exception_program)
            measured.counter_examples = deepcopy(applicable.counter_examples)
        existing = next((scope for scope in concept.scopes if scope.task_id == spec.task_id), None)
        if existing:
            previous = {(axis["name"], axis["lag"]): axis for axis in existing.support}
            for axis in measured.support:
                old_axis = previous.get((axis["name"], axis["lag"]))
                if old_axis is not None:
                    axis["lower"] = min(axis["lower"], old_axis["lower"])
                    axis["upper"] = max(axis["upper"], old_axis["upper"])
            total = measured.observed_rows + existing.observed_rows
            measured.fallback = (measured.fallback * measured.observed_rows + existing.fallback * existing.observed_rows) / total
            measured.observed_rows = total
            measured.support_examples = [*deepcopy(existing.support_examples),
                                         *[row for row in measured.support_examples if row not in existing.support_examples]]
            concept.scopes.remove(existing)
        concept.scopes.append(measured)

    def _record_abstract_use(self, candidate: Candidate, task_id: str) -> None:
        for template in self.abstracts:
            if template.template_id in candidate.parents:
                template.uses.append(task_id)
                source = next(c for c in self.concepts if c.concept_id == template.source_concept_id)
                source.reuse_history.append({"task_id": task_id, "kind": "refitted_abstraction",
                                             "template_id": template.template_id})

    def without(self, concept_ids: set[str]) -> "KnowledgeLibrary":
        removed = set(concept_ids)
        while True:
            descendants = {concept.concept_id for concept in self.concepts if removed.intersection(concept.parents)}
            descendants.update(template.template_id for template in self.abstracts
                               if template.source_concept_id in removed or removed.intersection(template.parents))
            if descendants.issubset(removed):
                break
            removed.update(descendants)
        measurements = []
        for record in self.measurements.to_dict():
            record["source_ids"] = [source for source in record["source_ids"] if source not in removed]
            record["successful"] = bool(record["successful"] and record["source_ids"])
            if record["source_ids"] or record["conflicts"]:
                measurements.append(record)
        return self.from_dict({
            **self.to_dict(),
            "concepts": [c.to_dict() for c in self.concepts if c.concept_id not in removed],
            "abstracts": [a.to_dict() for a in self.abstracts if a.template_id not in removed],
            "next_concept_index": self.next_concept_index,
            "deleted_ids": list(dict.fromkeys([*self.deleted_ids, *sorted(removed)])),
            "research_experience": self.research_experience,
            "action_experience": self.action_experience,
            "experiments": [record for record in self.experiments.to_dict()
                            if not removed.intersection(record["source_concepts"])],
            "methods": self.methods.to_dict(),
            "measurements": measurements,
            "meta_priors": [prior for prior in self.meta_priors
                            if prior["source_concept_id"] not in removed
                            and not removed.intersection(prior.get("parent_concept_ids", ()))],
            "program_experiment_methods": self.program_experiment_methods.to_dict(),
            "research_drafts": [record for record in self.research_drafts
                                if not removed.intersection(record["source_ids"])],
            "condition_coverages": [record for record in self.condition_coverages
                                    if not removed.intersection(record.get("source_ids", ()))],
            "instrument_models": self.instrument_models,
            "object_models": self.object_models,
            "structural_priors": deepcopy(self.structural_priors),
            "evolution": {"merges": [record for record in self.evolution.to_dict()["merges"]
                if not removed.intersection(definition.source_concept_id for definition in
                    _called_definitions(Program.from_dict(record["original"])))]},
            "definitions": [entry for entry in self._definition_records()
                            if entry["definition"]["source_concept_id"] not in removed
                            and entry["template_id"] not in removed],
        })

    def _definition_records(self) -> list[dict]:
        return [{"definition": definition.to_dict(),
                 "template_id": self.definition_bindings[definition_id]["template_id"],
                 "arguments": [argument.to_dict() for argument in self.definition_bindings[definition_id]["arguments"]]}
                for definition_id, definition in self.definitions.items()]

    def supporting_observations(self, program: Program, spec: TaskSpec) -> list[Observation]:
        """Recall successful full-model evidence in this task's observed system."""
        source = program.definition.source_concept_id if program.op == "call" else None
        result = []
        for concept in self.concepts:
            if concept.status == "revoked" or (concept.program != program and concept.concept_id != source):
                continue
            if concept.program != program and concept.program.expanded() != program.expanded():
                continue
            for scope in concept.scopes:
                same_system = scope.system_id is not None and scope.system_id == spec.context.get("system_id")
                if scope.verified and (scope.task_id == spec.task_id or same_system):
                    result.extend(_as_observation(row) for row in scope.support_examples)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"concepts": [c.to_dict() for c in self.concepts],
                "abstracts": [a.to_dict() for a in self.abstracts],
                "next_concept_index": self.next_concept_index, "deleted_ids": list(self.deleted_ids),
                "research_experience": deepcopy(self.research_experience),
                "action_experience": deepcopy(self.action_experience),
                "experiments": self.experiments.to_dict(), "methods": self.methods.to_dict(),
                "research_workflows": self.research_workflows.to_dict(),
                "measurements": self.measurements.to_dict(), "meta_priors": deepcopy(self.meta_priors),
                "instrument_models": deepcopy(self.instrument_models),
                "object_models": deepcopy(self.object_models),
                "research_drafts": deepcopy(self.research_drafts),
                "program_experiment_methods": self.program_experiment_methods.to_dict(),
                "evolution": self.evolution.to_dict(), "structural_priors": deepcopy(self.structural_priors),
                "research_opportunities": self.research_opportunities.to_dict(),
                "definitions": self._definition_records(),
                "transfer_serial": self._transfer_serial,
                "condition_coverages": deepcopy(self.condition_coverages)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeLibrary":
        payload = deepcopy(payload)
        concepts = [Concept(**{**row, "program": Program.from_dict(row["program"]),
                               "scopes": [EmpiricalScope(**scope) for scope in row.get("scopes", [])]})
                    for row in payload.get("concepts", [])]
        abstracts = [AbstractTemplate.from_dict(row) for row in payload["abstracts"]] if "abstracts" in payload else None
        return cls(concepts, abstracts, next_concept_index=payload.get("next_concept_index"),
                   deleted_ids=payload.get("deleted_ids", []),
                   research_experience=payload.get("research_experience", {}),
                   action_experience=payload.get("action_experience", {}),
                   experiments=payload.get("experiments", []), methods=payload.get("methods"),
                   evolution=payload.get("evolution"), structural_priors=payload.get("structural_priors"),
                   definitions=payload.get("definitions"), research_opportunities=payload.get("research_opportunities"),
                   measurements=payload.get("measurements", []), meta_priors=payload.get("meta_priors", []),
                   instrument_models=payload.get("instrument_models", []),
                   object_models=payload.get("object_models", []),
                   research_drafts=payload.get("research_drafts", []),
                   program_experiment_methods=payload.get("program_experiment_methods", []),
                   transfer_serial=payload.get("transfer_serial"),
                   condition_coverages=payload.get("condition_coverages", []),
                   research_workflows=payload.get("research_workflows"))
