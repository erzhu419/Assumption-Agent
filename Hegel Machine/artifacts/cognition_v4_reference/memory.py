"""Executable knowledge, parameterized abstractions and empirical scope repair."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence

from .abstractions import (AbstractTemplate, extract_templates, fit_template,
                           fitting_shape, screen_template)
from .language import Program
from .models import Candidate, Observation, SearchBudget, TaskSpec


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
                   fallback=sum(row.target for row in observations) / len(observations) if observations else 0.0)

    def matches_conditions(self, row: Observation) -> bool:
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
                "verified": self.verified}


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

    def to_dict(self) -> dict[str, Any]:
        return {**deepcopy(self.__dict__), "program": self.program.to_dict(),
                "scopes": [scope.to_dict() for scope in self.scopes]}


def _guard(program: Program, scope: EmpiricalScope | None) -> Program:
    if scope is None or not scope.conditions:
        return program
    condition = None
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


class KnowledgeLibrary:
    def __init__(self, concepts: list[Concept] | None = None,
                 abstracts: list[AbstractTemplate] | None = None,
                 *, next_concept_index: int | None = None,
                 deleted_ids: list[str] | None = None):
        self.concepts = list(concepts or [])
        self.abstracts = list(abstracts) if abstracts is not None else [
            template for concept in self.concepts for template in extract_templates(concept.program, concept.concept_id)]
        highest = max((int(concept.concept_id.removeprefix("concept_")) for concept in self.concepts
                       if concept.concept_id.removeprefix("concept_").isdigit()), default=0)
        self.next_concept_index = next_concept_index if next_concept_index is not None else highest + 1
        self.deleted_ids = list(deleted_ids or [])
        self.last_search: dict[str, Any] = {}
        self._pending_transfers: dict[str, dict[str, Any]] = {}
        self._search_serial = 0

    @staticmethod
    def _scope_for(concept: Concept, spec: TaskSpec) -> EmpiricalScope | None:
        exact = next((scope for scope in concept.scopes if scope.task_id == spec.task_id), None)
        if exact is not None:
            return exact
        system_id = spec.context.get("system_id")
        return next((scope for scope in concept.scopes if system_id and scope.system_id == system_id), None)

    def retrieve(self, spec: TaskSpec) -> list[Candidate]:
        eligible = [c for c in self.concepts
                    if c.status != "revoked" and spec.task_id not in c.excluded_tasks
                    and c.risk_quantile == spec.risk_quantile
                    and {name for name, _ in c.program.observed_variables() if name != "__target__"}.issubset(spec.feature_names)]
        eligible.sort(key=lambda c: (
            spec.task_id not in c.successful_tasks and self._scope_for(c, spec) is None,
            -len(set(c.successful_tasks)), len(c.counterexamples), c.program.size, c.concept_id))
        return [Candidate(c.concept_id, _guard(c.program, self._scope_for(c, spec)), "reuse",
                          "Execute a previously learned program within its measured context", (c.concept_id,), c.concept_id)
                for c in eligible]

    @staticmethod
    def _transfer_priority(template: AbstractTemplate, mapping: dict[str, str]) -> tuple:
        matched = [event for event in template.transfer_history
                   if not event.get("mapping") or event["mapping"] == mapping]
        successes = {event["task_id"] for event in matched if event["accepted"]}
        failures = {event["task_id"] for event in matched if not event["accepted"]}
        return -len(successes | set(template.uses)), len(failures), template.pattern.size, template.template_id

    def propose(self, spec: TaskSpec, observations: Sequence[Observation], budget: SearchBudget,
                *, structural_transfer: bool = True) -> list[Candidate]:
        """Screen structures on complete training trajectories, then refit a few.

        Fitting is shared only by identical parameterized symbolic shapes in
        this call. Every emitted result retains its actual source and remains
        an adaptation proposal, including sources excluded from this context.
        """
        from .evaluation import initial_prediction, point_loss
        from .abstractions import _parameter_degree

        started = budget.spent
        self._search_serial += 1
        self.last_search = {"task_id": spec.task_id, "training_rows": len(observations),
                            "template_bindings": 0, "unique_fit_shapes": 0,
                            "screened_shapes": 0, "deep_fits": 0,
                            "shared_fit_bindings": 0, "screen_compute": 0,
                            "fit_compute": 0, "candidates": 0, "distinct_programs": 0,
                            "compute_cost": 0, "ranked_shapes": [],
                            "fit_cache_scope": "this_call_training_observations"}
        if not observations:
            return []
        sources = {concept.concept_id: concept for concept in self.concepts}
        groups: dict[tuple, list[tuple[AbstractTemplate, dict[str, str]]]] = {}
        for template in self.abstracts:
            source = sources.get(template.source_concept_id)
            if source is None or source.status == "revoked" or template.status == "revoked" or source.risk_quantile != spec.risk_quantile:
                continue
            for mapping in template.mappings(spec.feature_names, structural_transfer=structural_transfer):
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
            members.sort(key=lambda member: self._transfer_priority(*member))

        # A fair first pass across history depths and nonlinear structures keeps
        # old insertion order from consuming all work before temporal models.
        families: dict[tuple, list[tuple]] = {}
        for shape, members in groups.items():
            template, mapping = members[0]
            family = (template.pattern.history_depth, _parameter_degree(template.pattern) > 1)
            families.setdefault(family, []).append(shape)
        for family in families:
            families[family].sort(key=lambda shape: self._transfer_priority(*groups[shape][0]))
        ordered = []
        while any(families.values()):
            for family in sorted(families):
                if families[family]:
                    ordered.append(families[family].pop(0))
        screen_budget = SearchBudget(max(0, budget.remaining * 2 // 3))
        screened = []
        for shape in ordered:
            template, mapping = groups[shape][0]
            result = screen_template(template, mapping, observations, screen_budget, spec.risk_quantile)
            if result is not None:
                screened.append((shape, result))
            if screen_budget.remaining < len(observations):
                break
        budget.spend(screen_budget.spent)
        self.last_search["screen_compute"] = budget.spent - started
        self.last_search["screened_shapes"] = len(screened)
        screened.sort(key=lambda item: (item[1]["loss"], item[1]["worst_trajectory_loss"],
                                        self._transfer_priority(*groups[item[0]][0])))
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

        candidates = []
        seen: set[tuple[str, Program]] = set()
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
                if local_budget.spent:
                    self.last_search["deep_fits"] += 1
                if fitted:
                    ranked = sorted(fitted, key=lambda program: (details["losses"][program], program.size))
                    if ranked[0] not in programs:
                        programs.append(ranked[0])
            for origin, origin_mapping in groups[shape]:
                for program in programs:
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
                        "template_id": origin.template_id, "mapping": dict(origin_mapping)}
        self.last_search.update(fit_compute=budget.spent - before_fit, candidates=len(candidates),
                                distinct_programs=len({candidate.program for candidate in candidates}),
                                compute_cost=budget.spent - started)
        return candidates

    def record_transfer_result(self, candidate: Candidate, spec: TaskSpec, *, accepted: bool,
                               loss: float, baseline_loss: float) -> None:
        """Remember an actually tested adaptation without validating its source law."""
        origin = self._pending_transfers.get(candidate.candidate_id.removeprefix("startup:"), {})
        for template in self.abstracts:
            if template.template_id in candidate.parents:
                template.transfer_history.append({
                    "task_id": spec.task_id, "accepted": bool(accepted),
                    "loss": float(loss), "baseline_loss": float(baseline_loss),
                    "mapping": dict(origin.get("mapping", {})),
                })

    def record_counterexample(self, concept_id: str, spec: TaskSpec, old_loss: float,
                              replacement_loss: float, example: Observation | dict[str, Any]) -> dict[str, Any]:
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
                action = "restricted_interval"
            else:
                # Overlapping measurements in a different system provide no
                # observable separator. Keep that contextual conflict explicit.
                if spec.task_id not in concept.excluded_tasks:
                    concept.excluded_tasks.append(spec.task_id)
                action = "restricted_context"
        return {"action": concept.status, "scope_change": action,
                "status": concept.status, "concept_id": concept_id, **event}

    def record_use(self, concept_id: str, task_id: str, *, spec: TaskSpec | None = None,
                   observations: Sequence[Observation] | None = None) -> None:
        concept = next(c for c in self.concepts if c.concept_id == concept_id)
        if task_id not in concept.successful_tasks:
            concept.successful_tasks.append(task_id)
        concept.reuse_history.append({"task_id": task_id, "kind": "executed_program"})
        if spec is not None:
            self._record_scope(concept, spec, observations)

    def learn(self, candidate: Candidate, spec: TaskSpec,
              observations: Sequence[Observation] | None = None) -> tuple[str, bool]:
        for concept in self.concepts:
            if concept.status != "revoked" and concept.program == candidate.program and concept.risk_quantile == spec.risk_quantile:
                self.record_use(concept.concept_id, spec.task_id, spec=spec, observations=observations)
                self._record_abstract_use(candidate, spec.task_id)
                return concept.concept_id, False
        concept_id = f"concept_{self.next_concept_index:04d}"
        self.next_concept_index += 1
        parents = list(dict.fromkeys((*candidate.parents, *([candidate.memory_id] if candidate.memory_id else []))))
        generation = 1 + max((concept.generation for concept in self.concepts if concept.concept_id in parents), default=-1)
        concept = Concept(concept_id, candidate.program, spec.risk_quantile,
                          spec.task_id, candidate.operation, successful_tasks=[spec.task_id],
                          parents=parents, generation=generation)
        self._record_scope(concept, spec, observations)
        self.concepts.append(concept)
        self.abstracts.extend(extract_templates(candidate.program, concept_id))
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
        return self.from_dict({
            "concepts": [c.to_dict() for c in self.concepts if c.concept_id not in removed],
            "abstracts": [a.to_dict() for a in self.abstracts if a.template_id not in removed],
            "next_concept_index": self.next_concept_index,
            "deleted_ids": list(dict.fromkeys([*self.deleted_ids, *sorted(removed)])),
        })

    def to_dict(self) -> dict[str, Any]:
        return {"concepts": [c.to_dict() for c in self.concepts],
                "abstracts": [a.to_dict() for a in self.abstracts],
                "next_concept_index": self.next_concept_index, "deleted_ids": list(self.deleted_ids)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeLibrary":
        payload = deepcopy(payload)
        concepts = [Concept(**{**row, "program": Program.from_dict(row["program"]),
                               "scopes": [EmpiricalScope(**scope) for scope in row.get("scopes", [])]})
                    for row in payload.get("concepts", [])]
        abstracts = [AbstractTemplate.from_dict(row) for row in payload["abstracts"]] if "abstracts" in payload else None
        return cls(concepts, abstracts, next_concept_index=payload.get("next_concept_index"),
                   deleted_ids=payload.get("deleted_ids", []))
