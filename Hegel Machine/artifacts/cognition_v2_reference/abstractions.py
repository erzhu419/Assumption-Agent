"""Reusable executable structures extracted from learned expressions.

An abstraction stores a tree, feature roles and constant slots. Instantiation
expands into the ordinary Program language, so a transferred concept can be
executed, compared and removed without a separate interpreter or hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Iterator, Sequence

from .language import Program
from .models import Observation, SearchBudget
from .synthesis import _fit_linear, _loss, _solve


def _constant_paths(program: Program, path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    if program.op == "const":
        return [path]
    return [found for index, child in enumerate(program.children)
            for found in _constant_paths(child, path + (index,))]


def _at(program: Program, path: tuple[int, ...]) -> Program:
    for index in path:
        program = program.children[index]
    return program


def _bind_constants(program: Program, bindings: dict[tuple[int, ...], float],
                    path: tuple[int, ...] = ()) -> Program:
    if path in bindings:
        return Program("const", value=float(bindings[path]))
    return Program(program.op, program.value, program.name, program.lag,
                   tuple(_bind_constants(child, bindings, path + (index,))
                         for index, child in enumerate(program.children)))


def _parameter_degree(program: Program) -> int:
    if program.op == "const":
        return 1
    if not program.children:
        return 0
    degrees = [_parameter_degree(child) for child in program.children]
    if program.op in {"add", "sub"}:
        return max(degrees)
    if program.op == "mul":
        return sum(degrees)
    return 2 if any(degrees) else 0


@dataclass
class AbstractTemplate:
    template_id: str
    source_concept_id: str
    pattern: Program
    roles: tuple[str, ...]
    source_features: tuple[str, ...]
    parameter_paths: tuple[tuple[int, ...], ...]
    kind: str = "theory"
    status: str = "available"
    parents: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)

    @property
    def parameters(self) -> tuple[float, ...]:
        return tuple(float(_at(self.pattern, path).value) for path in self.parameter_paths)

    def instantiate(self, mapping: dict[str, str],
                    parameters: Sequence[float] | None = None) -> Program:
        values = self.parameters if parameters is None else tuple(parameters)
        if len(values) != len(self.parameter_paths):
            raise ValueError("template parameter count differs")
        return _bind_constants(self.pattern, dict(zip(self.parameter_paths, values))).remap_features(mapping)

    def mappings(self, feature_names: Sequence[str], *, structural_transfer: bool = True) -> Iterator[dict[str, str]]:
        identity = dict(zip(self.roles, self.source_features))
        emitted_identity = set(self.source_features).issubset(feature_names)
        if emitted_identity:
            yield identity
        if not structural_transfer:
            return
        for names in permutations(feature_names, len(self.roles)):
            mapping = dict(zip(self.roles, names))
            if not emitted_identity or mapping != identity:
                yield mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id, "source_concept_id": self.source_concept_id,
            "pattern": self.pattern.to_dict(), "roles": list(self.roles),
            "source_features": list(self.source_features),
            "parameter_paths": [list(path) for path in self.parameter_paths],
            "kind": self.kind, "status": self.status, "parents": list(self.parents),
            "uses": list(self.uses),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AbstractTemplate":
        return cls(
            template_id=payload["template_id"], source_concept_id=payload["source_concept_id"],
            pattern=Program.from_dict(payload["pattern"]), roles=tuple(payload["roles"]),
            source_features=tuple(payload["source_features"]),
            parameter_paths=tuple(tuple(path) for path in payload["parameter_paths"]),
            kind=payload.get("kind", "theory"), status=payload.get("status", "available"),
            parents=list(payload.get("parents", [])), uses=list(payload.get("uses", [])),
        )


def extract_templates(program: Program, concept_id: str) -> list[AbstractTemplate]:
    subexpressions: list[Program] = []

    def visit(node: Program) -> None:
        # Retain useful computation, not every scalar multiplication in an
        # affine fit. History itself is a reusable observation transformation.
        nonlinear = node.op in {"abs", "ge", "lt", "and", "or", "if", "min", "max"}
        interaction = node.op == "mul" and all(child.observed_variables() for child in node.children)
        if (nonlinear or interaction or node.op == "lag") and node != program and node not in subexpressions:
            subexpressions.append(node)
        for child in node.children:
            visit(child)

    visit(program)
    templates = []
    for index, expression in enumerate([program, *subexpressions]):
        names = tuple(sorted({name for name, _ in expression.observed_variables() if name != "__target__"}))
        if not expression.observed_variables():
            continue
        roles = tuple(f"role_{index}" for index in range(len(names)))
        pattern = expression.remap_features(dict(zip(names, roles)))
        templates.append(AbstractTemplate(
            f"{concept_id}:abstract:{index}", concept_id, pattern, roles, names,
            tuple(_constant_paths(pattern)), "theory" if index == 0 else "subexpression",
            parents=[concept_id],
        ))
    return templates


def fit_template(template: AbstractTemplate, mapping: dict[str, str],
                 observations: Sequence[Observation], budget: SearchBudget,
                 risk_quantile: float = 0.5) -> list[Program]:
    """Fit constant slots and optional output calibration using only given rows.

    Basis evaluations and every threshold/parameter trial consume candidate ×
    row units, including trials that are discarded. No label outside this call
    participates in fitting or parameter selection.
    """
    initial = template.instantiate(mapping)
    variables = initial.observed_variables()
    rows = [row for row in observations
            if len(row.history) >= initial.history_depth
            and all(name in (row.features if lag == 0 else row.history[-lag])
                    for name, lag in variables)]
    if not rows:
        return []
    count = len(rows)
    programs: list[Program] = []

    def evaluate(program: Program) -> float | None:
        if not budget.spend(count):
            return None
        return _loss(program, rows, risk_quantile)

    best_loss = evaluate(initial)
    if best_loss is None:
        return []
    programs.append(initial)
    best = initial
    parameters = list(template.parameters)
    if parameters and _parameter_degree(template.pattern) <= 1 and budget.remaining >= (len(parameters) + 2) * count:
        base = template.instantiate(mapping, [0.0] * len(parameters))
        budget.spend(count)
        base_values = [base.predict(row) for row in rows]
        columns = []
        for index in range(len(parameters)):
            unit = [0.0] * len(parameters)
            unit[index] = 1.0
            basis = template.instantiate(mapping, unit)
            budget.spend(count)
            columns.append([basis.predict(row) - value for row, value in zip(rows, base_values)])
        gram = [[sum(a * b for a, b in zip(left, right)) for right in columns] for left in columns]
        for index in range(len(parameters)):
            gram[index][index] += 1e-9
        targets = [row.target - value for row, value in zip(rows, base_values)]
        fitted_parameters = _solve(gram, [sum(a * b for a, b in zip(column, targets)) for column in columns])
        fitted = template.instantiate(mapping, fitted_parameters)
        fitted_loss = evaluate(fitted)
        if fitted_loss is not None:
            programs.append(fitted)
            if fitted_loss < best_loss:
                best, best_loss, parameters = fitted, fitted_loss, fitted_parameters
    elif parameters:
        # Comparisons and conditional expressions need actual discrete trials.
        # Candidate thresholds come from current observed values of the operand.
        for _ in range(2):
            improved = False
            for index, path in enumerate(template.parameter_paths):
                values = [parameters[index], 0.0, 1.0, -1.0]
                if path:
                    parent = _at(template.pattern, path[:-1])
                    if parent.op in {"ge", "lt"}:
                        other = _at(template.instantiate(mapping, parameters), path[:-1]).children[1 - path[-1]]
                        if not budget.spend(count):
                            return programs
                        ordered = sorted({other.predict(row) for row in rows})
                        midpoints = [(left + right) / 2 for left, right in zip(ordered, ordered[1:])]
                        if len(midpoints) > 9:
                            midpoints = [midpoints[round(i * (len(midpoints) - 1) / 8)] for i in range(9)]
                        values.extend(midpoints)
                for value in dict.fromkeys(values):
                    trial_parameters = parameters[:]
                    trial_parameters[index] = value
                    trial = template.instantiate(mapping, trial_parameters)
                    trial_loss = evaluate(trial)
                    if trial_loss is None:
                        return programs
                    if trial_loss < best_loss:
                        best, best_loss, parameters = trial, trial_loss, trial_parameters
                        programs.append(best)
                        improved = True
            if not improved:
                break

    # A reusable nonlinear subexpression becomes an observed coordinate in an
    # ordinary new model. Its coefficients are refitted instead of copying old
    # predictions. Also permits a changed offset around a learned full theory.
    if budget.spend(count):
        calibrated = _fit_linear((best,), rows)
        if evaluate(calibrated) is not None and calibrated not in programs:
            programs.append(calibrated)
    if template.kind == "subexpression" and budget.spend(count):
        current = [Program("feature", name=name)
                   for name in sorted(set.intersection(*(set(row.features) for row in rows)))]
        combined = _fit_linear([*current, best], rows)
        if evaluate(combined) is not None and combined not in programs:
            programs.append(combined)
    return programs
