"""Reusable executable structures extracted from learned expressions.

An abstraction stores a tree, feature roles and constant slots. Instantiation
expands into the ordinary Program language, so a transferred concept can be
executed, compared and removed without a separate interpreter or hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import permutations
from math import isfinite
from typing import Any, Iterator, Sequence

import numpy as np

from .language import Program
from .models import Observation, SearchBudget
from .parallel_prediction import predict_rows
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
                         for index, child in enumerate(program.children)), program.definition)


def _comparison_slot_uses(program, path):
    """Locate a lexical const slot used directly as a comparison operand."""
    if path and _at(program, path[:-1]).op in {'ge', 'lt'}:
        return [(False, path[:-1], 1 - path[-1])]
    node, through_call = program, False
    for index in path:
        through_call = through_call or node.op == 'call'
        node = node.children[index]
    if not through_call:
        return []
    marker = Program('arg', name='selected_fitting_slot')

    def mark(node, remaining):
        if not remaining:
            return marker
        children = list(node.children)
        children[remaining[0]] = mark(children[remaining[0]], remaining[1:])
        return Program(node.op, node.value, node.name, node.lag, tuple(children), node.definition)

    uses = []

    def visit(node, origin=()):
        if node.op in {'ge', 'lt'}:
            for index, child in enumerate(node.children):
                if child == marker and node.children[1 - index] != marker:
                    uses.append((True, origin, 1 - index))
        for index, child in enumerate(node.children):
            visit(child, (*origin, index))

    visit(mark(program, path).expanded())
    return uses


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


def _slot_degree(program: Program, parameter_paths) -> int:
    """Dependence on selected constants, including lexical call arguments."""
    selected = set(parameter_paths)

    def degree(node, path=(), arguments=None):
        if path is not None and path in selected:
            return 1
        if node.op == 'arg':
            return (arguments or {}).get(node.name, 0)
        children = [degree(child, None if path is None else (*path, index), arguments)
                    for index, child in enumerate(node.children)]
        if node.op == 'call':
            return degree(node.definition.body, None, dict(zip(node.definition.arguments, children)))
        if not any(children):
            return 0
        if node.op in {'add', 'sub'}:
            return max(children)
        if node.op == 'mul':
            return sum(children)
        if node.op == 'if' and children[0] == 0:
            return max(children[1:])
        return 2

    return degree(program)


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
    transfer_history: list[dict[str, Any]] = field(default_factory=list)
    object_roles: dict[str, tuple[int, str]] = field(default_factory=dict)

    @property
    def parameters(self) -> tuple[float, ...]:
        return tuple(float(_at(self.pattern, path).value) for path in self.parameter_paths)

    def instantiate(self, mapping: dict[str, str],
                    parameters: Sequence[float] | None = None) -> Program:
        values = self.parameters if parameters is None else tuple(parameters)
        if len(values) != len(self.parameter_paths):
            raise ValueError("template parameter count differs")
        return _bind_constants(self.pattern, dict(zip(self.parameter_paths, values))).remap_features(mapping)

    def mappings(self, feature_names: Sequence[str], *, structural_transfer: bool = True,
                 object_schema: dict | None = None) -> Iterator[dict[str, str]]:
        identity = dict(zip(self.roles, self.source_features))
        emitted_identity = set(self.source_features).issubset(feature_names)
        emitted = []
        if emitted_identity:
            emitted.append(identity)
            yield identity
        if not structural_transfer:
            return
        for event in reversed(self.transfer_history):
            mapping = event.get("mapping", {})
            if (event.get("accepted") and set(mapping) == set(self.roles) and
                    set(mapping.values()).issubset(feature_names) and mapping not in emitted):
                emitted.append(mapping)
                yield dict(mapping)
        if object_schema and set(self.object_roles) == set(self.roles):
            indices = sorted({index for index, _ in self.object_roles.values()})
            for objects in permutations(object_schema["objects"], len(indices)):
                assignment = dict(zip(indices, objects))
                mapping = {name: object_schema["fields"][assignment[index]][role]
                           for name, (index, role) in self.object_roles.items()}
                if mapping not in emitted:
                    emitted.append(mapping)
                    yield mapping
            return
        for names in permutations(feature_names, len(self.roles)):
            mapping = dict(zip(self.roles, names))
            if mapping not in emitted:
                yield mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id, "source_concept_id": self.source_concept_id,
            "pattern": self.pattern.to_dict(), "roles": list(self.roles),
            "source_features": list(self.source_features),
            "parameter_paths": [list(path) for path in self.parameter_paths],
            "kind": self.kind, "status": self.status, "parents": list(self.parents),
            "uses": list(self.uses),
            "transfer_history": [dict(event) for event in self.transfer_history],
            "object_roles": {name: list(role) for name, role in self.object_roles.items()},
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
            transfer_history=[dict(event) for event in payload.get("transfer_history", [])],
            object_roles={name: tuple(role) for name, role in payload.get("object_roles", {}).items()},
        )


def extract_templates(program: Program, concept_id: str, *, object_schema: dict | None = None) -> list[AbstractTemplate]:
    # Parameter paths belong to the fitted expression. A learned call remains a
    # call in its concept, while template refitting sees its full computation.
    program = program.expanded()
    subexpressions: list[Program] = []

    def visit(node: Program) -> None:
        # Retain useful computation, not every scalar multiplication in an
        # affine fit. History itself is a reusable observation transformation.
        nonlinear = node.op in {"abs", "ge", "lt", "and", "or", "if", "min", "max", "accumulate"}
        interaction = node.op == "mul" and all(child.observed_variables() for child in node.children)
        if (nonlinear or interaction or node.op == "lag") and node != program and node not in subexpressions:
            subexpressions.append(node)
        for child in node.children:
            visit(child)

    visit(program)
    templates = []
    object_fields = {name: (index, role) for index, item in enumerate(object_schema["objects"])
                     for role, name in object_schema["fields"][item].items()} if object_schema else {}
    for index, expression in enumerate([program, *subexpressions]):
        names = tuple(sorted({name for name, _ in expression.observed_variables() if name != "__target__"}))
        if not expression.observed_variables():
            continue
        roles = tuple(f"role_{index}" for index in range(len(names)))
        pattern = expression.remap_features(dict(zip(names, roles)))
        templates.append(AbstractTemplate(
            f"{concept_id}:abstract:{index}", concept_id, pattern, roles, names,
            tuple(_constant_paths(pattern)), "theory" if index == 0 else "subexpression",
            parents=[concept_id], object_roles={role: object_fields[name] for role, name in zip(roles, names)
                                               if name in object_fields},
        ))
    return templates


def fitting_shape(template: AbstractTemplate, mapping: dict[str, str]) -> tuple:
    """Symbolic fit sharing for this dataset, never behavioral equivalence.

    Constants are parameter slots; actual observed names and history offsets
    remain in the key. Commutative/associative syntax shares a fit, but two
    expressions that merely agree on today's measurements stay separate.
    """
    def shape(node: Program) -> tuple:
        if node.op == "const":
            return ("parameter",)
        children = [shape(child) for child in node.children]
        if node.op == "accumulate":
            return node.op, node.name, node.lag, tuple(children), node.value
        if node.op in {"add", "mul", "and", "or", "min", "max"}:
            flattened = []
            for child in children:
                if child[0] == node.op:
                    flattened.extend(child[3])
                else:
                    flattened.append(child)
            children = sorted(flattened, key=repr)
        return node.op, node.name, node.lag, tuple(children)
    return template.kind, shape(template.instantiate(mapping))


def _structural_terms(program: Program) -> list[Program]:
    """Remove fitted affine coefficients while keeping actual computations."""
    if program.op == "const":
        return []
    if program.op in {"add", "sub"}:
        return list(dict.fromkeys(term for child in program.children for term in _structural_terms(child)))
    if program.op == "mul":
        if program.children[0].op == "const":
            return _structural_terms(program.children[1])
        if program.children[1].op == "const":
            return _structural_terms(program.children[0])
    if program.op == "if" and program.children[0].op == "available":
        # A trajectory's startup and mature dynamics need independent fitted
        # coefficients. The guards also keep lag terms executable at startup.
        condition, mature, startup = program.children
        zero, one = Program("const", value=0.0), Program("const", value=1.0)
        return [*[Program("if", children=(condition, term, zero))
                  for term in [one, *_structural_terms(mature)]],
                *[Program("if", children=(condition, zero, term))
                  for term in _structural_terms(startup)]]
    return [program]


def trajectory_losses(program: Program, observations: Sequence[Observation],
                      fallback: float, quantile: float) -> tuple[float, ...]:
    from .evaluation import point_loss, predict
    blocks: list[list[float]] = []
    previous_depth = 0
    for row in observations:
        depth = len(row.history)
        if not blocks or depth == 0 or depth < previous_depth:
            blocks.append([])
        blocks[-1].append(point_loss(predict(program, row, fallback), row.target, quantile))
        previous_depth = depth
    return tuple(sum(block) / len(block) for block in blocks)


def screen_template(template: AbstractTemplate, mapping: dict[str, str],
                    observations: Sequence[Observation], budget: SearchBudget,
                    risk_quantile: float = 0.5) -> dict[str, Any] | None:
    """Two-pass structural screen over all available trajectories.

    A cheap linear head refits the learned computations. Thresholds and other
    internal nonlinear parameters are left for shortlisted full fits. Initial
    history gaps still count in the loss, with the same fitted fallback used
    for every shape. This is training evidence used to schedule work.
    """
    from .evaluation import initial_prediction
    program = template.instantiate(mapping)
    variables = program.observed_variables()
    rows = [row for row in observations if len(row.history) >= program.minimum_history_depth
            and all(lag > len(row.history) or name in (row.features if lag == 0 else row.history[-lag])
                    for name, lag in variables)]
    if not rows or budget.remaining < len(rows) + len(observations):
        return None
    terms = _structural_terms(program)
    if template.kind == "subexpression":
        terms = list(dict.fromkeys([
            *[Program("feature", name=name) for name in sorted(set.intersection(*(set(row.features) for row in rows)))],
            *terms,
        ]))
    budget.spend(len(rows))
    fitted = _fit_linear(terms, rows)
    fallback = initial_prediction(observations, risk_quantile)
    budget.spend(len(observations))
    losses = trajectory_losses(fitted, observations, fallback, risk_quantile)
    return {"program": fitted, "loss": sum(losses) / len(losses),
            "worst_trajectory_loss": max(losses), "trajectory_losses": losses,
            "fitted_rows": len(rows), "rows": len(observations),
            "history_depth": program.history_depth, "fallback": fallback}


def eligible_fitting_rows(program: Program, observations: Sequence[Observation]) -> list[Observation]:
    """The existing complete-Program fitting domain, before any scoring."""
    variables = program.observed_variables()
    return [row for row in observations
            if len(row.history) >= program.minimum_history_depth
            and all(lag > len(row.history) or name in (row.features if lag == 0 else row.history[-lag])
                    for name, lag in variables)]


def quote_parameter_slot_fit(template: AbstractTemplate, observations: Sequence[Observation],
                             mapping: dict[str, str] | None = None) -> dict[str, Any]:
    """Bound one cold, complete finite ``fit_parameter_slots`` call, without prediction.

    This quotes the existing fitter's candidate-by-row account, including both
    old coordinate rounds and every conditional linear profile they can visit.
    Preparation, candidate assembly/checking and unpaid comparison belong to
    the caller's complete research visit. A finite fit does not establish a
    nonlinear optimum or scientific support.
    """
    for path in template.parameter_paths:
        if _at(template.pattern, path).op != 'const':
            raise ValueError('parameter_paths must select actual const slots')
    initial = template.instantiate(mapping or {})
    rows = eligible_fitting_rows(initial, observations)
    count, selected = len(rows), len(template.parameter_paths)
    degree = _slot_degree(template.pattern, template.parameter_paths)

    def lexical_nodes(node):
        return 1 + sum(lexical_nodes(child) for child in node.children)

    quote = {'scope': 'cold_complete_finite_fit_only', 'compute': 0,
        'maximum_outputs': 0, 'eligible_rows': count, 'selected_slot_count': selected,
        'selected_slot_degree': degree, 'expanded_program_nodes': initial.expanded().size,
        'lexical_program_nodes': lexical_nodes(initial), 'coordinate_compute': 0,
        'conditional_profile_count_bound': 0, 'conditional_compute': 0}
    if not count:
        return quote
    if not selected:
        return {**quote, 'compute': count, 'maximum_outputs': 1}
    if degree <= 1:
        # Initial, zero basis, p unit bases, ridge and stable evaluations.
        # Repeated Programs may cost less; restoration cannot assume hot values.
        return {**quote, 'compute': (selected + 4) * count, 'maximum_outputs': 3}

    comparison_uses = {path: _comparison_slot_uses(template.pattern, path)
                       for path in template.parameter_paths}
    trial_bound, threshold_trials, operand_compute = 0, 0, 0
    for path in template.parameter_paths:
        uses = comparison_uses[path]
        # The current value is already measured. Only 0, +1, -1 and at most
        # nine observed midpoints per operand can add new Program evaluations.
        trials = 3
        for expanded, origin, operand in uses:
            other = _at(initial.expanded() if expanded else initial, origin).children[operand]
            operand_rows = sum(len(row.history) >= other.minimum_history_depth for row in rows)
            operand_compute += operand_rows
            trials += min(9, max(0, operand_rows - 1))
        trial_bound += trials
        if uses:
            threshold_trials += trials
    coordinate = count + 2 * (trial_bound * count + operand_compute)
    outputs = 1 + 2 * trial_bound
    ordinary = tuple(path for path in template.parameter_paths if not comparison_uses[path])
    if threshold_trials and ordinary and _slot_degree(template.pattern, ordinary) <= 1:
        # Conditions are collected only from the old coordinate evaluations.
        # An ordinary-slot trial cannot change the threshold tuple; each new
        # threshold trial contributes at most one tuple, rather than an
        # unvisited Cartesian product of possible threshold values.
        profiles = 1 + 2 * threshold_trials
        profile_compute = profiles * (len(ordinary) + 4) * count
        quote.update(conditional_profile_count_bound=profiles,
                     conditional_compute=profile_compute)
        outputs += 2 * profiles  # Each profile can append ridge and stable.
    quote.update(compute=coordinate + quote['conditional_compute'],
                 maximum_outputs=outputs, coordinate_compute=coordinate)
    return quote


def _fit_slots(template, mapping, observations, budget, risk_quantile, *, report,
               parameter_degree, calibrate, _evaluation_state=None):
    started = budget.spent
    initial = template.instantiate(mapping)
    rows = eligible_fitting_rows(initial, observations)
    count = len(rows)
    programs: list[Program] = []
    measured_losses, residuals_by_program, predictions_by_program = (
        _evaluation_state if _evaluation_state is not None else ({}, {}, {}))
    methods = []
    linear = {'status': 'UNKNOWN', 'solver': 'numpy.linalg.lstsq', 'rows': count,
        'selected_slot_count': len(template.parameter_paths), 'rank': None,
        'singular_values': [], 'condition_number': None, 'rank_deficient': None,
        'parameter_identifiability': 'UNKNOWN', 'scope': 'linear_training_rows_only',
        'least_squares_objective': 'sum_squared_residuals',
        'evaluation_objective': 'mean_squared_error' if risk_quantile == .5 else 'quantile_loss',
        'reason': 'linear_basis_not_evaluated'}
    if report is not None:
        report.pop('linear_diagnostics', None)
        report.pop('conditional_linear_profiles', None)
        report.pop('conditional_linear_threshold_paths', None)
        report["losses"] = measured_losses
        report["fitted_rows"] = count
        report['parameter_paths'] = [list(path) for path in template.parameter_paths]
        report['selected_slot_degree'] = parameter_degree
        report['candidate_methods'] = methods
        if template.parameter_paths and parameter_degree <= 1:
            report['linear_diagnostics'] = linear

    def finish(completed):
        identity = ('not_applicable' if not template.parameter_paths else
            linear['parameter_identifiability'] if parameter_degree <= 1 else 'UNKNOWN')
        reason = ('no_executable_fitting_rows' if not rows else
            'parameter_fitting_budget_exhausted' if not completed else
            'nonlinear_parameter_search_is_local' if template.parameter_paths and parameter_degree > 1 else
            linear.get('reason') if template.parameter_paths and linear['status'] != 'computed' else
            'rank_deficient_parameter_fit' if linear['rank_deficient'] else None)
        if report is not None:
            report.update(status='UNKNOWN' if reason else 'completed', search_completed=completed,
                budget_exhausted=bool(rows) and not completed,
                parameter_identifiability=identity, actual_compute_cost=budget.spent - started,
                actual_fitted_rows=count if programs else 0, eligible_fitting_rows=count,
                global_optimum_established=False, scientific_support=False)
            if reason:
                report['reason'] = reason
            else:
                report.pop('reason', None)
        return programs, completed, rows

    if not rows:
        return finish(False)

    def append(program, method):
        programs.append(program)
        methods.append(method)

    def training_summary(program, parameters, penalty):
        residuals = residuals_by_program[program]
        sse = sum(value * value for value in residuals)
        return {'train_residual_sum_squares': sse, 'train_mse': sse / count,
            'train_residual_max_abs': max(abs(value) for value in residuals),
            'penalty': penalty, 'objective': sse + penalty * sum(value * value for value in parameters),
            'objective_kind': 'sum_squared_residuals_plus_parameter_penalty',
            'evaluation_loss': measured_losses[program]}

    def evaluate(program: Program) -> float | None:
        if program in measured_losses:
            return measured_losses[program]
        if program not in predictions_by_program:
            if not budget.spend(count):
                return None
            predictions_by_program[program] = predict_rows(program, rows)
        # The same _loss objective and summation order, retaining its residuals
        # for diagnostics without a second, uncharged prediction pass.
        residuals = [row.target - prediction for row, prediction in
                     zip(rows, predictions_by_program[program])]
        residuals_by_program[program] = residuals
        measured_losses[program] = (sum(value * value for value in residuals) / count
            if risk_quantile == .5 else sum(max(risk_quantile * value,
                (risk_quantile - 1) * value) for value in residuals) / count)
        return measured_losses[program]

    best_loss = evaluate(initial)
    if best_loss is None:
        return finish(False)
    append(initial, 'initial')
    best = initial
    parameters = list(template.parameters)
    complete = True
    basis_programs = None
    basis_required = (len(parameters) + 2) * count
    if parameters and parameter_degree <= 1 and _evaluation_state is not None:
        base = template.instantiate(mapping, [0.0] * len(parameters))
        basis_programs = [base]
        for index in range(len(parameters)):
            unit = [0.0] * len(parameters)
            unit[index] = 1.0
            basis_programs.append(template.instantiate(mapping, unit))
        basis_required = (sum(program not in predictions_by_program
                              for program in set(basis_programs)) + 1) * count
    if parameters and parameter_degree <= 1 and budget.remaining >= basis_required:
        base = template.instantiate(mapping, [0.0] * len(parameters))

        def basis_values(program):
            if _evaluation_state is not None and program in predictions_by_program:
                return predictions_by_program[program]
            budget.spend(count)
            values = predict_rows(program, rows)
            if _evaluation_state is not None:
                predictions_by_program[program] = values
            return values

        base_values = basis_values(base)
        columns = []
        for index in range(len(parameters)):
            unit = [0.0] * len(parameters)
            unit[index] = 1.0
            basis = template.instantiate(mapping, unit)
            columns.append([prediction - value for prediction, value in zip(basis_values(basis), base_values)])
        gram = [[sum(a * b for a, b in zip(left, right)) for right in columns] for left in columns]
        for index in range(len(parameters)):
            gram[index][index] += 1e-9
        targets = [row.target - value for row, value in zip(rows, base_values)]
        fitted_parameters = _solve(gram, [sum(a * b for a, b in zip(column, targets)) for column in columns])
        fitted = template.instantiate(mapping, fitted_parameters)
        fitted_loss = evaluate(fitted)
        if fitted_loss is not None:
            append(fitted, 'ridge_lstsq')
            linear['ridge'] = training_summary(fitted, fitted_parameters, 1e-9)
            if fitted_loss < best_loss:
                best, best_loss, parameters = fitted, fitted_loss, fitted_parameters
        # Both candidates reuse this exact paid basis and fixed containing
        # Program. Only the additional candidate's evaluation costs another N.
        if budget.can_spend(count):
            design = np.asarray(columns, dtype=float).T
            try:
                coefficients, _, rank, singular = np.linalg.lstsq(design, targets, rcond=None)
            except np.linalg.LinAlgError:
                linear['reason'] = 'stable_linear_solver_failed'
            else:
                stable_parameters = coefficients.tolist()
                stable = template.instantiate(mapping, stable_parameters)
                stable_loss = evaluate(stable)
                if stable_loss is not None:
                    append(stable, 'stable_lstsq')
                    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None
                    rank_deficient = int(rank) < len(template.parameter_paths)
                    basis_errors = design @ coefficients - np.asarray(targets)
                    linear.update(status='computed', rank=int(rank), singular_values=singular.tolist(),
                        condition_number=condition if condition is not None and isfinite(condition) else None,
                        rank_deficient=rank_deficient,
                        parameter_identifiability='UNKNOWN' if rank_deficient else 'full_rank_linear_design',
                        stable=training_summary(stable, stable_parameters, 0.),
                        linear_l2_residual_sum_squares=float(basis_errors @ basis_errors))
                    linear.update({key: linear['stable'][key] for key in
                        ('train_residual_sum_squares', 'train_mse', 'train_residual_max_abs')})
                    linear.pop('reason', None)
                    if stable_loss < best_loss:
                        best, best_loss, parameters = stable, stable_loss, stable_parameters
        else:
            linear['reason'] = 'stable_linear_candidate_budget_exhausted'
            complete = False
    elif parameters and parameter_degree <= 1:
        # An unaffordable linear basis is not a nonlinear fitting problem.
        # Keep the paid initial candidate and the actual denied atomic request
        # so the caller can fund the same least-squares method on continuation.
        budget.can_spend(basis_required)
        linear['reason'] = 'linear_basis_budget_exhausted'
        complete = False
    elif parameters:
        # Comparisons and conditional expressions need actual discrete trials.
        # Candidate thresholds come from current observed values of the operand.
        comparison_uses = {path: _comparison_slot_uses(template.pattern, path)
                           for path in template.parameter_paths}
        for _ in range(2):
            improved = False
            for index, path in enumerate(template.parameter_paths):
                values = [parameters[index], 0.0, 1.0, -1.0]
                uses = comparison_uses[path]
                if uses:
                    current = template.instantiate(mapping, parameters)
                    operands = dict.fromkeys(_at(current.expanded() if expanded else current, origin).children[operand]
                        for expanded, origin, operand in uses)
                    for other in operands:
                        # The complete program can execute a startup branch
                        # while this threshold operand still requires history.
                        operand_rows = [row for row in rows
                                        if len(row.history) >= other.minimum_history_depth]
                        if not budget.spend(len(operand_rows)):
                            return finish(False)
                        ordered = sorted(set(predict_rows(other, operand_rows)))
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
                        return finish(False)
                    if trial_loss < best_loss:
                        best, best_loss, parameters = trial, trial_loss, trial_parameters
                        append(best, 'coordinate_trial')
                        improved = True
            if not improved:
                break

        # A threshold makes the joint problem nonlinear, while the other
        # declared slots can remain affine with that threshold fixed. Preserve
        # the complete old finite route first, then reuse its paid predictions
        # and the ordinary linear fitter for conditions it actually visited.
        threshold_paths = tuple(path for path, uses in comparison_uses.items() if uses)
        ordinary_paths = tuple(path for path in template.parameter_paths if not comparison_uses[path])
        if (not calibrate and threshold_paths and ordinary_paths
                and _slot_degree(template.pattern, ordinary_paths) <= 1):
            conditions = {}
            for program in tuple(measured_losses):
                condition = tuple(_at(program, path).value for path in threshold_paths)
                conditions.setdefault(condition, program)
            profiles = []
            if report is not None:
                report['conditional_linear_profiles'] = profiles
                report['conditional_linear_threshold_paths'] = [list(path) for path in threshold_paths]
            for condition, pattern in conditions.items():
                local_report = {}
                selected = replace(template, pattern=pattern, parameter_paths=ordinary_paths)
                fitted, completed, _ = _fit_slots(selected, {}, rows, budget, risk_quantile,
                    report=local_report, parameter_degree=1, calibrate=False,
                    _evaluation_state=(measured_losses, residuals_by_program, predictions_by_program))
                profiles.append({'threshold_values': list(condition),
                    **{key: value for key, value in local_report.items() if key != 'losses'}})
                for program, method in zip(fitted, local_report['candidate_methods']):
                    if method != 'initial' and program not in programs:
                        append(program, 'conditional_' + method)
                        if measured_losses[program] < best_loss:
                            best, best_loss = program, measured_losses[program]
                if not completed:
                    return finish(False)

    if not calibrate:
        return finish(complete)
    # A reusable nonlinear subexpression becomes an observed coordinate in an
    # ordinary new model. Its coefficients are refitted instead of copying old
    # predictions. Also permits a changed offset around a learned full theory.
    if budget.spend(count):
        calibrated = _fit_linear((best,), rows)
        if evaluate(calibrated) is not None and calibrated not in programs:
            append(calibrated, 'output_calibration')
    if template.kind == "subexpression" and budget.spend(count):
        current = [Program("feature", name=name)
                   for name in sorted(set.intersection(*(set(row.features) for row in rows)))]
        combined = _fit_linear([*current, best], rows)
        if evaluate(combined) is not None and combined not in programs:
            append(combined, 'current_feature_calibration')
    return finish(complete)


def fit_parameter_slots(template: AbstractTemplate, mapping: dict[str, str],
                        observations: Sequence[Observation], budget: SearchBudget,
                        q: float = 0.5, *, report: dict[str, Any] | None = None) -> list[Program]:
    """Fit only declared const slots against the complete Program's targets.

    Lexical Definition arguments keep repeated holes shared. Constants outside
    ``parameter_paths`` and all operators/history offsets stay fixed. Linear
    selected parameters retain the ridge solve and add stable least squares on
    the same paid basis. Nonlinear dependence uses the existing finite trials;
    affine remaining slots are then fitted under the conditions actually tried.
    Rank deficiency and local trials retain UNKNOWN identifiability. No
    output calibration or extra current-feature model is appended.
    """
    for path in template.parameter_paths:
        if _at(template.pattern, path).op != 'const':
            raise ValueError('parameter_paths must select actual const slots')
    local_report = report if report is not None else {}
    local_report['losses'] = {}
    local_report.pop('reason', None)
    programs, _, _ = _fit_slots(template, mapping, observations, budget, q,
        report=local_report, parameter_degree=_slot_degree(template.pattern, template.parameter_paths),
        calibrate=False)
    local_report['parameter_scope'] = 'declared_const_slots_only'
    return programs


def fit_template(template: AbstractTemplate, mapping: dict[str, str],
                 observations: Sequence[Observation], budget: SearchBudget,
                 risk_quantile: float = 0.5, *, report: dict[str, Any] | None = None) -> list[Program]:
    """Fit constant slots and optional output calibration using only given rows.

    Basis evaluations and every threshold/parameter trial consume candidate ×
    row units, including trials that are discarded. No label outside this call
    participates in fitting or parameter selection.
    """
    return _fit_slots(template, mapping, observations, budget, risk_quantile,
        report=report, parameter_degree=_parameter_degree(template.pattern), calibrate=True)[0]
