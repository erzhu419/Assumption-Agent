"""Executable relation intervals retained inside complete consuming programs."""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations

from .language import Program
from .relation_revisions import ZERO, _distances, _expose, _relations, _replacement_call, _substitute
from .relation_conditions import TRUE
from .models import Observation


def _bind(program, values):
    if program.op == "arg" and program.name in values:
        return values[program.name]
    return replace(program, children=tuple(_bind(child, values) for child in program.children))


def _calls(program):
    return ({program} if program.op == "call" else set()) | set().union(
        *(_calls(child) for child in program.children))


def _mark_relations(program, relations, symbol):
    if program.op == "call":
        body = _substitute(program.definition.body,
                           dict(zip(program.definition.arguments, program.children)))
        marked = _mark_relations(body, relations, symbol)
        return program if marked == body else marked
    children = tuple(_mark_relations(child, relations, symbol) for child in program.children)
    if program in relations:
        children = (Program("arg", name=symbol), *children[1:])
    return replace(program, children=children)


def _delta(program, before, after):
    old, new = _bind(program, before), _bind(program, after)
    if old == new:
        return ZERO
    if program.op in {"add", "sub", "object_sum", "event_delta", "window_sum", "accumulate"}:
        return replace(program, children=tuple(_delta(child, before, after)
                                               for child in program.children)).simplified()
    if program.op == "mul":
        left, right = program.children
        dl, dr = _delta(left, before, after), _delta(right, before, after)
        return Program("add", children=(Program("add", children=(
            Program("mul", children=(dl, _bind(right, before))),
            Program("mul", children=(_bind(left, before), dr)))),
            Program("mul", children=(dl, dr)))).simplified()
    return Program("sub", children=(new, old)).simplified()


def _match(pattern, actual, bindings):
    if pattern.op in {"feature", "lag", "available"}:
        if pattern in bindings:
            return bindings[pattern] == actual
        bindings[pattern] = actual
        return True
    return ((pattern.op, pattern.value, pattern.name, pattern.lag) ==
            (actual.op, actual.value, actual.name, actual.lag)
            and len(pattern.children) == len(actual.children)
            and all(_match(left, right, bindings) for left, right in zip(pattern.children, actual.children)))


def _replace_leaves(program, bindings):
    if program in bindings:
        return bindings[program]
    return replace(program, children=tuple(_replace_leaves(child, bindings) for child in program.children))


def _parameter_relations(program, symbol):
    result = [program] if (program.op in {"neighbor_sum", "relation_sum"} and
        program.children[0] == Program("arg", name=symbol)) else []
    return [*result, *(item for child in program.children for item in _parameter_relations(child, symbol))]


class RelationIntervalResearch:
    """Keep the original call executable while testing its unresolved parameters."""

    def __init__(self, program, priors, budget, *, structure=None, research_program=None):
        self.program, self.parameters, self.values = program, [], {}
        self.compact_replacement = False
        self.candidate_serial = 0
        self.evidence_contexts, self.evidence_updates = [], []
        self._inherited_contexts = []
        self._inherited_domains = {}
        self._local_context = None
        self.status = "no_relation_interval"
        prior_by_source = {prior["source_concept_id"]: prior for prior in priors
                           if prior.get("status") != "revoked"}

        def stored(node, payload, source):
            steps = (payload.get("research_program") or {}).get("selection_steps", [])
            saved = next((step for step in reversed(steps)
                          if step["operation"] == "resolve_relation_intervals"), None)
            if saved is not None:
                self.compact_replacement |= saved.get("compact_replacement", False)
                template = Program.from_dict(saved["parameter_template"])
                values = {item["symbol"]: Program("const", value=item["radius"])
                          for item in saved["parameters"]}
                expected, actual = _bind(template, values).simplified().expanded(), node.expanded()
                if not budget.spend(expected.size + actual.size):
                    return None
                bindings = {}
                if not _match(expected, actual, bindings):
                    return None
                template = _replace_leaves(template, bindings)
                rename = {}
                for item in saved["parameters"]:
                    symbol = f"relation_parameter_{len(self.parameters)}"
                    rename[item["symbol"]] = Program("arg", name=symbol)
                    self.parameters.append({**deepcopy(item), "symbol": symbol})
                    if item.get("prior_radius_intervals") != item["radius_intervals"] and item.get("prior_radius_intervals"):
                        self._inherited_domains[symbol] = deepcopy(item["prior_radius_intervals"])
                for context in saved.get("evidence_contexts", []):
                    context = deepcopy(context)
                    context["parameter_template"] = _bind(
                        Program.from_dict(context["parameter_template"]), rename).to_dict()
                    context["parameters"] = {rename[key].name: value
                        for key, value in context["parameters"].items()}
                    self.evidence_contexts.append(context)
                    self._inherited_contexts.append(context)
                return _bind(template, rename)
            previous = payload.get("structure")
            records = ([{"radius": previous["fits"][0]["radius"],
                         "radius_interval": previous.get("radius_interval")}] if previous and
                       previous.get("fits") else
                       [step for step in steps if step["operation"] in {"revise_nested_relation", "revise_relation_type", "revise_relation_condition"}])
            records = [item for item in records if item.get("radius") is not None and
                       item.get("radius_interval") and item["radius_interval"][1] > item["radius_interval"][0]]
            if not records:
                return None
            opened = _expose(node, budget)
            if opened is None:
                return None
            marked = node
            for item in records:
                self.compact_replacement |= item.get("compact_replacement", False)
                related = [relation for relation in _relations(opened)
                           if relation.children[0].value == item["radius"] and
                           (relation.name or "symmetric") == item.get("relation", "symmetric") and
                           (relation.children[1] if relation.op == "relation_sum" else TRUE) ==
                           Program.from_dict(item.get("predicate", TRUE.to_dict()))]
                if not related:
                    continue
                symbol = f"relation_parameter_{len(self.parameters)}"
                self.parameters.append({"symbol": symbol, "source_concept_id": source,
                    "source_definition_id": node.definition.definition_id if node.op == "call" else None,
                    "relation": item.get("relation", "symmetric"),
                    "predicate": item.get("predicate", TRUE.to_dict()),
                    "radius": item["radius"], "radius_intervals": [list(item["radius_interval"])]})
                marked = _mark_relations(marked, set(related), symbol)
            return marked if marked != node else None

        def visit(node):
            if not budget.spend(1):
                return node
            if node.op == "call":
                prior = prior_by_source.get(node.definition.source_concept_id)
                payload = prior
                if prior and prior.get("input_program", {}).get("definition", {}).get("definition_id") == node.name:
                    payload = {**prior, "research_program": prior.get("input_research_program")}
                marked = stored(node, payload, node.definition.source_concept_id) if payload else None
                if marked is not None:
                    return marked
                body = _substitute(node.definition.body, dict(zip(node.definition.arguments, node.children)))
                count = len(self.parameters)
                opened = visit(body)
                return opened if len(self.parameters) > count else node
            return replace(node, children=tuple(visit(child) for child in node.children))

        own = {"structure": structure, "research_program": research_program}
        self.template = stored(program, own, None) if structure or research_program else None
        if self.template is None:
            self.template = visit(program)
        self.retained_calls = _calls(program) <= _calls(self.template)
        self.values = {item["symbol"]: item["radius"] for item in self.parameters}
        for parameter in self.parameters:
            parameter.setdefault("prior_radius_intervals", deepcopy(parameter["radius_intervals"]))
        self.original = dict(self.values)
        if self.parameters:
            self.status = "unresolved"

    @property
    def intervals(self):
        return [{**item, "radius": self.values[item["symbol"]], "radius_interval": interval}
                for item in self.parameters for interval in item["radius_intervals"]]

    def _program(self, values):
        old = {key: Program("const", value=value) for key, value in self.original.items()}
        new = {key: Program("const", value=value) for key, value in values.items()}
        if self.compact_replacement:
            return _replacement_call(self.program, _bind(self.template, new).simplified(),
                                     f"parameter_{self.candidate_serial}"), None
        if self.retained_calls:
            return _bind(self.template, new).simplified(), None
        correction = _delta(self.template, old, new)
        return Program("add", children=(self.program, correction)).simplified(), correction

    def _entry(self, values, interval, index, error, rows, cost):
        from .research_programs import _entry
        program, correction = self._program(values)
        original = {key: Program("const", value=value) for key, value in self.original.items()}
        symbolic = {key: Program("arg", name=key) for key in values}
        template = (self.template if correction is None else
            Program("add", children=(self.program, _delta(self.template, original, symbolic))).simplified())
        parameters = [{**deepcopy(item), "radius": values[item["symbol"]]} for item in self.parameters]
        parameter_terms = ([template] if correction is None else
            [self.program, _delta(self.template, original, symbolic)])
        steps = [{"operation": "retain_composed_call", "question": "Retain the complete existing computation",
                  "program": self.program, "loss": error},
                 {"operation": "resolve_relation_intervals", "question": "Resolve a parameter inside its actual consuming context",
                  "program": program, "loss": error, "parameter_template": template.to_dict(),
                  "parameters": parameters, "parameter_index": index, "radius_interval": list(interval),
                  "evidence_contexts": deepcopy(self.evidence_contexts),
                  "parameter_terms": [term.to_dict() for term in parameter_terms],
                  "compact_replacement": self.compact_replacement}]
        terms = [program] if correction is None else [self.program, correction]
        fit = {"program": program, "terms": terms, "fit_terms": terms,
               "coefficients": [0., *([1.] * len(terms))], "loss": error}
        entry = _entry(fit, steps, rows, cost, self.candidate_serial)
        self.candidate_serial += 1
        entry.update(parameter_index=index, parameter_values=dict(values),
                     validation_loss=error, selection_basis="all_acquired_evidence")
        return entry

    def _retain_experiments(self, experiments, tolerance):
        if not experiments:
            return
        if self._local_context is None:
            self._local_context = {"parameter_template": self.template.to_dict(),
                "parameters": dict(self.values), "trajectories": [], "tolerance": tolerance}
            self.evidence_contexts.append(self._local_context)
        for trajectory in experiments:
            rows = [asdict(row) for row in trajectory]
            if rows and rows not in self._local_context["trajectories"]:
                self._local_context["trajectories"].append(rows)

    def reopen(self, reason, *, parameter_index=None):
        """Withdraw inherited exclusions in this research, preserving the source record."""
        parameters = self.parameters if parameter_index is None else [self.parameters[parameter_index]]
        symbols = {item["symbol"] for item in parameters}
        withdrawn = [context for context in self._inherited_contexts
                     if symbols.intersection(context["parameters"])]
        if not withdrawn and not symbols.intersection(self._inherited_domains):
            return False
        for parameter in parameters:
            parameter["radius_intervals"] = deepcopy(parameter["prior_radius_intervals"])
            self._inherited_domains.pop(parameter["symbol"], None)
        self.evidence_contexts = [context for context in self.evidence_contexts
                                 if not any(context is old for old in withdrawn)]
        self._inherited_contexts = [context for context in self._inherited_contexts
                                   if not any(context is old for old in withdrawn)]
        self.evidence_updates.append({"reason": reason, "parameters": sorted(symbols),
            "withdrawn_trajectories": sum(len(context["trajectories"]) for context in withdrawn),
            "reopened_intervals": deepcopy(self.intervals)})
        self.status = "unresolved"
        return True

    def _supports_inherited_evidence(self, values, budget, tolerance):
        # Historical targets belong to their historical complete response.
        # A new sum/window consumer inherits the parameter evidence, not those
        # targets as if they had been observed for today's output.
        for context in self._inherited_contexts:
            program = _bind(Program.from_dict(context["parameter_template"]), {
                key: Program("const", value=values.get(key, value))
                for key, value in context["parameters"].items()})
            for trajectory in context["trajectories"]:
                if not budget.spend(len(trajectory)):
                    return None
                error = sum((program.predict(Observation(row["features"], row["target"],
                    tuple(row["history"]))) - row["target"]) ** 2 for row in trajectory) / len(trajectory)
                if error > max(tolerance, context["tolerance"]):
                    return False
        return True

    def alternatives(self, paid_rows, budget, tolerance=1e-7, *, experiments=()):
        """Fit pooled observations without diluting completed distinguishing probes.

        Experiments retain actual rows, not excluded parameter values. Each
        current complete program therefore reinterprets them after input repair.
        """
        start = budget.spent
        self._retain_experiments(experiments, tolerance)
        evaluated_rows = list(paid_rows)
        identities = {id(row) for row in evaluated_rows}
        trajectory_ids = []
        for trajectory in experiments:
            if not trajectory:
                continue
            trajectory_ids.append(tuple(id(row) for row in trajectory))
            for row in trajectory:
                if id(row) not in identities:
                    identities.add(id(row))
                    evaluated_rows.append(row)
        for index, parameter in enumerate(self.parameters):
            domains = parameter["radius_intervals"]
            if not domains:
                continue
            if all(high - low <= 1e-6 for low, high in domains):
                if self._inherited_contexts or parameter["symbol"] in self._inherited_domains:
                    if not budget.spend(len(evaluated_rows)):
                        self.status = "compute_exhausted"
                        return []
                    program, _ = self._program(self.values)
                    errors = {id(row): (program.predict(row) - row.target) ** 2 for row in evaluated_rows}
                    if (sum(errors[id(row)] for row in paid_rows) / len(paid_rows) > tolerance or
                        any(sum(errors[key] for key in trajectory) / len(trajectory) > tolerance
                            for trajectory in trajectory_ids)) and self.reopen(
                                "current_relation_counterexample", parameter_index=index):
                        return self.alternatives(paid_rows, budget, tolerance, experiments=experiments)
                continue
            relations = _parameter_relations(self.template, parameter["symbol"])
            distances = _distances(relations, paid_rows, budget)
            supported, candidates = [], []
            for low, high in domains:
                cuts = [low, *(value for value in distances if low < value < high), high]
                for left, right in zip(cuts, cuts[1:]):
                    fitted = []
                    for fraction in (.5, .25, .75):
                        if not budget.spend(len(evaluated_rows)):
                            self.status = "compute_exhausted"
                            return []
                        values = {**self.values, parameter["symbol"]: left + (right - left) * fraction}
                        program, _ = self._program(values)
                        errors = {id(row): (program.predict(row) - row.target) ** 2 for row in evaluated_rows}
                        error = sum(errors[id(row)] for row in paid_rows) / len(paid_rows)
                        if error <= tolerance and all(sum(errors[key] for key in trajectory) / len(trajectory)
                                                      <= tolerance for trajectory in trajectory_ids):
                            inherited = self._supports_inherited_evidence(values, budget, tolerance)
                            if inherited is None:
                                self.status = "compute_exhausted"
                                return []
                            if inherited:
                                fitted.append((values, error))
                    if fitted:
                        supported.append([left, right])
                        candidates.extend((values, (left, right), error) for values, error in fitted)
            parameter["radius_intervals"] = supported
            if candidates:
                self.status = "unresolved"
                return [self._entry(values, interval, index, error, len(paid_rows), budget.spent - start)
                        for values, interval, error in candidates]
            if self.reopen("current_relation_counterexample", parameter_index=index):
                return self.alternatives(paid_rows, budget, tolerance, experiments=experiments)
            self.status = "refuted"
            return []
        self.status = "resolved_to_precision" if self.parameters else "no_relation_interval"
        return []

    def adopt(self, entry):
        self.values = dict(entry["parameter_values"])

    def questions(self, entries):
        from .relation_language import instantiate_predicate, relation_objects

        if len(entries) < 2:
            return []
        parameter = self.parameters[entries[0]["parameter_index"]]
        radii = [entry["parameter_values"][parameter["symbol"]] for entry in entries]
        low, high = min(radii), max(radii)
        if low == high:
            return []
        depth = max(entry["candidate"].program.history_depth for entry in entries)
        predicates = []
        for relation in _parameter_relations(self.template, parameter["symbol"]):
            objects = relation_objects(relation)
            pairs = ([(source, target) for index, source in enumerate(objects)
                      for other, target in enumerate(objects) if index != other]
                     if relation.op == "relation_sum" else combinations(objects, 2))
            for source, target in pairs:
                distance = Program("sub", children=(target[1], source[1]))
                if relation.op == "relation_sum" and relation.name == "left":
                    distance = Program("sub", children=(source[1], target[1]))
                elif relation.op != "relation_sum" or relation.name != "right":
                    distance = Program("abs", children=(distance,))
                band = Program("and", children=(
                    Program("ge", children=(distance, Program("const", value=low))),
                    Program("lt", children=(distance, Program("const", value=high)))))
                predicates.append(Program("and", children=(band,
                    instantiate_predicate(relation.children[1], source, target)))
                    if relation.op == "relation_sum" else band)
        return [{"question_id": f"inherited_relation:{parameter['symbol']}:{index}",
            "predicate": predicate,
            "history_depth": depth, "uncovered_phases": ["enter", "maintain", "exit"] if depth else ["active"],
            "dependent_hypothesis_ids": [entry["candidate"].candidate_id for entry in entries]}
            for index, predicate in enumerate(dict.fromkeys(predicates))]
