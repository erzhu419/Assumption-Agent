"""Executable relation intervals retained inside complete consuming programs."""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations

from .language import Program
from .relation_revisions import _distances, _expose, _relations, _replacement_call, _substitute
from .relation_conditions import TRUE
from .models import Observation
from .relation_kernel_priors import (
    recall_relation_kernels, interval_union, kernel_key, parameter_relations as _parameter_relations)


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


class RelationIntervalResearch:
    """Keep the original call executable while testing its unresolved parameters."""

    def __init__(self, program, priors, budget, *, structure=None, research_program=None):
        self.program, self.parameters, self.values = program, [], {}
        self.compact_replacement = False
        self.candidate_serial = 0
        self.evidence_contexts, self.evidence_updates = [], []
        self._inherited_contexts = []
        self._inherited_domains = {}
        self._expanded_domains = {}
        self._local_context = None
        self._paid_partition = None
        self._readout_fit = None
        self._parameter_terms = None
        self.reuse_statistics = {"partition_rows_bound": 0, "geometry_paid_rows_reused": 0,
                                 "inherited_predictions_reused": 0}
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
                    self.parameters.append({**deepcopy(item), "symbol": symbol,
                        "source_concept_id": item.get("source_concept_id") or source})
                    if item.get("prior_radius_intervals") != item["radius_intervals"] and item.get("prior_radius_intervals"):
                        self._inherited_domains[symbol] = deepcopy(item["prior_radius_intervals"])
                if source is None and node is program and saved.get("parameter_terms"):
                    self._parameter_terms = [_bind(_replace_leaves(
                        Program.from_dict(term), bindings), rename)
                        for term in saved["parameter_terms"]]
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
        # Repeated uses of the same source relation in the same measured
        # coordinates share one parameter even inside different time operators.
        # Otherwise changing one copy cannot repair the other copies, and each
        # coordinate-wise trial incorrectly refutes a valid shared revision.
        shared, rename, parameters = {}, {}, []
        for item in self.parameters:
            relations = _parameter_relations(self.template, item["symbol"])
            key = (item.get("source_concept_id"), item["radius"],
                   frozenset(kernel_key(node) for node in relations),
                   tuple(map(tuple, item["radius_intervals"])))
            if item.get("source_concept_id") is not None and key in shared:
                rename[item["symbol"]] = Program("arg", name=shared[key])
                self._inherited_domains.pop(item["symbol"], None)
            else:
                shared[key] = item["symbol"]
                parameters.append(item)
        if rename:
            self.parameters = parameters
            self.template = _bind(self.template, rename)
            if self._parameter_terms is not None:
                self._parameter_terms = [_bind(term, rename) for term in self._parameter_terms]
            contexts = []
            for context in self.evidence_contexts:
                context["parameter_template"] = _bind(
                    Program.from_dict(context["parameter_template"]), rename).to_dict()
                context["parameters"] = {rename[key].name if key in rename else key: value
                    for key, value in context["parameters"].items()}
                if context not in contexts:
                    contexts.append(context)
            self.evidence_contexts = contexts
            self._inherited_contexts = list(contexts)
        self.retained_calls = _calls(program) <= _calls(self.template)
        # Opening a parameter-bearing call produces its complete new response.
        # Save that body under a distinct definition; adding a delta to the
        # old response repeats whole windows and nonlinear consumers. The old
        # callable remains in the research record and knowledge library.
        self.compact_replacement |= not self.retained_calls
        self.values = {item["symbol"]: item["radius"] for item in self.parameters}
        for parameter in self.parameters:
            parameter.setdefault("prior_radius_intervals", deepcopy(parameter["radius_intervals"]))
        self.original = dict(self.values)
        self._retain_fitted_columns(structure, research_program)
        kernels = recall_relation_kernels(self.template, self.parameters, priors, budget)
        for parameter in self.parameters:
            sources = kernels.get(parameter["symbol"], [])
            if not sources:
                continue
            parameter["kernel_priors"] = sources
            parameter["kernel_prior_status"] = "pending"
            parameter["kernel_origin_source_concept_id"] = parameter.get("source_concept_id")
            parameter["prior_radius_intervals"] = deepcopy(parameter["radius_intervals"])
            self._inherited_domains[parameter["symbol"]] = deepcopy(parameter["radius_intervals"])
            parameter["radius_intervals"] = interval_union(
                interval for source in sources for interval in source["radius_intervals"])
        if self.parameters:
            self.status = "unresolved"

    def _retain_fitted_columns(self, structure, research_program):
        """Carry the paid readout through a parameter edit, without another fit."""
        from .structural_research_program import (
            research_program_fit_contract, structural_program_fit_contract)

        if research_program and research_program.get("term_nodes"):
            fit = research_program_fit_contract(research_program)
        elif structure and structure.get("fits") and "fit_terms" in structure["fits"][0]:
            fit = structural_program_fit_contract(self.program, structure)
        else:
            return
        if fit["program"].expanded().simplified() != self.program.expanded().simplified():
            raise ValueError("Interval readout belongs to a different response")
        terms = fit["fit_terms"]
        if self._parameter_terms is None:
            # Match the actual relation computation, including its measured
            # roles and predicate. Equal numerical radii alone do not bind it.
            keys = {item["symbol"]: {kernel_key(node) for node in
                    _parameter_relations(self.template, item["symbol"])}
                    for item in self.parameters}
            templates = []
            for term in terms:
                marked = term
                relations = _relations(term.expanded())
                for item in self.parameters:
                    related = {node for node in relations
                        if node.children[0] == Program("const", value=item["radius"])
                        and kernel_key(node) in keys[item["symbol"]]}
                    if related:
                        marked = _mark_relations(marked, related, item["symbol"])
                templates.append(marked)
            self._parameter_terms = templates
        bindings = {symbol: Program("const", value=value) for symbol, value in self.original.items()}
        if len(self._parameter_terms) != len(terms) or any(
                _bind(template, bindings).expanded().simplified() != term.expanded().simplified()
                for template, term in zip(self._parameter_terms, terms)):
            raise ValueError("Interval parameter columns differ from the saved readout")
        self._readout_fit = fit

    @property
    def intervals(self):
        return [{**item, "radius": self.values[item["symbol"]], "radius_interval": interval}
                for item in self.parameters for interval in item["radius_intervals"]]

    def _program(self, values):
        if values == self.original:
            return self.program, None
        new = {key: Program("const", value=value) for key, value in values.items()}
        body = _bind(self.template, new).simplified()
        if self.compact_replacement:
            return _replacement_call(self.program, body, f"parameter_{self.candidate_serial}"), None
        return body, None

    def _entry(self, values, interval, index, error, rows, cost):
        from .research_programs import _entry
        program, _ = self._program(values)
        parameters = [{**deepcopy(item), "radius": values[item["symbol"]]} for item in self.parameters]
        kernel_sources = []
        for item in parameters:
            if item.get("kernel_prior_status") not in {"pending", "supported"}:
                continue
            sources = list(dict.fromkeys(source["source_concept_id"] for source in item["kernel_priors"]
                if any(low <= item["radius"] <= high for low, high in source["radius_intervals"])))
            if not sources:
                continue
            item["kernel_prior_status"] = "supported"
            item["kernel_source_ids"] = sources
            if len(sources) == 1:
                item["source_concept_id"] = sources[0]
            kernel_sources.extend(sources)
        steps = [{"operation": "retain_composed_call", "question": "Retain the complete existing computation",
                  "program": self.program, "loss": error},
                 {"operation": "resolve_relation_intervals", "question": "Resolve a parameter inside its actual consuming context",
                  "program": program, "loss": error, "parameter_template": self.template.to_dict(),
                  "parameters": parameters, "parameter_index": index, "radius_interval": list(interval),
                  "evidence_contexts": deepcopy(self.evidence_contexts),
                  "parameter_terms": [term.to_dict() for term in self._parameter_terms]
                      if self._readout_fit is not None else [self.template.to_dict()],
                  "compact_replacement": self.compact_replacement}]
        fit = {"program": program, "terms": [program], "fit_terms": [program],
               "coefficients": [0., 1.], "loss": error}
        if self._readout_fit is not None:
            bindings = {key: Program("const", value=value) for key, value in values.items()}
            terms = [_bind(term, bindings).simplified() for term in self._parameter_terms]
            fit.update(terms=terms, fit_terms=terms,
                       coefficients=list(self._readout_fit["coefficients"]))
        entry = _entry(fit, steps, rows, cost, self.candidate_serial)
        self.candidate_serial += 1
        entry.update(parameter_index=index, parameter_values=dict(values),
                     validation_loss=error, selection_basis="all_acquired_evidence")
        if kernel_sources:
            entry["method_source_ids"] = list(dict.fromkeys(kernel_sources))
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
            if parameter.get("kernel_priors"):
                parameter["kernel_prior_status"] = "withdrawn"
                parameter["source_concept_id"] = parameter.get("kernel_origin_source_concept_id")
                parameter.pop("kernel_source_ids", None)
        self.evidence_contexts = [context for context in self.evidence_contexts
                                 if not any(context is old for old in withdrawn)]
        self._inherited_contexts = [context for context in self._inherited_contexts
                                   if not any(context is old for old in withdrawn)]
        self.evidence_updates.append({"reason": reason, "parameters": sorted(symbols),
            "withdrawn_trajectories": sum(len(context["trajectories"]) for context in withdrawn),
            "reopened_intervals": deepcopy(self.intervals)})
        self.status = "unresolved"
        return True

    def reuse_paid_partition(self, paid_rows, completed_entry):
        """Bind a completed same-scene partition to its actual paid prefix.

        An earlier complete radius pass already split these observations at
        every measured edge. Only appended observations can introduce a cut
        inside one of its supported cells. This is not a transferred-domain
        assumption: the executable answer, template and parameters must match
        this restored research, and later use keeps the exact paid row prefix.
        """
        self._paid_partition = None
        self.reuse_statistics["partition_rows_bound"] = 0
        payload = completed_entry.get("research_program") or {}
        saved = next((step for step in reversed(payload.get("selection_steps", []))
                      if step["operation"] == "resolve_relation_intervals"), None)
        index = completed_entry.get("parameter_index")
        if (saved is None or payload.get("fit_rows") != len(paid_rows)
                or index is None or not 0 <= index < len(self.parameters)
                or completed_entry["candidate"].program != self.program
                or saved["parameter_template"] != self.template.to_dict()
                or saved["parameters"] != self.parameters
                or completed_entry.get("parameter_values") != self.values
                or completed_entry.get("selection_basis") != "all_acquired_evidence"):
            return False
        self._remember_completed_partition(paid_rows, index)
        return True

    def _remember_completed_partition(self, paid_rows, index):
        """Retain one actually completed partition for the ordinary next pass."""
        self._paid_partition = {"rows": deepcopy(tuple(paid_rows)), "template": self.template,
            "parameters": deepcopy(self.parameters), "values": dict(self.values),
            "partitioned_symbol": self.parameters[index]["symbol"]}
        self.reuse_statistics["partition_rows_bound"] = len(paid_rows)

    def _new_geometry_rows(self, paid_rows, parameter):
        saved = self._paid_partition
        if (saved is None or self.template != saved["template"]
                or parameter["symbol"] != saved["partitioned_symbol"]):
            return paid_rows
        prefix = saved["rows"]
        if len(paid_rows) < len(prefix) or any(old != current for old, current in zip(prefix, paid_rows)):
            return paid_rows
        previous = next((item for item in saved["parameters"]
                         if item["symbol"] == parameter["symbol"]), None)
        if (previous is None or parameter.get("source_concept_id") != previous.get("source_concept_id")
                or any(self.values[symbol] != value for symbol, value in saved["values"].items()
                       if symbol != parameter["symbol"])
                or not all(any(old_low <= low and high <= old_high
                               for old_low, old_high in previous["radius_intervals"])
                           for low, high in parameter["radius_intervals"])):
            return paid_rows
        self.reuse_statistics["geometry_paid_rows_reused"] += len(prefix)
        return paid_rows[len(prefix):]

    def _supports_inherited_evidence(self, values, budget, tolerance, *,
                                     evaluated_program=None, evaluated_rows=(), evaluated_errors=None):
        # Historical targets belong to their historical complete response.
        # A new sum/window consumer inherits the parameter evidence, not those
        # targets as if they had been observed for today's output.
        for context in self._inherited_contexts:
            program = _bind(Program.from_dict(context["parameter_template"]), {
                key: Program("const", value=values.get(key, value))
                for key, value in context["parameters"].items()})
            same_response = (evaluated_program is not None and evaluated_errors is not None
                             and program.expanded().simplified() == evaluated_program.expanded().simplified())
            for trajectory in context["trajectories"]:
                errors, missing = [], []
                for payload in trajectory:
                    row = Observation(payload["features"], payload["target"], tuple(payload["history"]))
                    matched = next((current for current in evaluated_rows if current == row), None) if same_response else None
                    if matched is None:
                        missing.append(row)
                    else:
                        errors.append(evaluated_errors[id(matched)])
                        self.reuse_statistics["inherited_predictions_reused"] += 1
                if not budget.spend(len(missing)):
                    return None
                errors.extend((program.predict(row) - row.target) ** 2 for row in missing)
                error = sum(errors) / len(trajectory)
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
            if (len(domains) == 1 and domains[0][1] - domains[0][0] <= 1e-6
                    and parameter.get("kernel_prior_status") != "pending"):
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
            # Sparse paid rows still carry measured preparation frames used by
            # their temporal consumers. Those frames can expose a boundary
            # even when every selected current frame has zero interaction.
            geometry_depth = max((node.history_depth for node in relations), default=0)
            geometry_fields = sorted({name for node in relations for name, _ in node.observed_variables()})
            frames = {}
            for row in self._new_geometry_rows(paid_rows, parameter):
                history = row.history
                for position in range(max(0, len(history) - self.program.history_depth), len(history) + 1):
                    point = row.features if position == len(history) else history[position]
                    past = history[:position]
                    key = (tuple(point.get(name) for name in geometry_fields),
                        tuple(tuple(frame.get(name) for name in geometry_fields)
                        for frame in past[-geometry_depth:]) if geometry_depth else ())
                    frames.setdefault(key, Observation(point, 0., past))
            distances = _distances(relations, frames.values(), budget)
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
                            inherited = self._supports_inherited_evidence(values, budget, tolerance,
                                evaluated_program=program, evaluated_rows=evaluated_rows, evaluated_errors=errors)
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
                # A still-supported existing explanation must not disappear
                # merely because the sampled representatives are different.
                # Keep its exact parameter value; only observed counterexamples
                # justify replacing it. It cannot rescue a refuted open cell.
                retained = next((interval for interval in supported
                    if interval[0] <= self.original[parameter["symbol"]] <= interval[1]), None)
                original_in_domains = all(any(low <= self.original[item["symbol"]] <= high
                    for low, high in item["radius_intervals"]) for item in self.parameters)
                if retained is not None and original_in_domains and budget.spend(len(evaluated_rows)):
                    errors = {id(row): (self.program.predict(row) - row.target) ** 2 for row in evaluated_rows}
                    error = sum(errors[id(row)] for row in paid_rows) / len(paid_rows)
                    if (error <= tolerance and all(sum(errors[key] for key in trajectory) / len(trajectory)
                            <= tolerance for trajectory in trajectory_ids) and
                            self._supports_inherited_evidence(self.original, budget, tolerance,
                                evaluated_program=self.program, evaluated_rows=evaluated_rows,
                                evaluated_errors=errors)):
                        candidates = [item for item in candidates if item[0] != self.original]
                        candidates.insert(0, (dict(self.original), tuple(retained), error))
                self.status = "unresolved"
                entries = [self._entry(values, interval, index, error, len(paid_rows), budget.spent - start)
                           for values, interval, error in candidates]
                self._remember_completed_partition(paid_rows, index)
                return entries
            if self.reopen("current_relation_counterexample", parameter_index=index):
                return self.alternatives(paid_rows, budget, tolerance, experiments=experiments)
            # A source search range is not a physical limit. Once all of its
            # cells fail current measurements, keep the consuming computation
            # and test the range exposed by the current measured geometry.
            if distances:
                upper = distances[-1] + max(1., abs(distances[-1])) * 1e-6
                expanded = [0., upper]
                previous = self._expanded_domains.get(parameter["symbol"])
                if previous is None or upper > previous[1]:
                    self._expanded_domains[parameter["symbol"]] = expanded
                    parameter["radius_intervals"] = [expanded]
                    self.evidence_updates.append({"reason": "current_geometry_expands_refuted_domain",
                        "parameters": [parameter["symbol"]], "previous_intervals": deepcopy(domains),
                        "reopened_intervals": deepcopy(self.intervals), "evidence_rows": len(paid_rows)})
                    return self.alternatives(paid_rows, budget, tolerance, experiments=experiments)
            self.status = "refuted"
            return []
        self.status = "resolved_to_precision" if self.parameters else "no_relation_interval"
        return []

    def adopt(self, entry):
        self.values = dict(entry["parameter_values"])
        saved = entry["research_program"]["selection_steps"][-1]
        self.parameters = deepcopy(saved["parameters"])

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
