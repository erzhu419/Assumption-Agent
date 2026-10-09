"""Learn executable research DAGs from purchased, outcome-bearing action traces.

The book does not supply a catalogue of procedures. A procedure's nodes and
edits come from actions actually attempted by its caller; its scope is learned
from successful and unsuccessful uses. Probe coordinates describe control
roles and fractions of public ranges, rather than a previous task's names.
"""

from copy import deepcopy
import json
import math

from .models import Probe
from .growth_procedures import abstract_recipe, bind_subquestions
from .knowledge_returns import matching_returns, return_priority
from .language import Program


DIAGNOSTICS = ("alias_error", "residual_error", "prior_error", "repeat_variance", "feature_count")
PRIMITIVES = {"fit", "vary_controls", "measure", "compose", "condition", "replicate", "confirm"}


def _roles(context, bounds):
    roles = context.get("control_roles", {})
    if isinstance(roles, str):
        roles = json.loads(roles)
    if roles:
        return {name: roles.get(name, f"control:{i}") for i, name in enumerate(bounds)}
    schema = context.get("object_schema", {})
    if isinstance(schema, str):
        schema = json.loads(schema)
    result = {name: f"object:{i}:{role}"
              for i, fields in enumerate(schema.get("controls", {}).values())
              for role, name in fields.items()}
    if schema.get("focus_control"):
        result[schema["focus_control"]] = "focus"
    return {name: result.get(name, f"control:{i}") for i, name in enumerate(bounds)}


def _channel_names(channels):
    return [item if isinstance(item, str) else
            item["name"] if isinstance(item, dict) else item.name for item in channels]


def _recipe(event, diagnostics):
    recipe = deepcopy(event.get("recipe", {}))
    bounds = recipe.pop("feature_bounds", diagnostics.get("feature_bounds", {}))
    context = {**diagnostics, **recipe.pop("context", {})}
    roles = _roles(context, bounds)
    probe = recipe.pop("probe", event.get("probe"))
    sequence = recipe.pop("sequence", event.get("sequence"))
    if probe is not None:
        sequence = probe.sequence if isinstance(probe, Probe) else probe["sequence"]
    if sequence is not None:
        if not bounds:
            raise ValueError("learning an acquisition requires its public feature_bounds")
        recipe["normalized_rows"] = [
            {roles[name]: ((float(value) - bounds[name][0]) /
                           (bounds[name][1] - bounds[name][0])
                           if bounds[name][1] != bounds[name][0] else 0.)
             for name, value in row.items()} for row in sequence]
    channels = recipe.pop("channels", None)
    channel = recipe.pop("channel", event.get("channel"))
    if channels is None and channel is not None:
        channels = [channel]
    if channels is not None:
        names = _channel_names(diagnostics.get("channel_names", ()))
        recipe["channel_ranks"] = [names.index(name) for name in channels]
    # These identify the old acquisition, not its repeatable experimental step.
    recipe.pop("probe_id", None)
    recipe.pop("control_names", None)
    return abstract_recipe(recipe)


def _compile(trace, diagnostics):
    nodes, event_nodes, evidence_nodes, failed_events, failed_evidence = [], {}, {}, set(), set()
    for index, event in enumerate(trace):
        event_id = event.get("event_id", f"event_{index}")
        consumes, produces = list(event.get("consumes", ())), list(event.get("produces", ()))
        explicit = event.get("depends_on", ())
        unusable = (event.get("success", True) is False or
                    any(item in failed_evidence for item in consumes) or
                    any(item in failed_events for item in explicit))
        if unusable:
            failed_events.add(event_id)
            failed_evidence.update(produces)
            continue
        dependencies = [node_id for item in explicit for node_id in event_nodes.get(item, ())]
        dependencies += [node_id for item in consumes for node_id in evidence_nodes.get(item, ())]
        if not any(key in event for key in ("consumes", "produces", "depends_on")) and nodes:
            dependencies.append(nodes[-1]["node_id"])
        dependencies = list(dict.fromkeys(dependencies))
        if event["action"] not in PRIMITIVES:
            # Recording or naming evidence is not another research operation.
            # Preserve its ancestry when a subsequent operation consumes it.
            event_nodes[event_id] = dependencies
            for item in produces:
                evidence_nodes[item] = dependencies
                failed_evidence.discard(item)
            continue
        node_id = f"step_{len(nodes)}"
        action = event["action"]
        if action == "confirm" and event.get("details", {}).get("model_confirmed") is False:
            action = "vary_controls"
        node = {"node_id": node_id, "action": action,
                "depends_on": dependencies,
                "recipe": _recipe(event, diagnostics)}
        nodes.append(node)
        event_nodes[event_id] = [node_id]
        for item in produces:
            evidence_nodes[item] = [node_id]
            failed_evidence.discard(item)
    # A completed trace may contain useful but ultimately unused explorations.
    # Retain the evidence ancestry of the confirmation, not their sunk cost.
    confirmations = [node["node_id"] for node in nodes if node["action"] == "confirm"]
    if confirmations:
        by_id = {node["node_id"]: node for node in nodes}
        needed = set(confirmations)
        pending = list(confirmations)
        while pending:
            for dependency in by_id[pending.pop()]["depends_on"]:
                if dependency not in needed:
                    needed.add(dependency)
                    pending.append(dependency)
        nodes = [node for node in nodes if node["node_id"] in needed]
        renamed = {node["node_id"]: f"step_{i}" for i, node in enumerate(nodes)}
        for node in nodes:
            node["node_id"] = renamed[node["node_id"]]
            node["depends_on"] = [renamed[item] for item in node["depends_on"]]
    comparison_ancestors = {}
    for node in nodes:
        node["consumes"] = [f"{dependency}:evidence" for dependency in node["depends_on"]]
        node["produces"] = [f"{node['node_id']}:evidence"]
        ancestors = [comparison_ancestors[item] for item in node["depends_on"]
                     if item in comparison_ancestors]
        if node["recipe"].get("instrument_comparison", {}).get("binding") == "compare_available":
            comparison_ancestors[node["node_id"]] = node["node_id"]
        elif ancestors:
            comparison_ancestors[node["node_id"]] = ancestors[-1]
            if ("channel_ranks" in node["recipe"] and
                    node["recipe"].get("channel_binding") != "compare_available"):
                node["recipe"].pop("channel_ranks")
                node["recipe"]["channel_binding"] = "selected"
                node["recipe"]["selection_from"] = ancestors[-1]
    return nodes


def _example(diagnostics, success, task_id):
    values = {key: float(diagnostics[key]) for key in DIAGNOSTICS
              if key in diagnostics and math.isfinite(float(diagnostics[key]))}
    return {"diagnostics": values, "success": bool(success), "task_id": task_id}


def _scope(method):
    """Conjoin measured separating thresholds; never infer scope from task IDs."""
    positives = [item["diagnostics"] for item in method["examples"] if item["success"]]
    negatives = [item["diagnostics"] for item in method["examples"] if not item["success"]]
    conditions = []
    unresolved = False
    for negative in negatives:
        separator = None
        for key in DIAGNOSTICS:
            values = [item[key] for item in positives if key in item]
            if key not in negative or len(values) != len(positives) or not values:
                continue
            if negative[key] < min(values):
                separator = {"feature": key, "op": ">", "value": (negative[key] + min(values)) / 2.}
            elif negative[key] > max(values):
                separator = {"feature": key, "op": "<", "value": (negative[key] + max(values)) / 2.}
            if separator is not None:
                break
        if separator is None:
            unresolved = True
        elif separator not in conditions:
            conditions.append(separator)
    method["applicability"] = {"conditions": conditions,
        "status": "conflicted" if unresolved else "contrastive" if negatives else "positive_only"}
    method["status"] = "supported" if positives and method["nodes"] else "unconfirmed"


def _matches(method, diagnostics):
    if method["status"] != "supported" or method["applicability"]["status"] == "conflicted":
        return False
    if "feature_bounds" in diagnostics:
        available = set(_roles(diagnostics, diagnostics["feature_bounds"]).values())
        required = {role for node in method["nodes"]
                    for row in node["recipe"].get("normalized_rows", ()) for role in row}
        if not required <= available:
            return False
    if "channel_names" in diagnostics:
        count = len(diagnostics["channel_names"])
        if any(index >= count for node in method["nodes"]
               if node["recipe"].get("channel_binding") != "compare_available"
               for index in node["recipe"].get("channel_ranks", ())):
            return False
    for rule in method["applicability"]["conditions"]:
        value = diagnostics.get(rule["feature"])
        if value is None or not math.isfinite(float(value)):
            return False
        if rule["op"] == ">" and not float(value) > rule["value"]:
            return False
        if rule["op"] == "<" and not float(value) < rule["value"]:
            return False
    return True


class ResearchMethodBook:
    """Persistent editable workflows, with evidence-limited applicability."""

    def __init__(self, methods=None, *, next_method_index=None):
        self.methods = deepcopy(methods or [])
        indices = [int(item["method_id"].removeprefix("method_")) for item in self.methods
                   if item["method_id"].removeprefix("method_").isdigit()]
        self.next_method_index = (next_method_index if next_method_index is not None
                                  else max(indices, default=-1) + 1)

    def to_dict(self):
        return {"methods": deepcopy(self.methods), "next_method_index": self.next_method_index}

    @classmethod
    def from_dict(cls, payload):
        return cls(payload.get("methods", []), next_method_index=payload.get("next_method_index"))

    def propose(self, diagnostics, *, candidates=None):
        eligible = [method for method in (self.methods if candidates is None else candidates)
                    if _matches(method, diagnostics)]
        if not eligible:
            return None
        # A contrastively supported scope is stronger than an untested transfer;
        # within that scope, measured output returns precede recency.
        eligible.sort(key=lambda method: (
            method["applicability"]["status"] != "contrastive",
            self._knowledge_priority(method, diagnostics), -method["version"],
            -self.methods.index(method)))
        return deepcopy(eligible[0])

    @staticmethod
    def _exclusive_returns(outcome):
        """Keep receipts whose complete measured offer has one actual origin."""
        exclusive = {(item.get("comparison_id"), item.get("lineage_id"),
                      Program.from_dict(item["program"]).expanded().simplified())
                     for item in outcome["comparisons"] if item["exclusive"]}
        return [receipt for receipt in outcome.get("source_returns", ())
                if len(receipt.get("joint_source_ids", ())) <= 1
                and receipt.get("comparisons")
                and all((item.get("comparison_id"), receipt.get("lineage_id"),
                    Program.from_dict(item["program"]).expanded().simplified()) in exclusive
                    for item in receipt["comparisons"])]

    def knowledge_value(self, method, diagnostics):
        """Price one observed downstream demand within the method's own scope.

        Method execution applicability and its consumers' response conditions
        are different questions. Each explicit demand identifies already paid
        evidence from an unfinished consumer; different conditions retain
        separate receipts instead of becoming a multi-task total gain.
        """
        result = {"priority": (0., 0.), "matched_demands": [], "selected_demand": None,
                  "ignored_demands": [], "basis": "no_actionable_matching_return"}
        if not _matches(method, diagnostics):
            result["basis"] = "method_not_applicable"
            return result
        explicit = "downstream_demands" in diagnostics
        demands = diagnostics.get("downstream_demands", [])
        if not explicit and diagnostics.get("transfer_condition") is not None:
            demands = [{"task_position": None, "condition": diagnostics["transfer_condition"],
                        "basis": "current_paid_condition"}]
        history = []
        for outcome in method.get("knowledge_outcomes", ()):
            production = outcome["production_diagnostics"]
            if _matches(method, production):
                for receipt in self._exclusive_returns(outcome):
                    if receipt not in history:
                        history.append(receipt)
        actionable_demands = []
        for demand in demands:
            condition = demand.get("condition")
            if (condition is None or demand.get("status") in {"completed", "resolved", "solved"}
                    or (explicit and demand.get("paid_rows", 0) <= 0)):
                result["ignored_demands"].append({**deepcopy(demand),
                    "reason": "no_unfinished_paid_demand"})
                continue
            receipts = matching_returns(history, condition)
            if not receipts:
                result["ignored_demands"].append({**deepcopy(demand),
                    "reason": "no_matching_returns"})
                continue
            valued = [receipt for receipt in receipts
                      if receipt.get("confirmation_status", "confirmed") == "confirmed"
                      or receipt.get("retained_normalized_gain", 0.) < 0.]
            matched = {**deepcopy(demand), "priority": return_priority(receipts, condition),
                "actionable": bool(valued), "receipts": [{key: deepcopy(receipt.get(key))
                    for key in ("source_id", "template_id", "episode_id", "lineage_id",
                                "confirmation_status", "value_status", "retained_normalized_gain",
                                "adaptation_compute", "comparison_compute")} for receipt in receipts]}
            result["matched_demands"].append(matched)
            if valued:
                actionable_demands.append(matched)
        if actionable_demands:
            selected = min(actionable_demands, key=lambda demand: demand["priority"])
            result.update(priority=selected["priority"], selected_demand=deepcopy(selected),
                          basis="paid_downstream_return")
        return result

    def _knowledge_priority(self, method, diagnostics):
        return self.knowledge_value(method, diagnostics)["priority"]

    def observe_knowledge_transfer(self, production, outcome, *, return_history):
        """Record a produced program's measured returns, not execution success.

        Source receipts retain their existing confirmation and condition rules.
        A later replicate may confirm earlier receipts; refresh their snapshots
        without changing the method's execution examples or applicability.
        """
        method = next((item for item in self.methods
                       if item["method_id"] == production["method_id"]), None)
        if method is None:
            return None
        record = {**deepcopy(outcome), "kind": "research_workflow_return",
                  "method_id": production["method_id"],
                  "production_task": production["source_task"],
                  "production_diagnostics": deepcopy(production.get("diagnostics", {})),
                  "measured_retained_gain": outcome["retained_gain"]}
        outcomes = method.setdefault("knowledge_outcomes", [])
        outcomes.append(record)
        for stored in outcomes:
            if stored["source_concept_id"] != outcome["source_concept_id"]:
                continue
            stored["source_returns"] = deepcopy([item for item in return_history
                if item.get("kind") == "research_return"
                and item.get("episode_id") == stored["episode_id"]])
            valued = [item for item in self._exclusive_returns(stored)
                if item.get("confirmation_status", "confirmed") == "confirmed"
                or item.get("retained_normalized_gain", 0.) < 0.]
            stored["retained_gain"] = (sum(item["retained_normalized_gain"] for item in valued)
                                       / len(valued) if valued else None)
        return deepcopy(record)

    def learn(self, trace, diagnostics, *, success, task_id):
        nodes = _compile(trace, diagnostics)
        for method in reversed(self.methods):
            if method["nodes"] == nodes:
                method["examples"].append(_example(diagnostics, success, task_id))
                _scope(method)
                return deepcopy(method)
        method = {"method_id": f"method_{self.next_method_index}", "parent_method_id": None,
                  "version": 1, "nodes": nodes,
                  "examples": [_example(diagnostics, success, task_id)]}
        self.next_method_index += 1
        _scope(method)
        self.methods.append(method)
        return deepcopy(method)

    def revise(self, method_id, trace, diagnostics, *, success, task_id):
        parent = next(method for method in self.methods if method["method_id"] == method_id)
        nodes = _compile(trace, diagnostics)
        if nodes == parent["nodes"]:
            raise ValueError("a method revision must change an executable step or dependency")
        child = {"method_id": f"method_{self.next_method_index}", "parent_method_id": method_id,
                 "version": parent["version"] + 1, "nodes": nodes,
                 "examples": [_example(diagnostics, success, task_id)]}
        self.next_method_index += 1
        _scope(child)
        self.methods.append(child)
        return deepcopy(child)

    def record_outcome(self, method_id, diagnostics, *, success, task_id):
        """Update scope from a measured replay, independently of later edits."""
        method = next(method for method in self.methods if method["method_id"] == method_id)
        method["examples"].append(_example(diagnostics, success, task_id))
        _scope(method)
        return deepcopy(method)


def instantiate(method, spec, channels=(), *, attempt=0):
    """Bind a learned DAG to current public geometry and instrument ranks.

    Returned actions are topologically ordered. The caller executes them, pays
    for every acquisition and fit, and supplies the resulting trace back to the
    book. Increment ``attempt`` on repeated executions so stochastic apparatuses
    acquire a new realization rather than replaying an old probe identifier.
    """
    bounds = spec.feature_bounds
    names_by_role = {role: name for name, role in _roles(spec.context, bounds).items()}
    channel_names = _channel_names(channels)
    actions = []
    for node in method["nodes"]:
        action = deepcopy(node)
        recipe = action["recipe"]
        if recipe.get("channel_binding") == "compare_available":
            action["channels"] = tuple(channel_names)
        elif recipe.get("channel_binding") == "selected":
            action["channels_from"] = recipe["selection_from"]
        elif "channel_ranks" in recipe:
            action["channels"] = tuple(channel_names[index] for index in recipe["channel_ranks"])
        if "normalized_rows" in recipe:
            sequence = []
            for row in recipe["normalized_rows"]:
                point = {name: (low + high) / 2. for name, (low, high) in bounds.items()}
                for role, fraction in row.items():
                    name = names_by_role[role]
                    low, high = bounds[name]
                    point[name] = low + float(fraction) * (high - low)
                sequence.append(point)
            action["probe"] = Probe(
                f"{spec.task_id}:{method['method_id']}:{attempt}:{node['node_id']}", tuple(sequence))
        actions.append(bind_subquestions(action, channel_names))
    return actions
