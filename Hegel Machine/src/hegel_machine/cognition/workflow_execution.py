"""Record and execute research workflows inside the unified research loop.

The session owns allocation and knowledge feedback. This recorder uses its
agent's acquisitions, view projection, response search and confirmation gate;
it does not own a second researcher or an independent observation ledger.
"""

from copy import deepcopy
import math

from .growth_methods import instantiate, _compile


class WorkflowEpisode:
    def __init__(self, book, command_spec, channels, *, context=None, attempt=0):
        self.book, self.spec, self.channels = book, command_spec, tuple(channels)
        self.diagnostics = {"feature_bounds": command_spec.feature_bounds,
            "channel_names": [channel.name for channel in channels],
            "feature_count": len(command_spec.feature_names), **(context or {})}
        self.attempt = attempt
        self.trace, self.locations, self.fit_ids, self.confirmations = [], {}, {}, []
        self.replayed = self.replay_program = self.replay_fit_id = self.method_id = None
        self.feedback_recorded = self.replay_failed = False
        self._row_owners = []

    def _event(self, action, *, consumes=(), recipe=None, **details):
        identifier = f"workflow_event_{len(self.trace)}"
        self.trace.append({"event_id": identifier, "action": action,
            "consumes": list(consumes), "produces": [identifier], "success": True,
            "recipe": recipe or {}, "details": details})
        return identifier

    def acquired(self, probe, channels, rows, *, purpose, command_rows=()):
        self._row_owners.extend(rows)
        self._row_owners.extend(command_rows)
        identifier = self._event("measure" if channels else "vary_controls",
            recipe={"sequence": list(probe.sequence), "channels": list(channels)}, purpose=purpose)
        for index, row in enumerate(rows):
            self.locations[id(row)] = identifier, index
        for index, row in enumerate(command_rows):
            self.locations[id(row)] = identifier, index

    def projected(self, originals, projected):
        self._row_owners.extend(projected)
        for original, row in zip(originals, projected):
            if id(original) in self.locations:
                self.locations[id(row)] = self.locations[id(original)]

    def _inputs(self, rows, dependencies):
        result = []
        for row in rows:
            if id(row) not in self.locations:
                return None
            identifier, index = self.locations[id(row)]
            if identifier not in dependencies:
                dependencies.append(identifier)
            dependency = dependencies.index(identifier)
            if not result or result[-1]["dependency"] != dependency:
                result.append({"dependency": dependency, "rows": []})
            result[-1]["rows"].append(index)
        return result

    def fitted(self, claims, training, validation, spec):
        if not claims or not training or not validation:
            return None
        dependencies = []
        train = self._inputs(training, dependencies)
        valid = self._inputs(validation, dependencies)
        if train is None or valid is None:
            return None
        names = self.diagnostics["channel_names"]
        view = spec.context.get("measurement_view")
        selected = ([view] if view in names else [name for name in names if name in spec.feature_names])
        identifier = self._event("fit", consumes=dependencies, recipe={
            "solver": "unified_research_view", "channels": selected,
            "training_inputs": train, "validation_inputs": valid,
            "risk_quantile": spec.risk_quantile})
        for claim in claims:
            self.fit_ids[claim["candidate"].program] = identifier
        return identifier

    def frozen(self, claims, training, validation, spec):
        missing = [claim for claim in claims if claim is not None and
                   claim["candidate"].program not in self.fit_ids]
        if missing:
            self.fitted(missing, training, validation, spec)

    def tested(self, program, rows, measured_loss, *, accepted=None):
        if program not in self.fit_ids or measured_loss is None:
            return
        ids = {self.locations[id(row)][0] for row in rows if id(row) in self.locations}
        if len(ids) != 1:
            return
        identifier = next(iter(ids))
        self.confirmations.append((program, identifier, float(measured_loss),
                                   self.fit_ids[program], accepted))

    def _replay_plan(self, method, *, legal_views, projectable, limits, confirmation_length):
        """Check executable bindings and known acquisition/confirmation demand.

        Fit search is paid by the native SearchBudget during execution. The
        row-based compute floor here is not a promise that search will finish.
        """
        # Geometry and available representations are preconditions of an
        # executable plan, checked before buying observations.
        try:
            steps = instantiate(method, self.spec, self.channels, attempt=self.attempt)
        except (KeyError, IndexError):
            return None, "geometry_or_instrument_unavailable"
        views = {tuple(view) for view in legal_views}
        fits = [step for step in steps if step["action"] == "fit"]
        supported = all(step["recipe"].get("solver") == "unified_research_view" or
                        step["recipe"].get("estimator") == "residual_expression_search" for step in fits)
        supported &= all(step["recipe"].get("risk_quantile", .5) == self.spec.risk_quantile
                         for step in fits)
        queries = [step for step in steps if "probe" in step and step["action"] != "confirm"]
        for step in queries:
            wanted = tuple(step.get("channels", ()))
            if wanted not in views and step["recipe"].get("channel_binding") == "compare_available":
                alternatives = [view for view in views if projectable is not None and
                                all((name,) in projectable(view) for name in wanted)]
                if alternatives:
                    step["channels"] = min(alternatives, key=lambda view: sum(
                        channel.cost_per_row for channel in self.channels if channel.name in view))
            if "channels_from" in step:
                # The choice is resolved only after its predecessor fits.
                continue
            supported &= tuple(step.get("channels", ())) in views
        # A trajectory is one reset experiment: never truncate/split a stored
        # temporal acquisition merely to satisfy a new maximum length.
        if (not fits or not supported or not queries or
                any(step["probe"].cost > self.spec.max_probe_length for step in queries)):
            return None, "trajectory_view_or_solver_not_available"
        rates = {channel.name: channel.cost_per_row for channel in self.channels}
        rows_needed = sum(step["probe"].cost for step in queries)
        # Reserve only representations this DAG can actually select. Merely
        # offering a priced instrument cannot make a command-only procedure
        # require a measurement budget. Dynamic comparisons reserve their
        # most expensive attainable outcome until paid fitting selects one.
        choices, acquired_views, confirmation_views = {}, set(), set()
        measurement_needed = 0
        for step in steps:
            if step["action"] == "confirm":
                continue
            bound_views = (choices[step["channels_from"]] if "channels_from" in step else
                           {tuple(step.get("channels", ()))})
            if "probe" in step:
                measurement_needed += step["probe"].cost * max(
                    (sum(rates[name] for name in view) for view in bound_views), default=0)
                acquired_views.update(bound_views)
            if step["action"] == "fit":
                alternatives = {tuple(item["channels"]) for item in step.get("subquestions", ())}
                if not alternatives:
                    alternatives = (bound_views if "channels" in step or "channels_from" in step
                                    else acquired_views)
                choices[step["node_id"]] = alternatives & views
                confirmation_views.update(choices[step["node_id"]])
        confirmation_rate = max((sum(rates[name] for name in view)
                                 for view in confirmation_views), default=0)
        if (rows_needed + confirmation_length > limits["observation_remaining"] or
                measurement_needed + confirmation_length * confirmation_rate > limits["measurement_remaining"] or
                2 * (rows_needed + confirmation_length) > limits["compute_remaining"]):
            return None, "execution_and_confirmation_not_funded"
        return {"steps": steps, "observation_rows": rows_needed,
                "measurement_cost": measurement_needed}, None

    def replay(self, *, acquire, project, fit, legal_views, remaining,
               confirmation_length, threshold, diagnostics, actions, projectable=None):
        self.diagnostics.update({key: value for key, value in diagnostics.items()
                                 if isinstance(value, (int, float)) and math.isfinite(value)})
        for key in ("transfer_condition", "downstream_demands"):
            if key in diagnostics:
                self.diagnostics[key] = deepcopy(diagnostics[key])
        views = {tuple(view) for view in legal_views}
        limits = remaining()
        plans, alternatives = {}, []
        for candidate in self.book.methods:
            plan, reason = self._replay_plan(candidate, legal_views=views,
                projectable=projectable, limits=limits, confirmation_length=confirmation_length)
            if plan is None:
                actions.append({"action": "defer_research_workflow",
                    "method_id": candidate["method_id"], "reason": reason})
                continue
            plans[candidate["method_id"]] = plan
            alternatives.append(candidate)
        method = self.book.propose(self.diagnostics, candidates=alternatives)
        if method is None:
            return None
        plan = plans[method["method_id"]]
        steps, rows_needed = plan["steps"], plan["observation_rows"]
        self.replayed = method
        value = self.book.knowledge_value(method, self.diagnostics)
        actions.append({"action": "replay_research_workflow", "method_id": method["method_id"],
                        "nodes": len(steps), "planned_observation_rows": rows_needed,
                        "diagnostics": deepcopy(self.diagnostics),
                        "knowledge_priority": list(value["priority"]),
                        "selection_value": deepcopy(value),
                        "alternatives": [{"method_id": candidate["method_id"],
                            "selection_value": self.book.knowledge_value(candidate, self.diagnostics),
                            "planned_observation_rows": plans[candidate["method_id"]]["observation_rows"]}
                            for candidate in alternatives]})
        evidence, choices, selected, purchased = {}, {}, None, []
        for step in steps:
            if step["action"] == "confirm":
                continue  # The native agent buys and decides fresh confirmation.
            channels = (choices[step["channels_from"]] if "channels_from" in step else
                        tuple(step.get("channels", ())))
            if "probe" in step:
                observed = acquire(channels, "replay_research_workflow", probe=step["probe"])
                if observed is None:
                    break
                evidence[step["node_id"]] = [(observed, channels)]
                purchased.append((observed, channels))
            if step["action"] != "fit":
                continue
            batches = []
            for dependency in step["depends_on"]:
                for observations, source in evidence.get(dependency, ()):
                    if not any(observations is existing[0] for existing in batches):
                        batches.append((observations, source))
            native = step["recipe"].get("solver") == "unified_research_view"
            subsets = [tuple(item["channels"]) for item in step.get("subquestions", ())]
            if not subsets:
                subsets = [channels] if native or "channels" in step or "channels_from" in step else [
                    tuple(batches[-1][1]) if batches else ()]
            trials = []
            for subset in subsets:
                if subset not in views or any(projectable is not None and subset not in projectable(source)
                                               for _, source in batches):
                    continue
                if native:
                    def inputs(kind):
                        rows = []
                        for item in step["recipe"][kind]:
                            dependency = step["depends_on"][item["dependency"]]
                            observations, source = evidence[dependency][0]
                            view = project(observations, source, subset)
                            rows.extend(view[index] for index in item["rows"])
                        return rows
                    training, validation = inputs("training_inputs"), inputs("validation_inputs")
                else:
                    data = [row for observations, source in batches for row in project(observations, source, subset)]
                    training = [row for index, row in enumerate(data) if index % 4 != 3]
                    validation = [row for index, row in enumerate(data) if index % 4 == 3]
                if not training or not validation:
                    continue
                candidates = fit(subset, training, validation)
                best = min(candidates, key=lambda item: (item["validation_loss"],
                           item["candidate"].program.size)) if candidates else None
                if best is not None:
                    trials.append((best, subset, candidates, training, validation))
            if trials:
                selected = min(trials, key=lambda item: (item[0]["validation_loss"],
                               item[0]["candidate"].program.size))
                choices[step["node_id"]] = selected[1]
                evidence[step["node_id"]] = batches
                if len(trials) > 1:
                    receipt = next((event for event in self.trace if event["event_id"] ==
                        self.fit_ids.get(selected[0]["candidate"].program)), None)
                    if receipt is not None:
                        receipt["recipe"]["instrument_comparison"] = {
                            "candidate_channels": [name for item in trials for name in item[1]],
                            "losses": {item[1][-1]: item[0]["validation_loss"] for item in trials if item[1]}}
                    actions.append({"action": "compare_research_workflow_views",
                        "method_id": method["method_id"], "selected_channels": list(selected[1]),
                        "validation_losses": [{"channels": list(item[1]),
                            "loss": item[0]["validation_loss"]} for item in trials]})
            else:
                break
        if selected is not None:
            self.replay_program = selected[0]["candidate"].program
            self.replay_fit_id = self.fit_ids.get(self.replay_program)
            adequate = max(selected[0]["training_loss"], selected[0]["validation_loss"]) <= threshold
            if not adequate:
                self.book.record_outcome(method["method_id"], self.diagnostics,
                                         success=False, task_id=self.spec.task_id)
                self.feedback_recorded = self.replay_failed = True
                actions.append({"action": "refute_research_workflow", "method_id": method["method_id"],
                    "basis": "paid_validation", "validation_loss": selected[0]["validation_loss"]})
            return {"selected": selected, "adequate": adequate, "purchased": purchased}
        return {"selected": None, "adequate": False, "purchased": purchased}

    def finish(self, program, *, confirmed, threshold, actions, concept=None):
        chosen_confirmation = None
        for candidate, identifier, measured_loss, frozen_fit_id, native_accepted in self.confirmations:
            accepted = bool(native_accepted if native_accepted is not None else measured_loss <= threshold)
            event = next(item for item in self.trace if item["event_id"] == identifier)
            event.update(action="confirm", consumes=[frozen_fit_id],
                         details={**event["details"], "model_confirmed": accepted})
            if accepted and confirmed and candidate == program:
                chosen_confirmation = identifier
            if (self.replayed is not None and not self.feedback_recorded and
                    candidate == self.replay_program and frozen_fit_id == self.replay_fit_id):
                self.book.record_outcome(self.replayed["method_id"], self.diagnostics,
                                         success=accepted, task_id=self.spec.task_id)
                self.feedback_recorded, self.replay_failed = True, not accepted
                actions.append({"action": "confirm_research_workflow" if accepted else "refute_research_workflow",
                    "method_id": self.replayed["method_id"], "basis": "native_fresh_confirmation",
                    "confirmation_loss": measured_loss})
        if not confirmed or chosen_confirmation is None:
            return None
        # A repaired response alone is not a changed executable procedure.
        # Preserve that distinction when a native fit finds different weights
        # from the same experiment DAG.
        failed_parent = bool(self.replayed and self.replay_failed and
                             _compile(self.trace, self.diagnostics) != self.replayed["nodes"])
        if failed_parent:
            method = self.book.revise(self.replayed["method_id"], self.trace, self.diagnostics,
                                      success=True, task_id=self.spec.task_id)
        else:
            method = self.book.learn(self.trace, self.diagnostics, success=True, task_id=self.spec.task_id)
        self.method_id = method["method_id"]
        actions.append({"action": "revise_research_workflow" if failed_parent else "compile_research_workflow",
                        "method_id": self.method_id, "nodes": len(method["nodes"]),
                        "parent_method_id": method.get("parent_method_id")})
        if concept is not None:
            production = {"kind": "research_workflow", "method_id": self.method_id,
                          "source_task": self.spec.task_id, "source_concept_id": concept.concept_id,
                          "diagnostics": deepcopy(self.diagnostics)}
            if production not in concept.production_methods:
                concept.production_methods.append(production)
        return method
