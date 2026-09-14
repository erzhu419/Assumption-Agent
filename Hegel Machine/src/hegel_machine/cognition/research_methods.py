"""Learn and revise executable research sequences from their later returns."""

from copy import deepcopy
from statistics import mean

from .method_conditions import method_applicability, same_research_condition
from .method_revision import distinguishing_experiment, revision_condition


class ResearchMethods:
    def __init__(self, state=None):
        state = state or {}
        self.methods = deepcopy(state.get("methods", []))
        self.revisions = deepcopy(state.get("revisions", []))
        self.last_choice = None

    @staticmethod
    def _completed(method, trace):
        return [step["action"] for step in trace if step.get("method_id") == method["method_id"]]

    @staticmethod
    def _outcome(step, *, source_task, kind):
        return {"action": step["action"], "profile": deepcopy(step.get("profile")),
                "gain": step["gain"], "observation_cost": step["observation_cost"],
                "compute_cost": step["compute_cost"], "source_task": source_task, "kind": kind}

    def observe_knowledge_transfer(self, production, outcome):
        """Measure later use of a method's output, separately from the producing task.

        The outcome describes actually paid source-local work and paired
        prediction comparisons. It is an observed production route's return,
        rather than the causal effect of removing that method from all research.
        """
        method = next((item for item in self.methods if item["method_id"] == production["method_id"]), None)
        if method is None:
            return None
        record = {**deepcopy(outcome), "method_id": production["method_id"],
            "production_action": production["action"], "production_profile": deepcopy(production.get("profile")),
            "production_task": production["source_task"]}
        method.setdefault("knowledge_outcomes", []).append(record)
        self.revisions.append({"method_id": method["method_id"], "change": "measure_produced_knowledge_return",
            "source_concept_id": outcome["source_concept_id"], "downstream_task": outcome["downstream_task"],
            "retained_gain": outcome["retained_gain"], "exclusive_compute": outcome["exclusive_compute"]})
        return deepcopy(record)

    def _with_knowledge_return(self, method, action, profile, immediate):
        records = [row for row in method.get("knowledge_outcomes", []) if row["production_action"] == action]
        exact = [row for row in records if same_research_condition(row.get("production_profile"), profile, action)]
        matched = exact or [row for row in records if method_applicability(
            row.get("production_profile"), profile, action)["status"] == "supported"]
        conditions = []
        for row in matched:
            for condition in row["downstream_conditions"]:
                if condition not in conditions:
                    conditions.append(deepcopy(condition))
        # These are later consumer tasks. Their observed distribution prices
        # the output of a matching production step; it is not evidence that
        # the current producer has the same response or control coverage.
        knowledge = {"gain": mean(row["retained_gain"] or 0. for row in matched) if matched else 0.,
            "compute_cost": mean(row["exclusive_compute"] for row in matched) if matched else 0.,
            "trials": sum(row["retained_gain"] is not None for row in matched),
            "unmeasured_trials": sum(row["retained_gain"] is None for row in matched),
            "downstream_conditions": conditions,
            "basis": "observed_output_return_under_matching_production_conditions" if matched else "no_matching_output_return"}
        return {**immediate, "knowledge_return": knowledge,
            "investment_gain": immediate["gain"] + knowledge["gain"],
            "investment_compute_cost": immediate["compute_cost"] + knowledge["compute_cost"]}

    def _return(self, method, trace, profile, *, continuation=False):
        """Use returns acquired under the proposed step's evidence conditions.

        A later neutral result in a different condition does not erase an
        earlier useful result. A measured complete continuation replaces the
        source step's promise when pricing another continuation of that kind.
        """
        index = len(self._completed(method, trace))
        if index >= len(method["steps"]):
            return None
        action = method["steps"][index]
        records = [row for row in method.get("step_outcomes", []) if row["action"] == action]
        exact = [row for row in records if same_research_condition(row.get("profile"), profile, action)]
        matched = exact or [row for row in records if
            method_applicability(row.get("profile"), profile, action)["status"] == "supported"]
        transferred = [row for row in matched if row["kind"] == "transfer"]
        matched = transferred or matched
        basis = "conditioned_step_return" if matched else "source_sequence_return"
        sequence_gain = None
        if len(method["steps"]) - index > 1:
            sequences = method.get("sequence_outcomes", [])
            exact_sequences = [row for row in sequences if same_research_condition(row.get("profile"), profile, action)]
            applicable_sequences = exact_sequences or [row for row in sequences if
                method_applicability(row.get("profile"), profile, action)["status"] == "supported"]
            transferred_sequences = [row for row in applicable_sequences if row["kind"] == "transfer"]
            applicable_sequences = transferred_sequences or applicable_sequences
            if applicable_sequences:
                spent_gain = sum(step["gain"] for step in trace if step.get("method_id") == method["method_id"])
                sequence_gain = max(-1., min(1., mean(row["gain"] for row in applicable_sequences) - spent_gain))
                basis = "remaining_sequence_return"
            elif not sequences:
                sequence_gain = method["observed_gain"]
        if continuation:
            returns = [row for row in method.get("continuation_outcomes", []) if
                       row["action"] == action and same_research_condition(row.get("profile"), profile, action)]
            if returns:
                matched, basis, sequence_gain = returns, "measured_continuation_return", None
        if matched:
            def work(row):
                source = row.get("profile") or {}
                ratio = ((profile or {}).get("observation_count", 0) / source["observation_count"]
                         if source.get("observation_count") and (profile or {}).get("observation_count") else 1.)
                return row["compute_cost"] * ratio
            return self._with_knowledge_return(method, action, profile, {
                    "gain": sequence_gain if sequence_gain is not None else mean(row["gain"] for row in matched),
                    "observation_cost": mean(row["observation_cost"] for row in matched),
                    "compute_cost": mean(work(row) for row in matched), "basis": basis,
                    "trials": len(matched), "transferred": any(row["kind"] == "transfer" for row in matched)})
        recorded = method.get("step_costs", [])
        costs = recorded[index] if index < len(recorded) else {
            "observation_cost": method["mean_observation_cost"], "compute_cost": method["mean_compute_cost"]}
        sources = method.get("step_profiles", [])
        source = sources[index] if index < len(sources) else None
        ratio = profile["observation_count"] / source["observation_count"] if (
            profile and source and source.get("observation_count")) else 1.
        return self._with_knowledge_return(method, action, profile, {
                "gain": method["observed_gain"], "observation_cost": costs["observation_cost"],
                "compute_cost": costs["compute_cost"] * ratio, "basis": basis, "trials": 0,
                "transferred": method["status"] == "supported" and not records})

    def observe_continuation(self, method_id, profile, feedback, *, source_task):
        """Learn what an executed method-backed revisit actually returned.

        The caller compares the saved and final predictors on common new
        evidence. This prices the complete revisit; it does not claim that its
        improvement was caused by this method rather than another action.
        """
        if not feedback.get("reference_rows") or feedback.get("execution_complete") is False:
            return None
        method = next(item for item in self.methods if item["method_id"] == method_id)
        record = {"action": feedback.get("method_action", method["steps"][0]),
                  "profile": deepcopy(profile), "gain": max(-1., min(1., feedback["normalized_gain"])),
                  "raw_gain": feedback["gain"], "reference_rows": feedback["reference_rows"],
                  "observation_cost": feedback["observation_cost"], "compute_cost": feedback["compute_cost"],
                  "source_task": source_task, "kind": "transfer"}
        method.setdefault("continuation_outcomes", []).append(record)
        self.revisions.append({"method_id": method_id, "change": "measure_complete_continuation_return",
                               "gain": record["gain"], "source_task": source_task})
        return deepcopy(record)

    def _applicability(self, method, trace, profile):
        completed = self._completed(method, trace)
        if completed != method["steps"][:len(completed)] or len(completed) >= len(method["steps"]):
            return None
        action = method["steps"][len(completed)]
        if action in {"task_value", "representative"} and revision_condition(profile):
            # A measured joint plan can still buy necessary observations. A
            # standalone coverage promise cannot keep reselling a failed fit.
            if not any(step in {"representation", "continuous_state", "state_transition", "structure"}
                       for step in method["steps"][len(completed) + 1:]):
                return None
        profiles = method.get("step_profiles", [])
        source = profiles[len(completed)] if len(profiles) > len(completed) else None
        applicability = method_applicability(source, profile, action)
        if applicability["status"] == "explore":
            # A successful later transfer teaches a new applicability condition;
            # the method is not forever confined to its first source profile.
            supported = [method_applicability(row.get("profile"), profile, action)
                         for row in method.get("step_outcomes", [])
                         if row["action"] == action and row["gain"] > 0]
            applicability = next((item for item in supported if item["status"] == "supported"), applicability)
        if applicability["status"] == "unsupported":
            return None
        # Target feedback belongs to the action and evidence, not the name of
        # whichever source method happened to recommend it. A different method
        # ID cannot resell the same failed fit. An unfinished sequence can still
        # take a different next step after a neutral preparation.
        for step in reversed(trace):
            if step["action"] != action or step["gain"] > 1e-8 or not step.get("execution_complete", True):
                continue
            prior = step.get("profile")
            if same_research_condition(prior, profile, action):
                return None
        for failed in method.get("failed_profiles", []):
            if same_research_condition(failed, profile, action):
                return None
        return applicability

    def observe(self, context, trace, *, condition, source_task):
        """Learn measured successful steps and their actual prerequisites.

        The shorter sequence is a method hypothesis. Its use in a later task
        supplies a separate return and can reject or specialize that hypothesis.
        """
        if not trace:
            return None
        current = trace[-1]
        used = current.get("method_id")
        if used:
            method = next(item for item in self.methods if item["method_id"] == used)
            method["trials"] += 1
            if current.get("execution_complete", True):
                method.setdefault("step_outcomes", []).append(self._outcome(
                    current, source_task=source_task, kind="transfer"))
            measured_steps = [step for step in trace if step.get("method_id") == used]
            completed = [step["action"] for step in measured_steps] == method["steps"]
            step_index = len(measured_steps) - 1
            if step_index < len(method.get("step_costs", [])):
                for field in ("compute_cost", "observation_cost"):
                    old = method["step_costs"][step_index][field]
                    measured_cost = current[field]
                    profiles = method.get("step_profiles", [])
                    source_profile = profiles[step_index] if step_index < len(profiles) else None
                    target_profile = current.get("profile")
                    if (field == "compute_cost" and source_profile and target_profile
                            and target_profile.get("observation_count")):
                        measured_cost *= source_profile["observation_count"] / target_profile["observation_count"]
                    method["step_costs"][step_index][field] = .5 * old + .5 * measured_cost
            # A neutral preparatory action does not measure the return of an
            # unfinished sequence. Its dependent next step still needs a trial.
            if completed and all(step.get("execution_complete", True) for step in measured_steps):
                method["observed_gain"] = max(-1., min(1., sum(step["gain"] for step in measured_steps)))
                method.setdefault("sequence_outcomes", []).append({
                    "profile": deepcopy(measured_steps[0].get("profile")),
                    "gain": method["observed_gain"], "source_task": source_task, "kind": "transfer"})
                for field in ("compute_cost", "observation_cost"):
                    method[f"mean_{field}"] = .5 * method[f"mean_{field}"] + .5 * sum(
                        step[field] for step in measured_steps)
            if current["gain"] < -1e-8 and current.get("execution_complete", True):
                if current.get("profile"):
                    method.setdefault("failed_profiles", []).append(deepcopy(current["profile"]))
                else:
                    method["excluded_conditions"].append(deepcopy(condition))
                method["status"] = "revising"
                self.revisions.append({"method_id": used, "change": "restrict_failed_condition",
                                       "condition": deepcopy(condition), "gain": current["gain"]})
            elif current["gain"] > 1e-8 and current.get("execution_complete", True):
                method["confirmations"] += 1
                method["status"] = "supported"
        relevant = [step for step in trace[-6:] if step["context"] == context]
        if not relevant or relevant[-1]["gain"] <= 1e-8 or not relevant[-1].get("execution_complete", True):
            return None
        original = [step["action"] for step in relevant]
        selected = {index for index, step in enumerate(relevant)
                    if step["gain"] > 1e-8 and step.get("execution_complete", True)}
        pending = list(selected)
        while pending:
            index = pending.pop()
            for action in relevant[index].get("requires_actions", []):
                dependency = next((before for before in range(index - 1, -1, -1)
                    if relevant[before]["action"] == action
                    and relevant[before].get("execution_complete", True)), None)
                if dependency is not None and dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        useful_steps = [relevant[index] for index in sorted(selected)]
        # Repeated positive attempts without a dependency are still one step;
        # an explicitly linked preparation is kept in execution order.
        if not any(step.get("requires_actions") for step in useful_steps):
            unique = {}
            for step in useful_steps:
                unique.setdefault(step["action"], step)
            useful_steps = list(unique.values())
        steps = [step["action"] for step in useful_steps]
        profiles = [deepcopy(step.get("profile")) for step in useful_steps]
        prior = next((item for item in self.methods if item["context"] == context
                      and item["condition"] == condition and item["steps"] == steps
                      and all(revision_condition(old) == revision_condition(new)
                              and ((not old and not new) or method_applicability(old, new, action)["status"] == "supported")
                              for old, new, action in zip(item.get("step_profiles", profiles), profiles, steps))), None)
        if prior:
            if prior["method_id"] != used:
                prior.setdefault("step_outcomes", []).append(self._outcome(
                    current, source_task=source_task, kind="source"))
                prior["observed_gain"] = .5 * prior["observed_gain"] + .5 * relevant[-1]["gain"]
            return prior["method_id"]
        identifier = f"method_{len(self.methods)+1}"
        method = {"method_id": identifier, "context": context, "condition": deepcopy(condition),
                  "steps": steps, "original_steps": original, "source_task": source_task,
                  "observed_gain": relevant[-1]["gain"], "trials": 0, "confirmations": 0,
                  "excluded_conditions": [], "status": "proposed",
                  "mean_compute_cost": sum(step["compute_cost"] for step in useful_steps),
                  "mean_observation_cost": sum(step["observation_cost"] for step in useful_steps)}
        method["step_profiles"] = profiles
        method["sequence_outcomes"] = [{"profile": deepcopy(profiles[0]),
            "gain": method["observed_gain"], "source_task": source_task, "kind": "source"}]
        method["step_outcomes"] = [self._outcome(step, source_task=source_task, kind="source")
                                   for step in useful_steps]
        method["step_costs"] = [{key: step[key] for key in ("observation_cost", "compute_cost")}
                                for step in useful_steps]
        self.methods.append(method)
        self.revisions.append({"method_id": identifier, "change": "remove_unproductive_steps" if steps != original
                               else "retain_successful_sequence", "before": original, "after": steps,
                               "source_task": source_task})
        return identifier

    def revise_plan(self, profile, options, trace=()):
        """Offer an affordable structural successor after repeated source failure.

        The caller supplies its executable, already priced options. This is a
        current-evidence revision, not an invented positive return for an
        untried method. Fresh feedback will determine whether it becomes one.
        """
        condition = revision_condition(profile)
        if not condition:
            return None
        eligible = {option.name: option for option in options
                    if method_applicability(None, profile, option.name)["can_try"]}
        structural = ["structure", "structural_discovery"] if profile.get("data", {}).get("object_groups") else []
        structural += ["representation", "state_transition", "continuous_state"]
        successor = next((name for name in structural if name in eligible), None)
        if successor is None:
            return None
        excluded = ["refit"]
        for action in ("task_value", "representative"):
            if action in eligible and not distinguishing_experiment(eligible[action]):
                excluded.append(action)
        return {"action": successor, "excluded_actions": excluded,
                "reason": "repeated_parameter_failure_requires_a_different_representation",
                "applicability": {"status": "supported", "condition": condition},
                "observation_cost": eligible[successor].observation_cost,
                "compute_cost": eligible[successor].compute_cost}

    def choose(self, context, condition, options, trace, *, task_id, observation_remaining, compute_remaining,
               profile=None, ordinary=None, ordinary_evidence=None):
        self.last_choice = None
        eligible = {option.name: option for option in options if option.observation_cost <= observation_remaining
                    and option.compute_cost <= compute_remaining}
        def unfinished(method):
            completed = [step["action"] for step in trace if step.get("method_id") == method["method_id"]]
            return bool(completed and completed == method["steps"][:len(completed)]
                        and len(completed) < len(method["steps"]))

        methods = [item for item in self.methods if item["context"] == context
                   and item["condition"] == condition and item["source_task"] != task_id
                   and condition not in item["excluded_conditions"]
                   and (self._return(item, trace, profile) and self._return(item, trace, profile)["investment_gain"] > 0
                        or unfinished(item))]
        def value(method):
            estimate = self._return(method, trace, profile)
            share = max(estimate["observation_cost"] / max(1, observation_remaining),
                        estimate["investment_compute_cost"] / max(1, compute_remaining), 1e-8)
            return estimate["investment_gain"] / share
        for method in sorted(methods, key=lambda item: (
                not unfinished(item),
                next((index for index, step in enumerate(trace) if step.get("method_id") == item["method_id"]), 0),
                -value(item))):
            applicability = self._applicability(method, trace, profile)
            if applicability is None:
                continue
            completed = [step["action"] for step in trace if step.get("method_id") == method["method_id"]]
            if completed != method["steps"][:len(completed)] or len(completed) >= len(method["steps"]):
                continue
            next_action = method["steps"][len(completed)]
            if next_action not in eligible:
                continue
            measured_return = self._return(method, trace, profile)
            if not completed and ordinary and ordinary.name != next_action:
                if applicability["status"] == "explore" or not (measured_return["transferred"]
                        or measured_return["knowledge_return"]["trials"] > 0 and measured_return["knowledge_return"]["gain"] > 0):
                    # Source success proposes a method. Its first use can be
                    # measured alongside a matching ordinary action; it does
                    # not yet justify replacing a different available plan.
                    continue
                ordinary_probe = ordinary.payload[1] if ordinary.payload else {}
                proposed_evidence = eligible[next_action].payload[1] if eligible[next_action].payload else {}
                if (ordinary_probe.get("unresolved_boundary_width", 0) > 0
                        and ordinary_probe.get("boundary_support")
                        and not proposed_evidence.get("boundary_support")):
                    continue
                if (ordinary_evidence and ordinary_evidence.get("expected_gain", 0) is not None
                        and ordinary_evidence["expected_gain"] > 0):
                    proposed_share = max(measured_return["observation_cost"] / max(1, observation_remaining),
                                         measured_return["investment_compute_cost"] / max(1, compute_remaining), 1e-8)
                    ordinary_share = ordinary_evidence.get("dominant_resource_share", 1.)
                    if measured_return["investment_gain"] / proposed_share <= ordinary_evidence["expected_gain"] / max(ordinary_share, 1e-8):
                        continue
                if (next_action in {"representative", "task_value"}
                        and (ordinary_probe.get("candidate_disagreement", 0) > 0
                             or ordinary_probe.get("expected_risk_reduction", 0) > 0)
                        and not (proposed_evidence.get("candidate_disagreement", 0) > 0
                                 or proposed_evidence.get("expected_risk_reduction", 0) > 0)):
                    # Current rival predictions support a concrete experiment;
                    # source-wide coverage returns cannot displace that probe.
                    continue
            self.last_choice = {"method_id": method["method_id"], "step": len(completed),
                                "steps": list(method["steps"]), "status": method["status"],
                                "applicability": applicability,
                                "return_estimate": measured_return,
                                "changes_action": ordinary is None or ordinary.name != next_action}
            return eligible[next_action]
        return None

    def opportunity(self, context, condition, *, observation_remaining, compute_remaining, task_id=None, trace=(),
                    profile=None, maximum_gain=None, visit_observation_budget=None, visit_compute_budget=None,
                    visit_quote=None):
        """Estimate a concrete next method using its acquired cost and return."""
        estimates, plans = {}, {}
        def estimate(method):
            identifier = method["method_id"]
            if identifier not in estimates:
                estimates[identifier] = self._return(method, trace, profile, continuation=True)
            return estimates[identifier]
        def plan(method):
            identifier = method["method_id"]
            if identifier not in plans:
                plans[identifier] = visit_quote(method, estimate(method))
            return plans[identifier]
        def costs(method):
            if visit_quote is not None:
                proposed = plan(method)
                return proposed["observation_budget"], proposed["compute_budget"]
            value = estimate(method)
            return value["observation_cost"], value["compute_cost"]
        def investment_compute(method):
            return costs(method)[1] + estimate(method)["knowledge_return"]["compute_cost"]
        choices = [method for method in self.methods if method["context"] == context
                   and method["source_task"] != task_id
                   and method["condition"] == condition and condition not in method["excluded_conditions"]
                   and estimate(method) is not None and estimate(method)["investment_gain"] > 0
                   and costs(method)[0] <= observation_remaining
                   and costs(method)[1] <= compute_remaining]
        choices = [method for method in choices if
                   [step["action"] for step in trace if step.get("method_id") == method["method_id"]]
                   == method["steps"][:sum(step.get("method_id") == method["method_id"] for step in trace)]
                   and sum(step.get("method_id") == method["method_id"] for step in trace) < len(method["steps"])]
        choices = [method for method in choices if self._applicability(method, trace, profile) is not None
                   and (not profile or self._completed(method, trace)
                        or self._applicability(method, trace, profile)["status"] == "supported")
                   and (visit_observation_budget is None or costs(method)[0] <= visit_observation_budget)
                   and (visit_compute_budget is None or costs(method)[1] <= visit_compute_budget)]
        if not choices:
            return None
        def gain(method):
            return min(estimate(method)["gain"], maximum_gain) if maximum_gain is not None else estimate(method)["gain"]
        def investment_gain(method):
            return gain(method) + estimate(method)["knowledge_return"]["gain"]
        def value(method):
            share = max(costs(method)[0] / max(1, observation_remaining),
                        investment_compute(method) / max(1, compute_remaining), 1e-8)
            return investment_gain(method) / share
        best = max(choices, key=value)
        if investment_gain(best) <= 0:
            return None
        result = {"method_id": best["method_id"], "steps": best["steps"],
                "value": value(best), "expected_gain": gain(best),
                "expected_knowledge_gain": estimate(best)["knowledge_return"]["gain"],
                "investment_gain": investment_gain(best), "investment_compute_cost": investment_compute(best),
                "source_gain": best["observed_gain"], "return_estimate": estimate(best), "applicability": self._applicability(best, trace, profile),
                "observation_cost": costs(best)[0], "compute_cost": costs(best)[1]}
        if visit_quote is not None:
            result["research_plan"] = deepcopy(plan(best))
        return result

    def to_dict(self):
        return deepcopy({"methods": self.methods, "revisions": self.revisions})
