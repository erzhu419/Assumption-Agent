"""Persistent research across tasks, using only public learning interfaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json

from .controller import CognitiveAgent, _representation_work
from .state_search import state_search_work
from .memory import KnowledgeLibrary
from .models import SearchBudget
from .evaluation import loss
from .workspace import ResearchWorkspace
from .action_policy import ResearchActionPolicy
from .experiment_knowledge import research_context
from .method_conditions import method_profile
from .decision_research import decision_quote, run_decision_research
from .continuation_plan import continuation_plan


class ResearchSession:
    """Cover tasks, then revisit unresolved questions within shared resources.

    Entries may be environments or zero-argument environment factories. A
    factory creates the same task afresh for each visit, which lets a simulator
    resume at episode boundaries without saving its private random stream.
    Persistent external environments can instead be supplied directly.
    """

    def __init__(self, config, environments, *, observation_budget, compute_budget,
                 max_visits=2, state=None, feedback=None, library=None):
        if observation_budget < 1 or compute_budget < 1 or max_visits < 1:
            raise ValueError("session resources and maximum visits must be positive")
        self.config = config
        self.sources = list(environments)
        if not self.sources:
            raise ValueError("a research session needs at least one task")
        self.initial_environments = [source() if callable(source) else source
                                     for source in self.sources]
        self.specs = [environment.spec for environment in self.initial_environments]
        self.limits = {"observation_budget": observation_budget,
                       "compute_budget": compute_budget, "max_visits": max_visits}
        task_ids = [spec.task_id for spec in self.specs]
        if state is not None:
            if state["task_ids"] != task_ids:
                raise ValueError("resumed task order differs from the saved research session")
            if state["config"] != asdict(config):
                raise ValueError("resume with the saved research configuration")
            if (observation_budget < state["total_observation_cost"]
                    or compute_budget < state["total_compute_cost"]
                    or max_visits < max(task["visits"] for task in state["tasks"])):
                raise ValueError("session limits cannot be smaller than work already performed")
            self.state = deepcopy(state)
            self.state["limits"] = dict(self.limits)
            self.state.setdefault("workspaces", {})
            self.library = KnowledgeLibrary.from_dict(state["knowledge"])
        else:
            self.library = library if library is not None else KnowledgeLibrary()
            self.state = {
                "experiment": "persistent_cognitive_research_session",
                "config": asdict(config), "limits": dict(self.limits),
                "task_ids": task_ids, "status": "ready", "episodes": [],
                "tasks": [{"task_index": index, "task_id": spec.task_id,
                           "visits": 0, "status": "unvisited", "unresolved": True,
                           "validation_loss": None, "initial_loss": None,
                           "normalized_excess_loss": None, "questions": [],
                           "latest_episode": None} for index, spec in enumerate(self.specs)],
                "total_observation_cost": 0, "total_compute_cost": 0,
                "scheduler_events": [],
                "workspaces": {},
            }
        from .delayed_feedback import DelayedFeedback
        self.feedback = feedback or (DelayedFeedback.from_dict(state["delayed_feedback"])
                                     if state and state.get("delayed_feedback") else None)
        self.state.setdefault("pending_experiments", {})
        self.state.setdefault("total_measurement_cost", 0)
        self.state.setdefault("total_experiment_cost", 0)

    def _continuation_quote(self, workspace, spec, frontier=None):
        """Price initialization and one common comparison from acquired rows."""
        rows = workspace.observations
        length = min(spec.max_probe_length, 4 * self.config.max_history + 4)
        experiment_rows = (frontier or {}).get("observation_cost", 0)
        comparison_rows = (frontier or {}).get("comparison_observation_cost", length)
        compared = len(rows) + experiment_rows + len(workspace.reference) + 2 * comparison_rows
        # Both state and sparse initializers quote executable work on the
        # current data. Scale their row passes for the proposed experiment.
        fitting = max(state_search_work(rows), _representation_work(rows, spec, self.config.max_history))
        fitting = fitting * (len(rows) + experiment_rows) / max(1, len(rows))
        computation = int(fitting + (min(self.config.max_candidates, 8) + 2) * compared
                          + len(rows) + experiment_rows + 1700)
        return experiment_rows + comparison_rows, computation

    def _pending(self, *, evaluator_budget=None):
        waiting = {item["task_index"] for item in self.state["pending_experiments"].values()}
        untouched = [task for task in self.state["tasks"] if task["visits"] == 0 and task["task_index"] not in waiting]
        unresolved = []
        remaining = self._remaining()
        observation_limit = min(self.config.observation_budget, remaining["observation_budget"])
        compute_limit = min(self.config.compute_budget, remaining["compute_budget"])
        for task in self.state["tasks"]:
            task.update(next_research=None, next_method=None, next_experiment=None)
            task.pop("deferred_method", None)
            if (not task["unresolved"] or not task["visits"] or task["visits"] >= self.limits["max_visits"]
                    or task["task_index"] in waiting):
                continue
            saved = self.state["workspaces"].get(str(task["task_index"]))
            if not saved:
                continue
            workspace = ResearchWorkspace.from_dict(saved)
            spec = self.specs[task["task_index"]]
            profile = method_profile(spec, workspace.observations, workspace.incumbent, workspace.method_trace,
                residual_loss=task["validation_loss"], baseline_loss=workspace.initial_loss,
                minimum_improvement=self.config.minimum_improvement)
            offers = []
            frontier = workspace.experiment_frontier
            if frontier and frontier.get("evidence_rows") == len(workspace.observations):
                obs, work = self._continuation_quote(workspace, spec, frontier)
                offers.append({"kind": "experiment", "action": "transition", "profile": profile,
                    "design": frontier["design"], "observation_cost": obs, "compute_cost": work,
                    "question": frontier["question"], "uncertainty": frontier["uncertainty"],
                    "comparison_observation_cost": frontier["comparison_observation_cost"]})
            if workspace.decision_frontier:
                quote = decision_quote(workspace)
                offers.append({"kind": "decision", "action": "compare_decisions", "profile": profile,
                    "design": workspace.decision_frontier["design"], **quote,
                    "question": workspace.decision_frontier["question"]})
            # Ordinary residual work remains a competitor, with a measured
            # recent return when one exists. Error itself is not promised gain.
            recent_gain = None
            if task.get("research_trigger"):
                recent_gain = None if task["research_trigger"] == "new_evidence" else task.get("expected_research_gain")
            elif task.get("continuation_return"):
                feedback = task["continuation_return"]
                recent_gain = (feedback.get("normalized_gain")
                               if feedback.get("execution_complete", True) else None)
            elif spec.decision_costs is not None and task.get("latest_episode") is not None:
                episode = self.state["episodes"][task["latest_episode"]]
                comparisons = [a["details"] for a in episode["actions"] if a["action"] == "compare"]
                if comparisons:
                    latest = comparisons[-1]
                    gain = latest.get("evidence_gain") or 0.
                    required = max(self.config.minimum_improvement, latest.get("dispersion_margin") or 0.)
                    recent_gain = gain / max(task["initial_loss"], self.config.minimum_improvement) if gain >= required else 0.
            if (task.get("normalized_excess_loss") or 0.) > 0. or task.get("research_trigger"):
                plan = continuation_plan(self.config, spec, workspace)
                offers.append({"kind": "revision", "action": "residual_revision", "profile": profile,
                    "observation_cost": plan["observation_budget"], "compute_cost": plan["compute_budget"],
                    "research_plan": plan, "expected_gain": recent_gain,
                    "knowledge_alternatives": task.get("new_knowledge_alternatives", []),
                    "incoming_evidence": task.get("research_trigger") == "new_evidence"})
            if self.config.use_memory and self.config.learn_research_policy:
                policy = ResearchActionPolicy(workspace.observations, spec, {})
                context = policy.set_condition("model_residual")
                def quote_method(method, estimate):
                    completed = sum(step.get("method_id") == method["method_id"] for step in workspace.method_trace)
                    return continuation_plan(self.config, spec, workspace,
                        action=method["steps"][completed], estimate=estimate)
                opportunity = self.library.methods.opportunity(context, research_context(spec, workspace.observations),
                    task_id=spec.task_id, trace=workspace.method_trace,
                    observation_remaining=remaining["observation_budget"], compute_remaining=remaining["compute_budget"],
                    profile=profile, maximum_gain=task["normalized_excess_loss"],
                    visit_observation_budget=observation_limit, visit_compute_budget=compute_limit,
                    visit_quote=quote_method)
                if opportunity:
                    completed = sum(step.get("method_id") == opportunity["method_id"] for step in workspace.method_trace)
                    plan = continuation_plan(self.config, spec, workspace,
                        action=opportunity["steps"][completed], estimate=opportunity["return_estimate"])
                    opportunity = {**opportunity, "research_plan": plan,
                        "observation_cost": plan["observation_budget"], "compute_cost": plan["compute_budget"],
                        "investment_compute_cost": plan["compute_budget"] +
                            opportunity["return_estimate"]["knowledge_return"]["compute_cost"]}
                    # An already frozen decision pair is a concrete prerequisite
                    # for refining that same explanation. Local step returns do
                    # not establish the value of restarting the whole visit.
                    if (workspace.decision_frontier and opportunity["steps"][completed] == "refit"
                            and opportunity["return_estimate"]["basis"] != "measured_continuation_return"):
                        task["deferred_method"] = {"method_id": opportunity["method_id"],
                            "reason": "compare_existing_decisions_before_repeating_parameter_fit"}
                        opportunity = None
                if opportunity:
                    offers.append({**opportunity, "kind": "method", "action": opportunity["steps"][completed],
                        "profile": profile, "method_opportunity": opportunity,
                        "knowledge_compute_cost": max(0., opportunity["investment_compute_cost"] - opportunity["compute_cost"])})
            quoted = [self.library.research_opportunities.estimate(offer,
                evaluator_budget=evaluator_budget if not untouched else None) for offer in offers]
            if evaluator_budget is not None and not untouched:
                task["evaluator_estimates"] = deepcopy(quoted)
            ranked = self.library.research_opportunities.rank_estimates(quoted,
                observation_remaining=remaining["observation_budget"], compute_remaining=remaining["compute_budget"],
                visit_observation_budget=observation_limit, visit_compute_budget=compute_limit)
            task["research_offers"] = ranked
            if not ranked:
                deferred = [offer for offer in quoted if offer["investment_gain"] > 0. or offer["exploration"]]
                if deferred:
                    # Keep valuable but unfunded work pending so increasing a
                    # session budget resumes it without inventing a new task.
                    selected = min(deferred, key=lambda offer: max(
                        offer["expected_observation_cost"] / max(1, observation_limit),
                        offer["expected_compute_cost"] / max(1, compute_limit)))
                    selected["value"] = 0.
                    task.update(next_research=selected, revisit_priority=0., revisit_basis="unfunded_research_opportunity")
                    unresolved.append(task)
                else:
                    basis = ("no_observed_continuation_gain" if task.get("continuation_return")
                             else "no_observed_revision_gain" if spec.decision_costs is not None
                             else "no_observed_research_value")
                    task.update(revisit_priority=0., revisit_basis=basis)
                continue
            selected = ranked[0]
            task["next_research"] = selected
            task["revisit_priority"] = selected["value"]
            task["revisit_basis"] = {"experiment": "unresolved_distinguishing_experiment",
                "method": "learned_method_opportunity", "decision": "unresolved_decision_question", "revision": task.get("research_trigger") or
                ("recent_supported_risk_improvement" if spec.decision_costs is not None else "unresolved_prediction_error")}[selected["kind"]]
            if selected["kind"] == "method":
                task["next_method"] = {**selected["method_opportunity"], "value": selected["value"]}
            elif selected["kind"] == "experiment":
                task["next_experiment"] = {key: selected[key] for key in (
                    "question", "uncertainty", "observation_cost", "comparison_observation_cost", "compute_cost")}
            unresolved.append(task)
        if untouched:
            first_visit_share = max(self.config.observation_budget / max(1, remaining["observation_budget"]),
                                    self.config.compute_budget / max(1, remaining["compute_budget"]))
            first_visit_value = 1. / max(first_visit_share, 1e-8)
            useful = [task for task in unresolved if task.get("next_method")
                      and task["next_method"].get("applicability", {}).get("status") == "supported"
                      and task["next_research"]["value"] > first_visit_value]
            if useful:
                return sorted(useful, key=lambda task: -task["next_research"]["value"]) + untouched
            return untouched
        return sorted(unresolved, key=lambda task: (-task["revisit_priority"], task["visits"], task["task_index"]))

    def _remaining(self):
        result = {name: self.limits[name] - self.state[cost]
                for name, cost in (("observation_budget", "total_observation_cost"),
                                   ("compute_budget", "total_compute_cost"))}
        result["observation_budget"] -= sum(item["observation_reserved"]
                                             for item in self.state["pending_experiments"].values())
        return result

    def _evaluator_budget(self, maximum, divisor):
        remaining = self._remaining()["compute_budget"]
        # Keep a full ordinary visit payable before investing in its evaluator.
        reserve = min(remaining, self.config.compute_budget)
        allowance = min(maximum, remaining // divisor, max(0, remaining - reserve))
        return SearchBudget(allowance if self.config.use_memory and self.config.learn_research_policy else 0)

    def _charge_evaluator(self, budget):
        if budget.spent:
            self.state["total_compute_cost"] += budget.spent
            self.state["evaluator_compute_cost"] = self.state.get("evaluator_compute_cost", 0) + budget.spent

    def submit_experiment(self, task_index, probe, *, source=None, channels=(), delay_seconds=0.):
        """Dispatch an experiment while other research questions remain runnable."""
        if self.feedback is None:
            raise ValueError("attach a DelayedFeedback apparatus to submit experiments")
        if probe.cost > self._remaining()["observation_budget"]:
            raise ValueError("experiment exceeds unreserved session observations")
        task = self.state["tasks"][task_index]
        if source is None:
            from .environments import make_suite
            for partition in self.state.get("partitions", [self.config.partition]):
                match = next((index for index, environment in enumerate(make_suite(self.config.seed, partition=partition))
                              if environment.spec.task_id == task["task_id"]), None)
                if match is not None:
                    source = {"kind": "suite", "task_index": match, "partition": partition, "seed": self.config.seed}
                    break
            if source is None:
                raise ValueError("provide this external task's experiment source")
        ticket = self.feedback.submit(probe, source=source, channels=channels, delay_seconds=delay_seconds)
        self.state["pending_experiments"][ticket] = {"task_index": task_index, "probe": asdict(probe),
            "observation_reserved": probe.cost, "previous_status": task["status"]}
        task["status"] = "waiting_for_feedback"
        self.state["scheduler_events"].append({"reason": "experiment_submitted", "task_index": task_index,
                                                "ticket_id": ticket, "reserved_observations": probe.cost})
        return ticket

    def poll_experiments(self):
        if self.feedback is None:
            return []
        from .models import Probe
        from .workspace import observation_from_dict
        self.feedback.poll()
        received = []
        for ticket, pending in list(self.state["pending_experiments"].items()):
            result = self.feedback.collect(ticket)
            if result is None:
                continue
            self.state["pending_experiments"].pop(ticket)
            task = self.state["tasks"][pending["task_index"]]
            if result["status"] == "succeeded":
                probe = Probe(pending["probe"]["probe_id"], tuple(pending["probe"]["sequence"]))
                self.add_evidence(pending["task_index"], probe,
                                  [observation_from_dict(row) for row in result["observations"]])
            else:
                task["status"] = pending["previous_status"]
            self.state["total_measurement_cost"] += result["costs"]["measurement_cost"]
            self.state["total_experiment_cost"] += result["costs"]["experiment_cost"]
            event = {"reason": "experiment_delivered", "ticket_id": ticket, "task_index": pending["task_index"],
                     "status": result["status"], "costs": result["costs"]}
            self.state["scheduler_events"].append(event)
            received.append(event)
        return received

    def _visit_config(self, task):
        remaining = self._remaining()
        observation_limit = min(self.config.observation_budget, remaining["observation_budget"])
        compute_limit = min(self.config.compute_budget, remaining["compute_budget"])
        spec = self.specs[task["task_index"]]
        workspace = self.state["workspaces"].get(str(task["task_index"]))
        selected = task.get("next_research")
        if selected and (selected["expected_observation_cost"] > observation_limit
                         or selected["expected_compute_cost"] > compute_limit):
            return None
        if selected and selected["kind"] == "decision":
            return replace(self.config, observation_budget=observation_limit, compute_budget=compute_limit,
                seed=self.config.seed + task["task_index"] * 10007 + task["visits"] * 1000003)
        if selected and selected.get("research_plan"):
            plan = selected["research_plan"]
            return replace(self.config, observation_budget=plan["observation_budget"],
                compute_budget=plan["compute_budget"], max_rounds=plan["max_rounds"],
                seed=self.config.seed + task["task_index"] * 10007 + task["visits"] * 1000003)
        saved_rows = len(workspace["observations"]) if workspace else 0
        initial_rows = 0 if saved_rows >= 3 else sum(probe.cost for probe in spec.initial_probes)
        # Fund the public initial queries and one new comparison before starting
        # an episode. The controller handles all later budget choices itself.
        calibration = min(16, observation_limit // 4) if spec.decision_costs is not None and self.config.risk_learning else 0
        if observation_limit - calibration < max(3, initial_rows) + 3:
            return None
        reserve = min(compute_limit // 4, observation_limit * (4 + len(spec.feature_names)))
        if spec.decision_costs is not None and self.config.risk_learning:
            reserve = min(compute_limit // 3, 3000)
        if compute_limit - reserve < 2 * max(3, initial_rows, saved_rows) + 6:
            return None
        return replace(self.config, observation_budget=observation_limit, compute_budget=compute_limit,
                       seed=self.config.seed + task["task_index"] * 10007 + task["visits"] * 1000003)

    def add_environment(self, source):
        """Append a newly available question to the ongoing research session."""
        environment = source() if callable(source) else source
        index = len(self.sources)
        self.sources.append(source)
        self.initial_environments.append(environment)
        self.specs.append(environment.spec)
        self.state["task_ids"].append(environment.spec.task_id)
        self.state["tasks"].append({
            "task_index": index, "task_id": environment.spec.task_id, "visits": 0,
            "status": "unvisited", "unresolved": True, "validation_loss": None,
            "initial_loss": None, "normalized_excess_loss": None, "questions": [],
            "latest_episode": None})
        self.state["status"] = "ready"
        return index

    def add_evidence(self, task_index, probe, observations):
        """Receive a completed external experiment and reopen its question."""
        if probe.cost > self._remaining()["observation_budget"]:
            raise ValueError("new evidence exceeds the remaining observation budget")
        task = self.state["tasks"][task_index]
        saved = self.state["workspaces"].get(str(task_index))
        workspace = ResearchWorkspace.from_dict(saved) if saved else ResearchWorkspace(task["task_id"])
        workspace.observations.extend(observations)
        workspace.queried.append(probe)
        self.state["workspaces"][str(task_index)] = workspace.to_dict()
        self.state["total_observation_cost"] += probe.cost
        task.update(unresolved=True, status="new_evidence", research_trigger="new_evidence",
                    expected_research_gain=None)
        self.state["status"] = "ready"
        self.state["scheduler_events"].append({"reason": "new_evidence", "task_index": task_index,
                                                "observation_cost": probe.cost, "rows": len(observations)})

    def _offer_new_knowledge(self, source_index, concept_ids):
        """Spend a bounded fit on old evidence when a new concept may help."""
        sources = [concept for concept in self.library.concepts if concept.concept_id in concept_ids]
        for task in self.state["tasks"]:
            index = task["task_index"]
            saved = self.state["workspaces"].get(str(index))
            if (index == source_index or not task["unresolved"] or not saved
                    or task["visits"] >= self.limits["max_visits"]):
                continue
            if task.get("normalized_excess_loss") == 0 and saved.get("experiment_frontier"):
                # This operation searches for a reduction of measured error.
                # An unresolved experiment with zero such error needs evidence.
                continue
            spec = self.specs[index]
            relevant = {concept.concept_id for concept in sources if concept.risk_quantile == spec.risk_quantile}
            if not relevant:
                continue
            workspace = ResearchWorkspace.from_dict(saved)
            if workspace.incumbent is None:
                continue
            rows = workspace.observations
            budget = SearchBudget(min(1200, self._remaining()["compute_budget"]))
            if not budget.spend(len(rows)):
                continue
            current = loss(workspace.incumbent.program, rows, workspace.fallback, spec.risk_quantile)
            fitting = SearchBudget(max(0, budget.remaining - 4 * len(rows)))
            candidates = self.library.propose(spec, rows, fitting,
                structural_transfer=self.config.structural_transfer,
                preserve_parameters=self.config.transfer_scheduling, source_concept_ids=relevant)
            budget.spend(fitting.spent)
            useful = []
            gains = []
            for candidate in candidates:
                if not budget.spend(len(rows)):
                    break
                improvement = current - loss(candidate.program, rows, workspace.fallback, spec.risk_quantile)
                if improvement > self.config.minimum_improvement:
                    useful.append(candidate)
                    gains.append(improvement)
            self.state["total_compute_cost"] += budget.spent
            self.state["scheduler_events"].append({"reason": "try_new_knowledge", "task_index": index,
                "source_task_index": source_index, "source_concept_ids": sorted(relevant),
                "compute_cost": budget.spent, "useful_candidates": len(useful)})
            if useful:
                workspace.bank = [*useful, *workspace.bank][:self.config.max_candidates]
                self.state["workspaces"][str(index)] = workspace.to_dict()
                alternatives = []
                for candidate in useful:
                    representation = method_profile(spec, [], candidate)["representation"]
                    structure = {key: representation[key] for key in (
                        "kind", "families", "history_depth", "structure")}
                    if structure not in alternatives:
                        alternatives.append(structure)
                alternatives.sort(key=lambda item: json.dumps(item, sort_keys=True))
                task.update(research_trigger="new_knowledge", expected_research_gain=max(gains) /
                            max(workspace.initial_loss, self.config.minimum_improvement),
                            new_knowledge_alternatives=alternatives, status="new_knowledge")

    def run(self, max_steps=None, on_checkpoint=None):
        """Return a JSON-serializable checkpoint after complete learning visits."""
        if max_steps is not None and max_steps < 0:
            raise ValueError("maximum steps must be nonnegative")
        steps = 0
        self.state["status"] = "running"
        while True:
            self.poll_experiments()
            paused = max_steps is not None and steps >= max_steps
            forecasting = self._evaluator_budget(256, 100)
            pending = self._pending(evaluator_budget=None if paused else forecasting)
            self._charge_evaluator(forecasting)
            if not pending:
                self.state["status"] = "waiting_for_feedback" if self.state["pending_experiments"] else "completed"
                break
            if paused:
                self.state["status"] = "paused"
                break
            affordable = next(((task, visit_config) for task in pending
                               if (visit_config := self._visit_config(task)) is not None), None)
            if affordable is None:
                self.state["status"] = "budget_exhausted"
                break
            task, visit_config = affordable
            index = task["task_index"]
            source = self.sources[index]
            environment = (source() if callable(source) else source) if task["visits"] else self.initial_environments[index]
            reason = ("unresolved_decision_question" if (task.get("next_research") or {}).get("kind") == "decision" else
                      "initial_coverage" if not task["visits"] else task.get("research_trigger") or
                      ("unresolved_distinguishing_experiment" if task.get("next_experiment") else
                      ("learned_method_opportunity" if task.get("next_method") else
                      ("recent_supported_risk_improvement" if task.get("revisit_basis") == "recent_supported_risk_improvement"
                       else "largest_unresolved_relative_loss"))))
            decision = {"step": len(self.state["episodes"]), "task_index": index,
                        "task_id": task["task_id"], "visit": task["visits"] + 1,
                        "reason": reason, "validation_loss": task["validation_loss"],
                        "normalized_excess_loss": task["normalized_excess_loss"],
                        "revisit_priority": task.get("revisit_priority"),
                        "query_seed": visit_config.seed,
                        "allocated_observation_budget": visit_config.observation_budget,
                        "allocated_compute_budget": visit_config.compute_budget}
            if task.get("next_research"):
                decision["research_opportunity"] = deepcopy(task["next_research"])
                saved = self.state["workspaces"][str(index)]
                saved["research_intent"] = {"kind": task["next_research"]["kind"]}
                if task["next_research"].get("research_plan"):
                    saved["research_intent"]["execution_plan"] = deepcopy(task["next_research"]["research_plan"])
            if task.get("next_experiment"):
                decision["experiment_opportunity"] = deepcopy(task["next_experiment"])
            if task.get("next_method"):
                decision["method_opportunity"] = task["next_method"]
                workspace = ResearchWorkspace.from_dict(self.state["workspaces"][str(index)])
                identifier = task["next_method"]["method_id"]
                method = next(item for item in self.library.methods.methods if item["method_id"] == identifier)
                completed = sum(step.get("method_id") == identifier for step in workspace.method_trace)
                workspace.requested_method = {"method_id": identifier, "step": completed,
                    "name": method["steps"][completed], "steps": list(method["steps"]),
                    "context": method["context"], "condition": method["condition"], "status": method["status"],
                    "profile": method_profile(environment.spec, workspace.observations, workspace.incumbent, workspace.method_trace,
                        residual_loss=task["validation_loss"], baseline_loss=workspace.initial_loss,
                        minimum_improvement=self.config.minimum_improvement)}
                self.state["workspaces"][str(index)] = workspace.to_dict()
            if (task.get("next_research") or {}).get("kind") == "decision":
                result = run_decision_research(environment, visit_config, self.library,
                    research_state=self.state["workspaces"][str(index)])
            else:
                result = CognitiveAgent(visit_config, self.library).run(environment,
                           research_state=self.state["workspaces"].get(str(index)))
            payload = result.to_dict()
            self.state["workspaces"][str(index)] = payload.pop("research_state")
            payload.update({"episode_index": len(self.state["episodes"]), "task_index": index,
                            "visit": task["visits"] + 1, "status": "completed", "scheduler": decision})
            self.state["episodes"].append(payload)
            self.state["scheduler_events"].append(decision)
            self.state["total_observation_cost"] += result.observation_cost
            self.state["total_compute_cost"] += result.compute_cost
            measured = result.validation_loss
            excess = (max(0.0, measured - self.config.minimum_improvement)
                      if measured is not None else None)
            experiment_open = bool(self.state["workspaces"][str(index)].get("experiment_frontier")
                                   or self.state["workspaces"][str(index)].get("decision_frontier"))
            unresolved = excess is None or excess > 0 or experiment_open
            task.update({"visits": task["visits"] + 1,
                         "status": ("unresolved_experiment" if experiment_open and excess == 0 else
                                    "unresolved" if unresolved else "adequate_on_learning_evidence"),
                         "unresolved": unresolved, "validation_loss": measured,
                         "initial_loss": result.initial_loss,
                         "normalized_excess_loss": (excess / max(result.initial_loss, self.config.minimum_improvement, 1e-12)
                                                    if excess is not None else None),
                         "questions": payload["questions"] if unresolved else [],
                         "latest_episode": payload["episode_index"]})
            task.pop("research_trigger", None)
            task.pop("expected_research_gain", None)
            task.pop("new_knowledge_alternatives", None)
            returns = [action["details"] for action in payload["actions"] if action["action"] == "continuation_feedback"]
            if returns:
                task["continuation_return"] = {**returns[-1], "observation_cost": result.observation_cost,
                                               "compute_cost": result.compute_cost}
                executed = [action["details"] for action in payload["actions"]
                            if action["action"] == "execute_method_step"]
                if executed and decision.get("method_opportunity"):
                    identifier = decision["method_opportunity"]["method_id"]
                    request = next((step for step in executed if step.get("method_id") == identifier), None)
                    if request:
                        self.library.methods.observe_continuation(identifier, request.get("profile"),
                            {**task["continuation_return"], "method_action": request["name"]},
                            source_task=task["task_id"])
            selected = decision.get("research_opportunity")
            if selected:
                execution_name = {"experiment": "execute_research_question", "decision": "execute_decision_question",
                                  "method": "execute_method_step"}.get(selected["kind"])
                executed = selected["kind"] == "revision" or any(
                    action["action"] == execution_name and (selected["kind"] != "method" or
                        action["details"].get("method_id") == selected["method_id"]) for action in payload["actions"])
                feedback = {**(returns[-1] if returns else {}), "execution_complete": bool(executed and returns
                    and returns[-1].get("execution_complete", True))}
                evaluation_work = self._evaluator_budget(1200, 20)
                outcome = self.library.research_opportunities.observe(selected, feedback,
                    observation_cost=result.observation_cost, compute_cost=result.compute_cost,
                    evaluator_budget=evaluation_work)
                self._charge_evaluator(evaluation_work)
                decision["research_return"] = outcome
                if forecasting.spent or evaluation_work.spent:
                    decision["evaluator_costs"] = {"forecast_compute_cost": forecasting.spent,
                                                   "revision_compute_cost": evaluation_work.spent}
            if self.config.use_memory and result.concepts_added:
                self._offer_new_knowledge(index, result.concepts_added)
            steps += 1
            if on_checkpoint is not None:
                on_checkpoint(self.to_dict())
        return self.to_dict()

    def to_dict(self):
        pending = self._pending()
        report = deepcopy(self.state)
        report["knowledge"] = self.library.to_dict()
        if self.feedback is not None:
            report["delayed_feedback"] = self.feedback.to_dict()
        report["pending"] = [task["task_index"] for task in pending]
        report["remaining"] = self._remaining()
        report["task_count"] = len(self.specs)
        report["visited_tasks"] = sum(task["visits"] > 0 for task in report["tasks"])
        report["adequate_tasks"] = sum(task["visits"] > 0 and not task["unresolved"] for task in report["tasks"])
        report["unresolved_tasks"] = [task["task_index"] for task in report["tasks"] if task["unresolved"]]
        report["completed_visits"] = len(report["episodes"])
        report["prediction_coverage"] = report["visited_tasks"] / report["task_count"]
        report["concepts_learned"] = len(self.library.concepts)
        report["memory_revisions"] = sum(event["action"] in {"restricted", "revoked"}
                                         for row in report["episodes"] for event in row["memory_events"])
        report["interpretation"] = (
            "Completed means the requested coverage and eligible revisits have ended; unresolved losses remain reported. "
            "Scheduling uses learning-side losses, not independent final evaluation. "
            "A nonzero risk loss alone does not trigger a revisit; the latest comparisons must still support a useful revision. "
            "Research workspaces restore acquired observations, alternatives and pending plans; only new queries incur observation cost. "
            "Factories reconnect the same task for each visit; deterministic per-visit query seeds change requested controls.")
        return report
