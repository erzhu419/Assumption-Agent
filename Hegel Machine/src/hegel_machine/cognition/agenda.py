"""A dynamic, shared-budget agenda for the existing research engines."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from .controller import CognitiveAgent
from .delayed_feedback import DelayedFeedback
from .memory import KnowledgeLibrary
from .method_conditions import method_profile
from .models import Probe, ResearchConfig, SearchBudget, TaskSpec
from .research_graph import ResearchGraph
from .workspace import ResearchWorkspace, observation_from_dict


RESOURCES = ("observation_cost", "compute_cost", "measurement_cost", "experiment_cost")


def _spec(payload):
    return TaskSpec(payload["task_id"], payload["description"], tuple(payload["feature_names"]),
        tuple(Probe(item["probe_id"], tuple(item["sequence"])) for item in payload["initial_probes"]),
        tuple(Probe(item["probe_id"], tuple(item["sequence"])) for item in payload["available_probes"]),
        payload.get("risk_quantile", .5), payload.get("context", {}), payload.get("feature_bounds", {}),
        payload.get("max_probe_length", 12), tuple(payload["decision_costs"]) if payload.get("decision_costs") else None)


class ResearchAgenda:
    """Choose actual questions rather than execute a prearranged stage list.

    Each entry names an available apparatus and a bounded engine invocation.
    Instrument research can yield after one real step; object research is one
    finite episode with its own observation, computation and instrument caps.
    The same library is passed to every engine. Public quotes are enforceable
    caps, not fabricated expected gains. Delivered experiments use the existing
    local asynchronous apparatus and retain their separate resource charges.
    """

    def __init__(self, *, directory, observation_budget=512, compute_budget=400000,
                 measurement_budget=4096, experiment_budget=8, library=None, state=None,
                 resources=None):
        self.directory = Path(directory)
        self.resources = resources or {}
        self.state = deepcopy(state) if state else {"kind": "dynamic_research_agenda",
            "limits": dict(zip(RESOURCES, (observation_budget, compute_budget, measurement_budget, experiment_budget))),
            "spent": {name: 0 for name in RESOURCES}, "questions": {}, "events": [],
            "visits": [], "pending_experiments": {}, "status": "ready"}
        self.library = KnowledgeLibrary.from_dict(state["knowledge"]) if state else (library or KnowledgeLibrary())
        self.graph = ResearchGraph.from_dict(state.get("research_graph")) if state else ResearchGraph()
        for identifier, item in self.state["questions"].items():
            if identifier not in self.graph.nodes:
                self.graph.add_question(identifier, item["question"])
        self.feedback = (DelayedFeedback.from_dict(state["feedback"]) if state else DelayedFeedback(
            self.directory / "experiments", observation_budget=observation_budget,
            measurement_budget=measurement_budget, experiment_budget=experiment_budget))

    def _open(self, identifier, source):
        if identifier in self.resources:
            item = self.resources[identifier]
            return item() if callable(item) else item
        if source["kind"] == "suite":
            from .environments import make_suite
            return make_suite(source["seed"], source.get("partition", "development"))[source["task_index"]]
        if source["kind"] == "measurement":
            from .measurement_lab import make_lab
            return make_lab(source["task_index"], source["seed"], backend=source.get("backend", "local"))
        if source["kind"] == "structure":
            from .structure_lab import StructureLab
            return StructureLab(**{key: source[key] for key in ("objects", "readout", "variant", "seed") if key in source})
        if source["kind"] == "mechanism":
            from .mechanism_lab import MechanismLab
            return MechanismLab(**{key: source[key] for key in ("family", "variant", "seed") if key in source})
        raise ValueError("supply a supported apparatus source or a concrete resource")

    def _add(self, identifier, kind, *, source, config, question, measurement_budget=0, max_visits=2,
             depends_on=(), answer_tolerance=None):
        if identifier in self.state["questions"]:
            raise ValueError("question already exists")
        apparatus = self._open(identifier, source)
        try:
            spec = asdict(apparatus.spec)
        finally:
            if hasattr(apparatus, "close") and (identifier not in self.resources or callable(self.resources[identifier])):
                apparatus.close()
        self.graph.add_question(identifier, question or spec["description"], depends_on=depends_on)
        self.state["questions"][identifier] = {"question_id": identifier, "kind": kind,
            "question": question or spec["description"], "source": deepcopy(source), "spec": spec,
            "config": asdict(config), "measurement_budget": measurement_budget, "max_visits": max_visits,
            "status": "ready", "visits": 0, "research_state": None, "result": None,
            "answer_tolerance": config.minimum_improvement if answer_tolerance is None else answer_tolerance,
            "answer_concepts": [], "blocked_on": [],
            "incoming_evidence": False, "costs": {name: 0 for name in RESOURCES}}
        self.state["events"].append({"event": "question_added", "question_id": identifier, "kind": kind})
        self.state["status"] = "ready"
        return identifier

    def add_task(self, identifier, *, source, config=None, question=None, max_visits=2, research_state=None,
                 depends_on=(), answer_tolerance=None):
        self._add(identifier, "task", source=source, config=config or ResearchConfig(),
                  question=question, max_visits=max_visits, depends_on=depends_on, answer_tolerance=answer_tolerance)
        self.state["questions"][identifier]["research_state"] = deepcopy(research_state)
        return identifier

    def add_measurement(self, identifier, *, source, config=None, question=None, measurement_budget=512,
                        depends_on=(), answer_tolerance=None):
        return self._add(identifier, "measurement", source=source, question=question,
            config=config or ResearchConfig(observation_budget=128, compute_budget=100000, max_rounds=3),
            measurement_budget=measurement_budget, depends_on=depends_on, answer_tolerance=answer_tolerance)

    def add_structure(self, identifier, *, source, config=None, question=None, measurement_budget=4096,
                      depends_on=(), answer_tolerance=None):
        return self._add(identifier, "structure", source=source, question=question,
            config=config or ResearchConfig(observation_budget=192, compute_budget=160000, max_rounds=3),
            measurement_budget=measurement_budget, max_visits=1, depends_on=depends_on,
            answer_tolerance=answer_tolerance)

    def add_mechanism(self, identifier, *, source, config=None, question=None, measurement_budget=4096,
                      depends_on=(), answer_tolerance=None):
        return self._add(identifier, "mechanism", source=source, question=question,
            config=config or ResearchConfig(observation_budget=192, compute_budget=160000, max_rounds=3),
            measurement_budget=measurement_budget, max_visits=1, depends_on=depends_on,
            answer_tolerance=answer_tolerance)

    def _reopen_nodes(self, identifier):
        pending, seen = [identifier], set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            item = self.state["questions"][key]
            item.update(answer_concepts=[], incoming_evidence=bool(item["research_state"]))
            if item["kind"] == "measurement" and item["research_state"]:
                paid = (item["result"] or {}).get("costs", {})
                item["measurement_limits"] = {
                    "observation_cost": paid.get("observation_cost", 0) + item["config"]["observation_budget"],
                    "compute_cost": paid.get("compute_cost", 0) + item["config"]["compute_budget"],
                    "measurement_cost": paid.get("measurement_cost", 0) + item["measurement_budget"]}
                if item["research_state"]["stage"] == "completed":
                    item["research_state"]["stage"] = "research"
                item["research_state"].pop("status", None)
            if item["status"] != "waiting_for_feedback":
                item["status"] = "ready"
            item["max_visits"] = max(item["max_visits"], item["visits"] + 1)
            pending.extend(node.question_id for node in self.graph.nodes.values() if key in node.depends_on)
        self.state["status"] = "ready"
        return sorted(seen)

    def set_dependencies(self, identifier, depends_on):
        """Revise a concrete research program and reopen its dependent work."""
        previous = list(self.graph.nodes[identifier].depends_on)
        self.graph.set_dependencies(identifier, depends_on)
        if previous != self.graph.nodes[identifier].depends_on:
            reopened = self._reopen_nodes(identifier)
            self.state["events"].append({"event": "research_dependencies_changed", "question_id": identifier,
                "depends_on": list(self.graph.nodes[identifier].depends_on), "reopened": reopened})

    def _dependency_demand(self):
        live = {concept.concept_id for concept in self.library.concepts if concept.status != "revoked"}
        for identifier, node in self.graph.nodes.items():
            published = self.state["questions"][identifier].get("answer_concepts", [])
            if node.answer is not None and published and not live.intersection(published):
                self.graph._invalidate(identifier)
                self._reopen_nodes(identifier)
        demands = {identifier: set() for identifier in self.graph.nodes}
        for identifier, node in self.graph.nodes.items():
            item = self.state["questions"][identifier]
            item["blocked_on"] = [key for key in node.depends_on
                if key not in self.graph.nodes or self.graph.nodes[key].answer is None]
            if item["status"] not in {"ready", "waiting_for_feedback"}:
                continue
            pending, seen = list(item["blocked_on"]), set()
            while pending:
                key = pending.pop()
                if key in seen or key not in self.graph.nodes:
                    continue
                seen.add(key)
                demands[key].add(identifier)
                pending.extend(dependency for dependency in self.graph.nodes[key].depends_on
                    if dependency not in self.graph.nodes or self.graph.nodes[dependency].answer is None)
        return {key: sorted(value) for key, value in demands.items()}

    def remaining(self):
        reserved = self.feedback.reserved
        return {name: max(0, self.state["limits"][name] - self.state["spent"][name] - reserved.get(name, 0))
                for name in RESOURCES}

    def _offer(self, item):
        state = item["research_state"]
        raw_workspace = state.get("workspace") if state and item["kind"] == "measurement" else state
        workspace = ResearchWorkspace.from_dict(raw_workspace) if raw_workspace else None
        result = item["result"] or {}
        episode = result.get("episode") or (result if item["kind"] == "task" else {})
        profile = method_profile(_spec(item["spec"]), workspace.observations if workspace else [],
            workspace.incumbent if workspace else None, residual_loss=episode.get("validation_loss"),
            baseline_loss=workspace.initial_loss if workspace else None)
        config = item["config"]
        paid = result.get("costs", {}) if item["kind"] == "measurement" else {}
        quote = {"observation_cost": config["observation_budget"] - paid.get("observation_cost", 0),
                 "compute_cost": config["compute_budget"] - paid.get("compute_cost", 0),
                 "measurement_cost": item["measurement_budget"] - paid.get("measurement_cost", 0),
                 "experiment_cost": 0}
        if item["kind"] == "measurement" and item.get("measurement_limits"):
            quote.update({name: limit - paid.get(name, 0) for name, limit in item["measurement_limits"].items()})
        stage = state["stage"] if state and item["kind"] == "measurement" else "research"
        dependencies = self.graph.nodes[item["question_id"]].depends_on
        answers = [{"question_id": key, "program": self.graph.nodes[key].answer.to_dict()}
                   for key in dependencies if key in self.graph.nodes and self.graph.nodes[key].answer is not None]
        return {"kind": f"agenda_{item['kind']}", "action": stage, "question_id": item["question_id"],
                "question": item["question"], "profile": profile, "incoming_evidence": item["incoming_evidence"],
                "knowledge_alternatives": answers,
                "design": ((workspace.decision_frontier or workspace.experiment_frontier).get("design", {})
                           if workspace else {}), "expected_gain": None, **quote}

    def offers(self, *, evaluator_budget=None):
        remaining = self.remaining()
        demands = self._dependency_demand()
        result = []
        for item in self.state["questions"].values():
            if item["status"] != "ready" or item.get("blocked_on"):
                continue
            offer = self._offer(item)
            if any(offer[name] < 0 or offer[name] > remaining[name] for name in RESOURCES):
                continue
            quote = self.library.research_opportunities.estimate(offer,
                evaluator_budget=evaluator_budget if item["research_state"] else None)
            # A new question and an unfinished instrument procedure require
            # an initial trial; neither is assigned a fictitious positive gain.
            unstarted = item["visits"] == 0
            unfinished = item["kind"] == "measurement" and item["research_state"]["stage"] != "completed" if item["research_state"] else False
            exploration = (unstarted and (quote["expected_gain"] or 0.) <= 0.) or (
                quote["expected_gain"] is None and (unfinished or quote["exploration"]))
            if not exploration and (quote["expected_gain"] or 0.) <= 0.:
                continue
            costs = {**offer, "observation_cost": quote["expected_observation_cost"],
                     "compute_cost": quote["expected_compute_cost"]}
            if any(costs[name] > remaining[name] for name in RESOURCES):
                continue
            share = max(costs[name] / max(1, remaining[name]) for name in RESOURCES)
            result.append({**quote, "exploration": exploration, "resource_share": share,
                "demand_count": len(demands[item["question_id"]]), "dependents": demands[item["question_id"]],
                "value": (quote["expected_gain"] or 0.) / max(share, 1e-8)})
        return sorted(result, key=lambda offer: (offer["exploration"], -offer["value"],
                                                -offer["demand_count"], offer["resource_share"]))

    def _evaluator_budget(self, cap, fraction):
        """Spend only surplus above the next complete executable visit."""
        remaining = self.remaining()
        quoted = self.offers()
        if quoted:
            reserve = quoted[0]["expected_compute_cost"]
        else:
            # A learned evaluator may discover value where the old estimate
            # offers none. Its forecast still cannot consume the only budget
            # capable of carrying out one of those concrete questions.
            possible = [self._offer(item) for item in self.state["questions"].values()
                        if item["status"] == "ready" and not item.get("blocked_on")]
            possible = [offer for offer in possible if all(0 <= offer[name] <= remaining[name] for name in RESOURCES)]
            reserve = min((offer["compute_cost"] for offer in possible), default=0)
        allowance = min(cap, remaining["compute_cost"] // fraction,
                        max(0, remaining["compute_cost"] - reserve))
        return SearchBudget(int(allowance))

    def _pay_evaluator(self, budget, phase):
        if budget.spent:
            self.state["evaluator_compute_cost"] = self.state.get("evaluator_compute_cost", 0) + budget.spent
            self.state["spent"]["compute_cost"] += budget.spent
            self.state["events"].append({"event": "research_evaluator", "phase": phase,
                                          "compute_cost": budget.spent})

    def add_evidence(self, identifier, probe, observations, *, measurement_cost=0):
        """Pay for a newly delivered external batch before entering its rows."""
        if self.state["questions"][identifier]["kind"] != "task":
            raise ValueError("external response delivery currently continues ordinary task workspaces")
        costs = {"observation_cost": probe.cost, "measurement_cost": measurement_cost}
        if any(amount > self.remaining()[name] or amount < 0 for name, amount in costs.items()):
            raise ValueError("external evidence exceeds shared agenda resources")
        for name, amount in costs.items():
            self.state["spent"][name] += amount
            self.state["questions"][identifier]["costs"][name] += amount
        self._deliver_evidence(identifier, probe, observations)

    def _deliver_evidence(self, identifier, probe, observations):
        item = self.state["questions"][identifier]
        if item["kind"] != "task":
            raise ValueError("external response delivery currently continues ordinary task workspaces")
        workspace = (ResearchWorkspace.from_dict(item["research_state"]) if item["research_state"]
                     else ResearchWorkspace(item["spec"]["task_id"]))
        workspace.observations.extend(observations)
        workspace.queried.append(probe)
        self.graph._invalidate(identifier)
        self._reopen_nodes(identifier)
        item.update(research_state=workspace.to_dict(), incoming_evidence=True, status="ready",
                    max_visits=max(item["max_visits"], item["visits"] + 1))
        self.state["events"].append({"event": "evidence_delivered", "question_id": identifier,
                                     "probe_id": probe.probe_id, "rows": len(observations)})

    def submit_experiment(self, identifier, probe, *, source=None, channels=(), delay_seconds=0.):
        item = self.state["questions"][identifier]
        if item["kind"] != "task":
            raise ValueError("delayed response experiments currently target ordinary task questions")
        source = source or item["source"]
        from .measurement_lab import CHANNELS
        rates = {channel.name: channel.cost_per_row for channel in CHANNELS}
        required = {"observation_cost": probe.cost, "compute_cost": 0,
                    "measurement_cost": probe.cost * sum(rates[name] for name in channels), "experiment_cost": 1}
        if any(required[name] > self.remaining()[name] for name in RESOURCES):
            raise ValueError("delayed experiment exceeds the shared agenda resources")
        ticket = self.feedback.submit(probe, source=source, channels=channels, delay_seconds=delay_seconds)
        self.state["pending_experiments"][ticket] = {"question_id": identifier, "probe": asdict(probe)}
        item["status"] = "waiting_for_feedback"
        self.state["events"].append({"event": "experiment_submitted", "question_id": identifier, "ticket_id": ticket})
        return ticket

    def poll_experiments(self):
        delivered = []
        for ticket, request in list(self.state["pending_experiments"].items()):
            result = self.feedback.collect(ticket)
            if result is None:
                continue
            item = self.state["questions"][request["question_id"]]
            for name, amount in result["costs"].items():
                self.state["spent"][name] += amount
                item["costs"][name] += amount
            if result["status"] == "succeeded":
                raw = request["probe"]
                self._deliver_evidence(item["question_id"], Probe(raw["probe_id"], tuple(raw["sequence"])),
                                  [observation_from_dict(row) for row in result["observations"]])
            else:
                item["status"] = "ready"
            del self.state["pending_experiments"][ticket]
            delivered.append({"ticket_id": ticket, "status": result["status"], "costs": result["costs"]})
        return delivered

    def _execute(self, item, offer):
        config = ResearchConfig(**item["config"])
        apparatus = self._open(item["question_id"], item["source"])
        previous = item["result"] or {}
        try:
            if item["kind"] == "task":
                episode = CognitiveAgent(config, self.library).run(apparatus, research_state=item["research_state"])
                result = episode.to_dict()
                item["research_state"] = result["research_state"]
                costs = {"observation_cost": episode.observation_cost, "compute_cost": episode.compute_cost}
                unresolved = (episode.validation_loss is None or episode.validation_loss > config.minimum_improvement
                              or bool(result["research_state"].get("decision_frontier"))
                              or bool(result["research_state"].get("experiment_frontier")))
                status = "ready" if unresolved and item["visits"] + 1 < item["max_visits"] else "completed"
            elif item["kind"] == "measurement":
                from .measurement_research import MeasurementResearchAgent
                limits = item.get("measurement_limits", {})
                config = replace(config, observation_budget=limits.get("observation_cost", config.observation_budget),
                                 compute_budget=limits.get("compute_cost", config.compute_budget))
                result = MeasurementResearchAgent(config, measurement_budget=limits.get("measurement_cost", item["measurement_budget"]),
                    library=self.library).run(apparatus, research_state=item["research_state"], max_steps=1)
                costs = {name: result["costs"].get(name, 0) - previous.get("costs", {}).get(name, 0) for name in RESOURCES}
                item["research_state"] = result["research_state"]
                status = "ready" if result["status"] == "paused" else result["status"]
            else:
                if item["kind"] == "mechanism":
                    from .mechanism_discovery import run_mechanism_research as research
                else:
                    from .structural_discovery import run_structural_research as research
                result = research(apparatus, config=config, library=self.library,
                                  measurement_budget=item["measurement_budget"])
                costs, status = result["costs"], result.get("status", "completed")
                item["research_state"] = result["episode"]["research_state"]
        finally:
            if hasattr(apparatus, "close") and (item["question_id"] not in self.resources or callable(self.resources[item["question_id"]])):
                apparatus.close()
        costs = {name: costs.get(name, 0) for name in RESOURCES}
        item.update(result=result, status=status, visits=item["visits"] + 1, incoming_evidence=False)
        for name, cost in costs.items():
            item["costs"][name] += cost
            self.state["spent"][name] += cost
        episode = result if item["kind"] == "task" else result.get("episode") or {}
        new_episode = (item["kind"] != "measurement" or len(result["research_state"]["episodes"])
                       > len((previous.get("research_state") or {}).get("episodes", [])))
        actual = [action["details"] for action in episode.get("actions", [])
                  if new_episode and action["action"] == "continuation_feedback"]
        feedback = {"execution_complete": True, **actual[-1]} if actual else {
            "execution_complete": status == "completed", "reference_rows": 0,
            "gain": None, "normalized_gain": None}
        answer = self._record_answer(item, episode, costs)
        evaluator = self._evaluator_budget(1200, 20)
        self.library.research_opportunities.observe(offer, feedback,
            observation_cost=costs["observation_cost"], compute_cost=costs["compute_cost"],
            evaluator_budget=evaluator)
        self._pay_evaluator(evaluator, "feedback")
        self.state["visits"].append({"question_id": item["question_id"], "kind": item["kind"],
            "costs": costs, "quoted_costs": {name: offer[name] for name in RESOURCES},
            "feedback": deepcopy(feedback), "status": status, "concepts_after": len(self.library.concepts),
            "answer": answer, "evaluator_forecast": deepcopy(offer.get("evaluator_forecast")),
            "evaluator_feedback_compute_cost": evaluator.spent})

    def _record_answer(self, item, episode, costs):
        from .language import Program
        from .memory import _called_definitions

        node = self.graph.nodes[item["question_id"]]
        node.compute_cost += costs["compute_cost"]
        measured = episode.get("validation_loss")
        program = Program.from_dict(episode["program"]) if episode.get("program") else None
        concepts = [concept.concept_id for concept in self.library.concepts
            if program is not None and concept.status != "revoked"
            and concept.program.expanded().simplified() == program.expanded().simplified()]
        workspace = episode.get("research_state") or {}
        accepted = bool(program is not None and episode.get("validation_rows")
            and measured is not None and measured <= item.get("answer_tolerance", item["config"]["minimum_improvement"])
            and not workspace.get("experiment_frontier") and not workspace.get("decision_frontier")
            and concepts and item["status"] == "completed")
        node.answer, node.loss, node.status = (program, measured, "answered") if accepted else (None, measured, "pending")
        item["answer_concepts"] = concepts if accepted else []
        used = {episode.get("selected_memory_id"), *(
            definition.source_concept_id for definition in _called_definitions(program))} if program else set()
        consumed = {key: sorted(used.intersection(self.state["questions"][key].get("answer_concepts", [])))
                    for key in node.depends_on if key in self.state["questions"]}
        record = {"accepted": accepted, "validation_loss": measured, "concepts": item["answer_concepts"],
                  "dependency_consumption": consumed, "selected_operation": episode.get("selected_operation")}
        self.graph.events.append({"event": "apparatus_answer" if accepted else "apparatus_unanswered",
                                  "question_id": node.question_id, **record})
        return record

    def run(self, *, max_steps=None, on_checkpoint=None):
        steps = 0
        while max_steps is None or steps < max_steps:
            self.poll_experiments()
            evaluator = self._evaluator_budget(256, 100)
            offers = self.offers(evaluator_budget=evaluator)
            self._pay_evaluator(evaluator, "forecast")
            remaining = self.remaining()
            offers = [offer for offer in offers
                if offer["expected_observation_cost"] <= remaining["observation_cost"]
                and offer["expected_compute_cost"] <= remaining["compute_cost"]
                and all(offer[name] <= remaining[name] for name in RESOURCES)]
            for offer in offers:
                priced = {**offer, "observation_cost": offer["expected_observation_cost"],
                          "compute_cost": offer["expected_compute_cost"]}
                share = max(priced[name] / max(1, remaining[name]) for name in RESOURCES)
                offer.update(resource_share=share, value=(offer["expected_gain"] or 0.) / max(share, 1e-8))
            offers.sort(key=lambda offer: (offer["exploration"], -offer["value"],
                                          -offer["demand_count"], offer["resource_share"]))
            if not offers:
                break
            chosen = offers[0]
            self.state["events"].append({"event": "research_selected", "question_id": chosen["question_id"],
                "offers": [{key: offer[key] for key in ("question_id", "kind", "expected_gain", "resource_share", "value",
                                                       "demand_count", "dependents")}
                           for offer in offers]})
            self._execute(self.state["questions"][chosen["question_id"]], chosen)
            steps += 1
            if on_checkpoint:
                on_checkpoint(self.to_dict())
        pending = self.state["pending_experiments"]
        ready = [item for item in self.state["questions"].values() if item["status"] == "ready"]
        exhausted = any(item["status"] == "budget_exhausted" for item in self.state["questions"].values())
        self.state["status"] = ("paused" if ready and self.offers() else "waiting_for_feedback" if pending
                                else "blocked_on_dependencies" if any(item.get("blocked_on") for item in ready)
                                else "no_payable_research" if ready or exhausted else "completed")
        return self.to_dict()

    def to_dict(self):
        return {**deepcopy(self.state), "knowledge": self.library.to_dict(), "feedback": self.feedback.to_dict(),
                "remaining": self.remaining(), "research_graph": self.graph.to_dict(),
                "evaluator_compute_cost": self.state.get("evaluator_compute_cost", 0)}


def agenda_summary(report):
    return {key: report[key] for key in ("kind", "status", "limits", "spent", "remaining", "visits", "pending_experiments")} | {
        "questions": [{key: item[key] for key in ("question_id", "kind", "question", "status", "visits", "costs")}
                      | {"blocked_on": item.get("blocked_on", []), "answer_concepts": item.get("answer_concepts", [])}
                      for item in report["questions"].values()], "concepts": len(report["knowledge"]["concepts"]),
        "evaluator_compute_cost": report.get("evaluator_compute_cost", 0)}
