"""Propose, challenge and revise executable assumptions through public experiments."""

from copy import deepcopy
from dataclasses import asdict, replace
from random import Random

from .evaluation import initial_prediction, loss
from .language import Program
from .memory import KnowledgeLibrary
from .meta_prior_claims import compile_claims
from .models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from .research_graph import ResearchGraph
from .workspace import candidate_from_dict, observation_from_dict


def _serialize(claim):
    result = dict(claim)
    candidate = result.pop("candidate")
    result["candidate"] = {**asdict(candidate), "program": candidate.program.to_dict()}
    for key in ("terms", "basis"):
        if key in result:
            result[key] = [program.to_dict() for program in result[key]]
    if isinstance(result.get("source_program"), Program):
        result["source_program"] = result["source_program"].to_dict()
    return deepcopy(result)


def _deserialize(claim):
    result = deepcopy(claim)
    result["candidate"] = candidate_from_dict(result["candidate"])
    for key in ("terms", "basis"):
        if key in result:
            result[key] = [Program.from_dict(program) for program in result[key]]
    if result.get("source_program"):
        result["source_program"] = Program.from_dict(result["source_program"])
    return result


def _sources(program):
    result = []
    if program.op == "call":
        result.append(program.definition.source_concept_id)
    for child in program.children:
        result.extend(_sources(child))
    return list(dict.fromkeys(result))


def _dimension(claim):
    if "effective_dimension" in claim:
        return claim["effective_dimension"]
    if claim["kind"] == "reuse_compiled":
        return 1
    return len(claim["candidate"].program.required_features())


class MetaPriorResearch:
    """Research world structure without receiving an apparatus's mechanism label.

    A research step fits hypotheses or executes a new experiment. Saved state
    retains rejected evidence, executable child answers and the pending query.
    Environments with external state can supply snapshot()/restore(snapshot);
    other callers retain their environment while resuming the saved research.
    """

    def __init__(self, config=None, library=None):
        self.config = config or ResearchConfig(compute_budget=60000, minimum_improvement=1e-7)
        self.library = library if library is not None else KnowledgeLibrary()

    def _acquire(self, environment, state, probe, purpose):
        rows = list(environment.observe(probe))
        state["costs"]["observation_cost"] += probe.cost
        state["observations"].extend(asdict(row) for row in rows)
        state["queries"].append(asdict(probe))
        state["events"].append({"action": "observe", "purpose": purpose,
            "probe_id": probe.probe_id, "observation_cost": probe.cost, "returned_rows": len(rows)})
        return rows

    def _random_points(self, spec, seed, count):
        rng = Random(seed)
        return [{name: rng.uniform(low, high) for name, (low, high) in spec.feature_bounds.items()}
                for _ in range(count)]

    def _counterexample(self, state, event):
        state["counterexamples"].append(event)
        if self.config.use_memory:
            for prior in self.library.meta_priors:
                if prior["source_concept_id"] in event["source_ids"]:
                    prior["counterexamples"].append({**deepcopy(event), "task_id": state["task_id"],
                                                     "replacement_concept_id": None})
                    prior["status"] = "conditional"

    def _recall(self, spec):
        if not self.config.use_memory:
            return []
        eligible = {concept.concept_id for concept in self.library.concepts if concept.status != "revoked"}
        ids = {record["source_concept_id"] for record in self.library.meta_priors
               if record["source_concept_id"] in eligible}
        calls = self.library.callable_atoms(spec, structural_transfer=False,
                                             limit=len(self.library.definitions))
        # Complete learned responses are reusable coordinates. Subexpressions
        # do not inherit a complete model's validation claim.
        return list(dict.fromkeys(call for call in calls if call.definition.source_concept_id in ids
            and any(concept.concept_id == call.definition.source_concept_id
                    and concept.program.expanded().simplified() == call.expanded().simplified()
                    for concept in self.library.concepts)))[:4]

    def _propose(self, environment, state, budget):
        spec = environment.spec
        rows = [observation_from_dict(row) for row in state["observations"]]
        seeds = self._recall(spec)
        proposed = []
        # First price an actual refit of the learned coordinate. Its new
        # predictions still have to survive a later, unqueried experiment.
        if seeds and not state["counterexamples"]:
            local = SearchBudget(min(budget.remaining, 4 * len(rows) * len(seeds)))
            proposed = compile_claims(spec, rows, local, seed_programs=seeds)
            budget.spend(local.spent)
            reused = [claim for claim in proposed if claim["kind"] == "reuse_compiled"]
            if reused and state["incumbent"] is None:
                prior = min(reused, key=lambda claim: claim["training_loss"])
                source_ids = _sources(prior["candidate"].program)
                origin = next((record for record in reversed(self.library.meta_priors)
                               if record["source_concept_id"] in source_ids), None)
                if origin and origin["counter_prediction"].get("type") == "projection_nullspace":
                    prior["counter_prediction"] = deepcopy(origin["counter_prediction"])
                state["incumbent"] = _serialize(prior)
                if prior["training_loss"] > self.config.minimum_improvement:
                    self._counterexample(state, {"family_id": prior["family_id"], "kind": prior["kind"],
                        "source_ids": source_ids, "loss": prior["training_loss"],
                        "probe_id": state["queries"][-1]["probe_id"],
                        "observations": deepcopy(state["observations"])})
        supported = [claim for claim in proposed if claim["training_loss"] <= self.config.minimum_improvement]
        if not supported:
            local = SearchBudget(budget.remaining // 2)
            proposed = compile_claims(spec, rows, local, seed_programs=seeds)
            budget.spend(local.spent)
        else:
            # A successful reusable computation can still be simplified to a
            # raw affine response. Compare that real alternative, too.
            local = SearchBudget(min(budget.remaining, (len(spec.feature_names) + 2) * len(rows)))
            proposed.extend(compile_claims(spec, rows, local))
            budget.spend(local.spent)
        incumbent = _deserialize(state["incumbent"]) if state["incumbent"] else None
        if incumbent is not None and budget.spend(len(rows)):
            incumbent["training_loss"] = loss(incumbent["candidate"].program, rows, state["fallback"], spec.risk_quantile)
            proposed.append(incumbent)
        raw_supported = [claim for claim in proposed
                         if not _sources(claim["candidate"].program)
                         and claim["training_loss"] <= self.config.minimum_improvement]
        recursive_question = (incumbent is not None and state["counterexamples"]
                              and state["depth"] < self.config.max_question_depth)
        if recursive_question and raw_supported:
            # An available explanation already fits the paid evidence. Its
            # next useful test is a new observation; only a failed comparison
            # creates a reason to buy a dependent residual fit.
            state["events"].append({"action": "defer_recursive_hypothesis",
                "reason": "test_adequate_observed_coordinate_model_first",
                "candidate_kinds": [claim["kind"] for claim in raw_supported]})
        if recursive_question and not raw_supported:
            base = incumbent["candidate"].program
            retained = []
            for seed in seeds:
                if not budget.spend(len(rows)):
                    break
                retained.append((loss(seed, rows, state["fallback"], spec.risk_quantile), seed))
            if retained:
                # Retain a genuinely learned mechanism when asking what was
                # missing. A failed new readout can already have absorbed the
                # missing mechanism into spurious high-order coefficients.
                base = min(retained, key=lambda item: item[0])[1]
            if budget.spend(len(rows)):
                residuals = [Observation(dict(row.features), row.target - base.predict(row),
                    tuple({key: value for key, value in past.items() if key != "__target__"}
                          for past in row.history)) for row in rows]
                local = SearchBudget(budget.remaining // 2)
                children = compile_claims(spec, residuals, local)
                budget.spend(local.spent)
                if children:
                    best_loss = min(claim["training_loss"] for claim in children)
                    child = min((claim for claim in children
                                 if claim["training_loss"] <= best_loss + self.config.minimum_improvement),
                                key=lambda claim: (_dimension(claim), claim["candidate"].program.size))
                    graph = ResearchGraph.from_dict(state["graph"])
                    child_id = f"meta_residual_{state['round']}"
                    node = graph.add_residual_question(child_id, base, rows)
                    node.expression = child["candidate"].program
                    if state.get("last_combined_id"):
                        graph.set_dependencies(child_id, (state["last_combined_id"],))
                    answers = graph.execute_ready(rows, budget, max_nodes=2,
                                                  risk_quantile=spec.risk_quantile)
                    combined_id = f"{child_id}:combined"
                    combined = next((answer for answer in answers
                                     if answer.candidate_id == f"answer:{combined_id}"), None)
                    if combined is not None:
                        proposed.append({"family_id": "T09", "kind": "residual_extension",
                            "candidate": combined, "terms": [base, child["candidate"].program],
                            "claim": "Retain the old mechanism and explain its measured counterexamples with a dependent mechanism",
                            "counter_prediction": deepcopy(child["counter_prediction"]),
                            "training_loss": graph.nodes[combined_id].loss,
                            "child_claim": _serialize(child), "parent_family": incumbent["family_id"],
                            "effective_dimension": _dimension(incumbent) + _dimension(child)})
                        state["last_combined_id"] = combined_id
                        state["depth"] += 1
                    state["graph"] = graph.to_dict()
                    state["events"].append({"action": "recursive_hypothesis", "question_id": child_id,
                        "child_family": child["family_id"], "parent_family": incumbent["family_id"],
                        "answered": combined is not None})
        if not proposed:
            state["status"] = "budget_exhausted"
            return
        # Preserve the strongest candidate from each interpretation, including
        # a genuine dependent answer. Numerical ties favor simpler structure.
        unique = {}
        for claim in proposed:
            key = (claim["kind"], claim["candidate"].program)
            if key not in unique or claim["training_loss"] < unique[key]["training_loss"]:
                unique[key] = claim
        proposed = sorted(unique.values(), key=lambda claim: claim["training_loss"])
        state["candidates"] = [_serialize(claim) for claim in proposed[:self.config.max_candidates]]
        state["events"].append({"action": "propose_hypotheses", "families": [claim["family_id"] for claim in proposed],
                                "kinds": [claim["kind"] for claim in proposed], "rows": len(rows)})
        state["stage"] = "test"

    def _probe(self, spec, state, claims, budget, length):
        pool = self._random_points(spec, self.config.seed + 1009 * (state["round"] + 1), 48)
        # A learned quotient supplies its own concrete falsification: change
        # discarded directions, retaining its supposedly sufficient coordinate.
        contrasts = []
        for claim in claims:
            condition = claim.get("counter_prediction", {})
            names = condition.get("features", [])
            for direction in condition.get("null_directions", [])[:2]:
                center = {name: (low + high) / 2 for name, (low, high) in spec.feature_bounds.items()}
                distance = min(((spec.feature_bounds[name][1] - spec.feature_bounds[name][0]) / (3 * abs(value))
                                for name, value in zip(names, direction) if abs(value) > 1e-8), default=0.)
                contrasts.extend({**center, **{name: center[name] + sign * distance * value
                    for name, value in zip(names, direction)}} for sign in (-1, 0, 1))
        ranked = []
        if all(claim["candidate"].program.minimum_history_depth == 0 for claim in claims):
            for point in pool:
                if not budget.spend(len(claims)):
                    break
                predictions = [claim["candidate"].program.predict(Observation(point, 0.)) for claim in claims]
                ranked.append((max(predictions) - min(predictions), point))
        ranked.sort(key=lambda item: -item[0])
        points = []
        for point in [*contrasts, *(point for _, point in ranked), *pool]:
            if point not in points:
                points.append(point)
            if len(points) >= length:
                break
        return Probe(f"meta_test_{state['round']}", tuple(points[:length]))

    def _test(self, environment, state, budget):
        spec = environment.spec
        claims = [_deserialize(claim) for claim in state["candidates"]]
        counterexamples = [[observation_from_dict(row) for row in event["observations"]]
                           for event in state["counterexamples"]]
        remaining = self.config.observation_budget - state["costs"]["observation_cost"]
        length = min(12, spec.max_probe_length, remaining)
        comparison_rows = length + sum(map(len, counterexamples))
        if length < 3 or budget.remaining < comparison_rows * len(claims):
            state["status"] = "budget_exhausted"
            return
        prediction = SearchBudget(max(0, budget.remaining - comparison_rows * len(claims)))
        probe = self._probe(spec, state, claims, prediction, length)
        budget.spend(prediction.spent)
        rows = self._acquire(environment, state, probe, "distinguish_world_hypotheses")
        prior = _deserialize(state["incumbent"]) if state["incumbent"] else None
        tested = []
        for claim in claims:
            if not budget.spend(comparison_rows):
                break
            claim["validation_loss"] = loss(claim["candidate"].program, rows, state["fallback"], spec.risk_quantile)
            claim["counterexample_loss"] = max((loss(claim["candidate"].program, examples,
                state["fallback"], spec.risk_quantile) for examples in counterexamples), default=0.)
            claim["status"] = ("supported" if max(claim["training_loss"], claim["validation_loss"], claim["counterexample_loss"])
                               <= self.config.minimum_improvement else "refuted")
            tested.append(claim)
        state["claims"].extend(_serialize(claim) for claim in tested)
        if not tested:
            state["status"] = "budget_exhausted"
            return
        minimum = min(max(claim["training_loss"], claim["validation_loss"], claim["counterexample_loss"]) for claim in tested)
        eligible = [claim for claim in tested if max(claim["training_loss"], claim["validation_loss"], claim["counterexample_loss"])
                    <= minimum + self.config.minimum_improvement]
        chosen = min(eligible, key=lambda claim: (_dimension(claim), claim["candidate"].program.size))
        # Record the old structure's actual new counterexample even if a fresh
        # proposal already explains it. The next task keeps that qualification.
        if prior is not None:
            old = next((claim for claim in tested if claim["candidate"].program == prior["candidate"].program), None)
            if old is not None and old["validation_loss"] > self.config.minimum_improvement:
                event = {"family_id": old["family_id"], "kind": old["kind"],
                    "source_ids": _sources(old["candidate"].program), "loss": old["validation_loss"],
                    "probe_id": probe.probe_id, "observations": [asdict(row) for row in rows]}
                self._counterexample(state, event)
        state["incumbent"] = _serialize(chosen)
        state["round"] += 1
        state["events"].append({"action": "test_hypotheses", "probe_id": probe.probe_id,
            "chosen_family": chosen["family_id"], "chosen_kind": chosen["kind"],
            "training_loss": chosen["training_loss"], "validation_loss": chosen["validation_loss"],
            "observations": len(rows)})
        if chosen["status"] == "supported":
            self._retain(environment, state, chosen)
            state["stage"], state["status"] = "completed", "completed"
        elif state["round"] >= self.config.max_rounds:
            state["stage"] = "propose"
            state["status"] = "unresolved"
        else:
            if not state["counterexamples"]:
                self._counterexample(state, {"family_id": chosen["family_id"], "kind": chosen["kind"],
                    "source_ids": _sources(chosen["candidate"].program), "loss": chosen["validation_loss"],
                    "probe_id": probe.probe_id, "observations": [asdict(row) for row in rows]})
            state["stage"] = "propose"

    def _retain(self, environment, state, chosen):
        if not self.config.use_memory:
            return
        rows = [observation_from_dict(row) for row in state["observations"]]
        program = chosen["candidate"].program
        parents = _sources(program)
        candidate = replace(chosen["candidate"], parents=tuple(parents))
        concept_id, created = self.library.learn(candidate, environment.spec, rows)
        record = {"prior_id": f"meta_prior_{len(self.library.meta_priors) + 1:04d}",
            "family_id": chosen["family_id"], "kind": chosen["kind"], "claim": chosen["claim"],
            "counter_prediction": deepcopy(chosen["counter_prediction"]),
            "source_concept_id": concept_id, "parent_concept_ids": parents,
            "task_id": environment.spec.task_id, "source_features": list(environment.spec.feature_names),
            "status": "supported", "support_rows": len(rows),
            "validation_loss": chosen["validation_loss"], "counterexamples": [], "uses": 0,
            "probe_ids": [query["probe_id"] for query in state["queries"]]}
        for previous in self.library.meta_priors:
            if previous["source_concept_id"] in parents:
                previous["uses"] += 1
            for event in previous["counterexamples"]:
                if event["task_id"] == state["task_id"] and event["replacement_concept_id"] is None:
                    event["replacement_concept_id"] = concept_id
        self.library.meta_priors.append(record)
        state["prior_updates"].append({"prior_id": record["prior_id"], "source_concept_id": concept_id,
            "created_concept": created, "used_sources": parents, "family_id": chosen["family_id"]})

    def _report(self, environment, state):
        if hasattr(environment, "snapshot"):
            state["environment_state"] = environment.snapshot()
        selected = _deserialize(state["incumbent"]) if state["incumbent"] else None
        return {"status": state["status"], "task_id": state["task_id"],
            "program": selected["candidate"].program.to_dict() if selected else None,
            "fallback": state["fallback"], "costs": deepcopy(state["costs"]),
            "family_id": selected["family_id"] if selected else None,
            "kind": selected["kind"] if selected else None,
            "claims": deepcopy(state["claims"]), "events": deepcopy(state["events"]),
            "prior_updates": deepcopy(state["prior_updates"]), "research_state": deepcopy(state)}

    def run(self, environment, *, research_state=None, max_steps=None, on_checkpoint=None):
        if max_steps is not None and max_steps < 1:
            raise ValueError("research steps must be positive")
        state = deepcopy(research_state) if research_state else {
            "task_id": environment.spec.task_id, "stage": "initial", "status": "running",
            "costs": {"observation_cost": 0, "compute_cost": 0}, "observations": [], "queries": [],
            "incumbent": None, "fallback": 0., "candidates": [], "claims": [], "events": [],
            "counterexamples": [], "graph": {}, "depth": 0, "round": 0, "prior_updates": []}
        if state["task_id"] != environment.spec.task_id:
            raise ValueError("saved research belongs to another task")
        if research_state and state.get("environment_state") is not None:
            environment.restore(state["environment_state"])
        if state["status"] == "completed":
            return self._report(environment, state)
        state["status"] = "running"
        budget = SearchBudget(self.config.compute_budget, state["costs"]["compute_cost"])
        steps = 0
        while state["status"] == "running" and (max_steps is None or steps < max_steps):
            if state["stage"] == "initial":
                length = min(16, environment.spec.max_probe_length, self.config.observation_budget // 2)
                if length < 3:
                    state["status"] = "budget_exhausted"
                    break
                initial = list(environment.spec.initial_probes)
                probe = (initial[0] if initial and 3 <= initial[0].cost <= length else
                    Probe("meta_initial", tuple(self._random_points(environment.spec, self.config.seed, length))))
                rows = self._acquire(environment, state, probe, "initial_structure_evidence")
                state["fallback"] = initial_prediction(rows, environment.spec.risk_quantile)
                state["stage"] = "propose"
            elif state["stage"] == "propose":
                self._propose(environment, state, budget)
            else:
                self._test(environment, state, budget)
            state["costs"]["compute_cost"] = budget.spent
            steps += 1
            if on_checkpoint:
                on_checkpoint(self._report(environment, state))
        if state["status"] == "running":
            state["status"] = "paused"
        return self._report(environment, state)
