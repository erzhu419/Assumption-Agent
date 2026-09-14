"""The common researcher with experiments fixed before observing any outcomes."""

from dataclasses import asdict
from random import Random

from .evaluation import initial_prediction, loss
from .language import Program
from .meta_prior_research import _sources
from .models import Candidate, Probe, SearchBudget
from .research_graph import ResearchGraph
from .unified_research import UnifiedResearchAgent


class FixedExperimentResearchAgent(UnifiedResearchAgent):
    """Learn representations and reuse knowledge under a fixed paid design.

    All instruments and all four control trajectories are committed before the
    first response is observed. A learned hypothesis cannot shorten the design,
    change an instrument, or request another trajectory after confirmation.
    """

    def run(self, laboratory):
        channels = tuple(channel.name for channel in laboratory.channels)
        rate = sum(channel.cost_per_row for channel in laboratory.channels)
        length = min(24, laboratory.spec.max_probe_length,
                     self.config.observation_budget // 4)
        if rate:
            length = min(length, self.measurement_budget // (4 * rate))
        if length < 1:
            raise ValueError("fixed experimental design needs four funded observations")
        rng = Random(self.config.seed)
        probes = [Probe(f"fixed_experiment_{index}", tuple(
            {name: rng.uniform(*bounds) for name, bounds in laboratory.spec.feature_bounds.items()}
            for _ in range(length))) for index in range(4)]
        actions = [{"action": "commit_fixed_experiments", "channels": list(channels),
                    "probes": [asdict(probe) for probe in probes]}]
        budget, graph, transfer, batches = SearchBudget(self.config.compute_budget), ResearchGraph(), [], []
        observation_cost, measurement_cost = 0, 0

        def acquire(index, purpose):
            nonlocal observation_cost, measurement_cost
            batch = laboratory.acquire(probes[index], channels)
            observation_cost += batch.observation_cost
            measurement_cost += batch.measurement_cost
            batches.append({"probe": asdict(probes[index]), "channels": list(channels),
                            "observations": [asdict(row) for row in batch.observations]})
            actions.append({"action": "acquire", "purpose": purpose,
                "probe_id": probes[index].probe_id, "channels": list(channels),
                "observation_cost": batch.observation_cost, "measurement_cost": batch.measurement_cost})
            return batch.observations

        training = [*acquire(0, "fit"), *acquire(1, "fit")]
        validation = acquire(2, "compare_on_fixed_trajectory")
        spec = laboratory.spec_for(channels)
        fallback = initial_prediction(training, spec.risk_quantile)
        # Reserve the final comparison while using exactly the same fitting and
        # reusable computations as the adaptive experimental-design arm.
        fitting = SearchBudget(max(0, budget.remaining - length))
        claims = self._research_view(spec, training, validation, fitting, actions, transfer, graph)
        budget.spend(fitting.spent)
        selected = self._best(claims)
        if selected is None:
            selected = {"candidate": Candidate("fixed_design:initial",
                Program("const", value=fallback), "initial"), "kind": "initial",
                "family_id": "T02", "validation_loss": None}
        program = selected["candidate"].program
        confirmation = acquire(3, "confirm_frozen_program")
        confirmation_loss = None
        if budget.spend(len(confirmation)):
            confirmation_loss = loss(program, confirmation, fallback, spec.risk_quantile)
        confirmed = confirmation_loss is not None and confirmation_loss <= self.config.minimum_improvement
        actions.append({"action": "validate_frozen_program", "program": program.to_dict(),
                        "loss": confirmation_loss, "accepted": confirmed,
                        "observation_rows": len(confirmation)})
        sources = _sources(program)
        used_sources = sources if confirmed else []
        for event in transfer:
            event["accepted"] = bool(confirmed and set(event["source_ids"]) & set(used_sources)
                                     and event["status"] == "supported")
            if not confirmed and set(event["source_ids"]) & set(sources):
                event.update(status="counterexample", confirmation_loss=confirmation_loss)
        concepts_added = []
        if self.config.use_memory:
            for event in transfer:
                if event["status"] != "counterexample":
                    continue
                for prior in self.library.meta_priors:
                    if prior["source_concept_id"] in event["source_ids"]:
                        prior["counterexamples"].append({"task_id": spec.task_id,
                            "loss": event.get("confirmation_loss", event["validation_loss"]),
                            "kind": "failed_role_binding", "program": event["program"],
                            "replacement_concept_id": None})
                        prior["status"] = "conditional"
            if confirmed:
                concept_id, added = self.library.learn(selected["candidate"], spec,
                                                       [*training, *validation])
                concepts_added = [concept_id] if added else []
                for source_id in used_sources:
                    self.library.record_use(source_id, spec.task_id)
                for prior in self.library.meta_priors:
                    if prior["source_concept_id"] in used_sources:
                        prior["uses"] += 1
                    for event in prior["counterexamples"]:
                        if event["task_id"] == spec.task_id and event["replacement_concept_id"] is None:
                            event["replacement_concept_id"] = concept_id
                if not any(prior["source_concept_id"] == concept_id for prior in self.library.meta_priors):
                    self.library.meta_priors.append({
                        "prior_id": f"meta_prior_{len(self.library.meta_priors) + 1:04d}",
                        "source_concept_id": concept_id, "parent_concept_ids": used_sources,
                        "family_id": selected["family_id"], "kind": selected["kind"],
                        "claim": "The selected representation predicts a new precommitted trajectory",
                        "counter_prediction": {"type": "future_trajectory",
                            "condition": "The rebound computation predicts another measured trajectory"},
                        "status": "supported", "task_id": spec.task_id,
                        "source_features": list(spec.feature_names),
                        "control_features": list(spec.feature_bounds),
                        "support_rows": len(training) + len(validation),
                        "validation_loss": confirmation_loss,
                        "probe_ids": [probe.probe_id for probe in probes],
                        "counterexamples": [], "uses": 0})
        return {"task_id": spec.task_id, "status": "completed" if confirmed else "unresolved",
                "program": program.to_dict(), "expression": program.describe(), "fallback": fallback,
                "selected_kind": selected["kind"], "selected_channels": list(channels),
                "validation_loss": selected["validation_loss"], "confirmation_loss": confirmation_loss,
                "costs": {"observation_cost": observation_cost, "measurement_cost": measurement_cost,
                          "compute_cost": budget.spent, "queries": len(probes)},
                "actions": actions, "transfer": transfer, "used_sources": used_sources,
                "method_source_ids": [], "concepts_added": concepts_added,
                "research_graph": graph.to_dict(), "batches": batches}
