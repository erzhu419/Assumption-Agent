"""Run the observation → question → experiment → revision → memory loop."""

from __future__ import annotations

from dataclasses import asdict
from random import Random
from time import perf_counter
from typing import Protocol

from .diagnosis import diagnose, research_steps
from .evaluation import initial_prediction, loss, predict
from .language import Program
from .memory import KnowledgeLibrary
from .models import (ActionRecord, Candidate, EpisodeResult, Observation, Probe,
                             ResearchConfig, SearchBudget, TaskSpec)
from .probing import choose_probe, construct_probes, observed_classes
from .planning import execute_revision_plan, revision_plans
from .synthesis import synthesize
from .uncertainty import fit_decision_model


class LearningEnvironment(Protocol):
    spec: TaskSpec

    def observe(self, probe: Probe) -> list[Observation]: ...


ALL_OPERATIONS = ("refit", "representation", "relation", "composition",
                  "robustification", "idealization")


def _split(rows: list[Observation], fraction: float):
    stride = max(3, round(1 / fraction))
    train = [row for i, row in enumerate(rows) if i % stride != stride - 1]
    validation = [row for i, row in enumerate(rows) if i % stride == stride - 1]
    if not validation:
        return rows[:-1], rows[-1:]
    return train, validation


class CognitiveAgent:
    def __init__(self, config: ResearchConfig | None = None,
                 library: KnowledgeLibrary | None = None, proposer=None):
        self.config = config or ResearchConfig()
        self.library = library if library is not None else KnowledgeLibrary()
        self.proposer = proposer

    def run(self, environment: LearningEnvironment) -> EpisodeResult:
        config, spec = self.config, environment.spec
        started_at = perf_counter()
        # Leave room to evaluate the final model and its decision uncertainty.
        # Search must not spend the resources needed to use its own result.
        final_reserve = min(config.compute_budget // 4,
                            config.observation_budget * (4 + len(spec.feature_names)))
        compute = SearchBudget(config.compute_budget - final_reserve)
        observations: list[Observation] = []
        actions, questions = [], []
        observation_cost, used_probes = 0, set()
        rng = Random(config.seed)
        plan_records, llm_probes, llm_records = [], [], []
        proposer = self.proposer
        if proposer is None and config.proposer in {"codex", "hybrid"}:
            from .proposers import CodexProposer
            proposer = CodexProposer(timeout_seconds=config.llm_timeout_seconds,
                                     token_budget=config.llm_token_budget, max_calls=config.llm_calls_per_task,
                                     max_history=config.max_history)

        def acquire(probe, round_number, reason):
            nonlocal observation_cost
            if observation_cost + probe.cost > config.observation_budget:
                return []
            rows = environment.observe(probe)
            observation_cost += probe.cost
            used_probes.add(probe.probe_id)
            actions.append(ActionRecord(round_number, "observe", reason,
                                        observation_cost=probe.cost,
                                        details={"probe_id": probe.probe_id, "rows": len(rows)}))
            return rows

        for probe in spec.initial_probes:
            observations.extend(acquire(probe, 0, "Initial evidence under the common task interface"))
        if len(observations) < 3:
            raise ValueError("observation budget must fund at least three initial observations")
        train, validation = _split(observations, config.validation_fraction)
        fallback = initial_prediction(train, spec.risk_quantile)
        incumbent = Candidate("initial", Program("const", value=fallback), "refit", "Initial fitted prediction")
        if not compute.spend(len(validation)):
            raise ValueError("compute budget cannot fund the initial candidate comparison")
        initial_loss = loss(incumbent.program, validation, fallback, spec.risk_quantile)
        current_loss = initial_loss
        retrieved = self.library.retrieve(spec) if config.use_memory else []
        candidates = [incumbent, *retrieved]
        previous = None
        stop_reason = "round_budget"

        def available_experiments(round_number):
            generated = construct_probes(spec, candidates, round_number=round_number,
                                         seed=config.seed, max_history=config.max_history) if config.synthesize_probes else []
            pool = [*spec.available_probes, *generated, *llm_probes]
            return [p for p in pool if p.probe_id not in used_probes
                    and p.cost <= config.observation_budget - observation_cost]

        if config.use_memory and config.use_abstractions:
            local_budget = SearchBudget(min(compute.remaining // 4, 3000))
            adapted = self.library.propose(spec, train, local_budget, structural_transfer=config.structural_transfer)
            compute.spend(local_budget.spent)
            retrieved.extend(adapted)
            candidates.extend(adapted)
            actions.append(ActionRecord(0, "adapt_knowledge", "Refit learned abstractions and map their observable roles",
                                        compute_cost=local_budget.spent,
                                        details={"candidates": len(adapted), "source_concepts": [c.memory_id for c in adapted]}))

        for round_number in range(config.max_rounds):
            confirmed_reuse = False
            train, validation = _split(observations, config.validation_fraction)
            if not compute.spend(len(validation)):
                stop_reason = "compute_budget"
                break
            current_loss = loss(incumbent.program, validation, fallback, spec.risk_quantile)

            # Retrieve and actually evaluate previous knowledge before paying
            # for a new search. This creates a measurable opportunity to reuse.
            if round_number == 0 and retrieved:
                tested = []
                evaluated_retrieval = []
                for candidate in retrieved:
                    if not compute.spend(len(validation)):
                        break
                    measured = loss(candidate.program, validation, fallback, spec.risk_quantile)
                    tested.append({"memory_id": candidate.memory_id, "operation": candidate.operation,
                                   "loss": measured})
                    evaluated_retrieval.append((measured, candidate))
                    if measured + config.complexity_weight * candidate.program.size < (
                            current_loss + config.complexity_weight * incumbent.program.size):
                        incumbent, current_loss = candidate, measured
                actions.append(ActionRecord(round_number, "retrieve", "Test executable concepts on current evidence",
                                            details={"retrieved": [c.memory_id for c in retrieved],
                                                     "tested": tested,
                                                     "selected": incumbent.memory_id}))
                candidates = [incumbent, *[candidate for measured, candidate in evaluated_retrieval
                               if candidate is not incumbent
                               and measured <= current_loss + max(config.minimum_improvement, current_loss * 0.25)]]

            if not compute.spend(len(observations)):
                stop_reason = "compute_budget"
                break
            question = diagnose(incumbent.program, observations, fallback,
                                spec.risk_quantile, max_depth=config.max_question_depth)
            questions.append(question)
            actions.append(ActionRecord(round_number, "diagnose", question.question,
                                        compute_cost=len(observations), details=asdict(question)))

            if (incumbent.memory_id and current_loss <= config.minimum_improvement
                    and question.evidence["squared_residual"] <= config.minimum_improvement):
                # Verify cheap reuse with a fresh experiment before stopping.
                affordable = available_experiments(round_number)
                selected_probe = None
                if affordable:
                    if config.active_probes:
                        selected_probe, _ = choose_probe(affordable, candidates[:config.max_candidates], fallback, compute)
                    else:
                        selected_probe = rng.choice(affordable)
                if selected_probe and compute.spend(selected_probe.cost):
                    fresh = acquire(selected_probe, round_number, "Check reused knowledge on a newly selected discriminating experiment")
                    fresh_loss = loss(incumbent.program, fresh, fallback, spec.risk_quantile)
                    observations.extend(fresh)
                    actions.append(ActionRecord(round_number, "reuse_check", "Measured fresh evidence for a retrieved program",
                                                compute_cost=len(fresh), details={"loss": fresh_loss,
                                                "memory_id": incumbent.memory_id, "expression": incumbent.program.describe(),
                                                "accepted": fresh_loss <= config.minimum_improvement}))
                    if fresh_loss <= config.minimum_improvement:
                        current_loss = fresh_loss
                        if incumbent.program.size <= 3 or config.strategy != "adaptive":
                            stop_reason = "knowledge_reused"
                            break
                        confirmed_reuse = True
                        reused_program = incumbent.program
                    train, validation = _split(observations, config.validation_fraction)
                    if not confirmed_reuse:
                        if not compute.spend(len(observations)):
                            stop_reason = "compute_budget"
                            break
                        question = diagnose(incumbent.program, observations, fallback, spec.risk_quantile,
                                            max_depth=config.max_question_depth)
                        questions.append(question)
                        actions.append(ActionRecord(round_number, "diagnose", "Reconsider the explanation after failed transfer",
                                                    compute_cost=len(observations), details=asdict(question)))

            if config.strategy == "fixed":
                steps = [("refit", "Fit a fixed current-observation model family")]
            elif config.strategy == "search":
                steps = [(op, "Search the shared primitive language") for op in ALL_OPERATIONS]
            else:
                steps = list(research_steps(question, max_depth=config.max_question_depth))
                if confirmed_reuse:
                    steps = [("idealization", "Simplify successfully reused knowledge before finishing")]
                if not steps and (not config.active_probes or len(candidates) < 2):
                    stop_reason = "adequate_and_compact"
                    break

            # All proposed operation families get bounded work this round;
            # no diagnostic must first prove every other family impossible.
            round_allowance = min(compute.remaining // 2,
                                  max(1200, config.compute_budget // config.max_rounds))
            quota = max(1, round_allowance // max(1, len(steps)))
            proposed = []
            if proposer is not None:
                batch = proposer.propose(train, incumbent.program, spec, compute)
                proposed.extend(batch.candidates)
                llm_probes.extend(batch.probes)
                llm_records.append(batch.usage)
                actions.append(ActionRecord(round_number, "llm_proposal", "Execute model-proposed programs and experiments under the shared interface",
                                            details={"candidates": len(batch.candidates), "probes": len(batch.probes),
                                                     "usage": batch.usage}))
            if config.joint_planning and config.strategy == "adaptive" and config.proposer != "codex":
                plans = revision_plans(question, joint=True, depth=config.planning_depth)
                if confirmed_reuse:
                    from .models import ResearchQuestion
                    plans = revision_plans(ResearchQuestion("Simplify successfully reused knowledge", {}, ("idealization",)))
                for plan in plans:
                    local_budget = SearchBudget(min(compute.remaining // 2, round_allowance // max(1, len(plans))))
                    generated = execute_revision_plan(plan, train, incumbent=incumbent.program,
                                                      budget=local_budget, risk_quantile=spec.risk_quantile,
                                                      max_history=config.max_history)
                    compute.spend(local_budget.spent)
                    proposed.extend(generated)
                    plan_records.append({"round": round_number, **plan.to_dict()})
                    for stage in plan.steps:
                        actions.append(ActionRecord(round_number, stage["operation"], plan.question,
                                                    compute_cost=stage["compute_cost"], details=stage))
                steps = []
            if config.proposer == "codex":
                steps = []
            for operation, reason in steps:
                rows = train
                if operation in {"representation", "composition"}:
                    history_rows = [row for row in train if row.history]
                    if len(history_rows) >= 3:
                        rows = history_rows
                local_budget = SearchBudget(min(quota, compute.remaining))
                generated = synthesize(rows, operations=(operation,), budget=local_budget,
                                       incumbent=incumbent.program, risk_quantile=spec.risk_quantile,
                                       max_history=config.max_history)
                compute.spend(local_budget.spent)
                proposed.extend(generated[:32])
                actions.append(ActionRecord(round_number, operation, reason,
                                            compute_cost=local_budget.spent,
                                            details={"candidates": len(generated),
                                                     "parent_expression": incumbent.program.describe()}))

            unique = []
            encodings = []
            for candidate in [incumbent, *candidates, *proposed]:
                encoded = candidate.program.to_dict()
                if encoded not in encodings:
                    encodings.append(encoded)
                    unique.append(candidate)
            ranked = []
            scoring_cost = 0
            for candidate in unique:
                if not compute.spend(len(validation)):
                    break
                measured = loss(candidate.program, validation, fallback, spec.risk_quantile)
                scoring_cost += len(validation)
                score = measured + config.complexity_weight * candidate.program.size
                ranked.append((score, measured, candidate))
            if not ranked:
                stop_reason = "compute_budget"
                break
            ranked.sort(key=lambda row: (row[0], row[2].program.size))
            best_score, best_loss, best = ranked[0]
            # Spend new observations on explanations still compatible with the
            # evidence, instead of repeatedly distinguishing a refuted program.
            candidates = [row[2] for row in ranked
                          if row[1] <= best_loss + max(config.minimum_improvement, best_loss * 0.25)][:config.max_candidates]
            current_entry = next((r for r in ranked if r[2] is incumbent), None)
            current_score = current_entry[0] if current_entry else float("inf")
            if current_entry:
                current_loss = current_entry[1]
            actions.append(ActionRecord(round_number, "compare", "Compare executed candidates under the fixed task loss",
                                        compute_cost=scoring_cost,
                                        details={"candidates_scored": len(ranked), "best_loss": best_loss,
                                                 "best_expression": best.program.describe(),
                                                 "validation_is_adaptive_development": True}))
            if best_score + config.minimum_improvement < current_score:
                previous = incumbent
                incumbent, current_loss = best, best_loss
                actions.append(ActionRecord(round_number, "revise", "Adopt the measured executable revision in working memory",
                                            details={"operation": best.operation,
                                                     "before": previous.program.describe(),
                                                     "after": best.program.describe(),
                                                     "before_program": previous.program.to_dict(),
                                                     "after_program": best.program.to_dict(),
                                                     "development_loss": best_loss}))

            if confirmed_reuse:
                stop_reason = "knowledge_reused_and_simplified" if incumbent.program != reused_program else "knowledge_reused"
                break

            if round_number + 1 == config.max_rounds:
                break
            affordable = available_experiments(round_number)
            if not affordable:
                stop_reason = "observation_budget"
                break
            before_probe_compute = compute.spent
            old_groups = observed_classes(candidates, observations, fallback, compute)
            old_groups_complete = sum(map(len, old_groups)) == len(candidates)
            if config.active_probes and config.strategy != "fixed":
                probe, disagreement = choose_probe(affordable, candidates, fallback, compute)
            else:
                probe, disagreement = rng.choice(affordable), None
            actions.append(ActionRecord(round_number, "plan_experiment", "Choose an affordable discriminating trajectory",
                                        compute_cost=compute.spent - before_probe_compute,
                                        details={"disagreement": disagreement,
                                                 "policy": "disagreement" if config.active_probes else "random"}))
            if probe is None:
                stop_reason = "compute_budget"
                break
            fresh = acquire(probe, round_number, "Resolve competing explanations with new measurements")
            if fresh and previous and compute.spend(2 * len(fresh)):
                revised_loss = loss(incumbent.program, fresh, fallback, spec.risk_quantile)
                parent_loss = loss(previous.program, fresh, fallback, spec.risk_quantile)
                if revised_loss > parent_loss + max(0.05, parent_loss * 0.25):
                    actions.append(ActionRecord(round_number, "rollback", "New observations favor the previous working hypothesis",
                                                compute_cost=2 * len(fresh),
                                                details={"revised_loss": revised_loss, "parent_loss": parent_loss,
                                                         "rejected_program": incumbent.program.to_dict(),
                                                         "restored_program": previous.program.to_dict(),
                                                         "rejected_expression": incumbent.program.describe()}))
                    incumbent, previous = previous, None
                else:
                    actions.append(ActionRecord(round_number, "prediction_check", "Score both hypotheses before using new observations for fitting",
                                                compute_cost=2 * len(fresh),
                                                details={"revision_loss": revised_loss, "parent_loss": parent_loss}))
            observations.extend(fresh)
            new_groups = observed_classes(candidates, observations, fallback, compute)
            actions.append(ActionRecord(round_number, "update_observable_classes", "Keep programs and revise their empirical grouping after new evidence",
                                        details={"before": old_groups, "after": new_groups,
                                                 "before_complete": old_groups_complete,
                                                 "after_complete": sum(map(len, new_groups)) == len(candidates),
                                                 "programs_retained": len(candidates), "tolerance_digits": 6}))

        compute.limit = config.compute_budget
        train, validation = _split(observations, config.validation_fraction)
        validation_measured = compute.spend(2 * len(validation))
        current_baseline_loss = initial_loss
        if validation_measured:
            current_loss = loss(incumbent.program, validation, fallback, spec.risk_quantile)
            current_baseline_loss = loss(Program("const", value=fallback), validation, fallback, spec.risk_quantile)
        decision_model = {}
        if spec.decision_costs is not None:
            before_decision_compute = compute.spent
            decision_model = fit_decision_model(incumbent.program, train, validation, fallback=fallback,
                                                costs=spec.decision_costs, budget=compute)
            actions.append(ActionRecord(config.max_rounds, "revise_uncertainty",
                                        "Select an empirical uncertainty model and decision under fixed task costs",
                                        compute_cost=compute.spent - before_decision_compute, details=decision_model))
        memory_events, added = [], []
        if config.use_memory and validation_measured:
            for candidate in retrieved:
                if candidate.operation != "reuse":
                    continue
                if incumbent.operation == "reuse" and candidate.memory_id == incumbent.memory_id:
                    self.library.record_use(candidate.memory_id, spec.task_id, spec=spec, observations=observations)
                    continue
                if not compute.spend(len(validation)):
                    break
                old_loss = loss(candidate.program, validation, fallback, spec.risk_quantile)
                if old_loss > current_loss + max(0.05, current_loss * 0.25):
                    if not compute.spend(len(validation)):
                        break
                    predicted_examples = [(row, predict(candidate.program, row, fallback)) for row in validation]
                    example, prediction = max(predicted_examples, key=lambda pair: abs(pair[0].target - pair[1]))
                    memory_events.append(self.library.record_counterexample(
                        candidate.memory_id, spec, old_loss, current_loss,
                        {"features": example.features, "target": example.target,
                         "history": list(example.history),
                         "prediction": prediction}))
            if current_loss + config.minimum_improvement < current_baseline_loss and incumbent.operation != "reuse":
                identifier, created = self.library.learn(incumbent, spec, observations=observations)
                if created:
                    added.append(identifier)
                    memory_events.append({"action": "learn", "concept_id": identifier,
                                          "expression": incumbent.program.describe()})
        attempted_calls = [entry for entry in llm_records if entry.get("calls")]
        llm_usage = {"calls": sum(entry.get("calls", 0) for entry in llm_records),
                     "wall_time_seconds": sum(entry.get("wall_time_seconds", 0) for entry in llm_records),
                     "records": llm_records,
                     "costs_complete": all(entry.get("total_tokens") is not None for entry in attempted_calls)}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            llm_usage[key] = (sum(entry.get(key, 0) or 0 for entry in attempted_calls)
                              if llm_usage["costs_complete"] else None)
        return EpisodeResult(spec.task_id, incumbent.program, fallback, observations, actions, questions,
                             observation_cost, compute.spent, current_loss if validation_measured else None, initial_loss,
                             incumbent.operation, incumbent.memory_id, added, memory_events, stop_reason,
                             plans=plan_records, wall_time_seconds=perf_counter() - started_at,
                             llm_usage=llm_usage, uncertainty_model=decision_model,
                             validation_rows=len(validation) if validation_measured else 0)
