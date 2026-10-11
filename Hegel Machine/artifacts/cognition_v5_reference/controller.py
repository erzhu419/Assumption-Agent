"""Discover models, compare prospective predictions, and revise reusable knowledge."""
from __future__ import annotations

from dataclasses import asdict
from math import sqrt
from random import Random
from statistics import mean, pstdev
from time import perf_counter
from typing import Protocol

from .diagnosis import diagnose, research_steps
from .evaluation import initial_prediction, loss, point_loss, predict
from .language import Program
from .memory import KnowledgeLibrary
from .models import ActionRecord, Candidate, EpisodeResult, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from .planning import execute_revision_plan, revision_plans
from .probing import choose_probe, choose_valuable_probe, construct_probes, observed_classes, transition_probe
from .selection import PlanAllocation, rank_candidates, representative_probe, warm_start_candidates, screen_candidates
from .sparse_search import sparse_candidates
from .synthesis import synthesize
from .uncertainty import fit_decision_model


class LearningEnvironment(Protocol):
    spec: TaskSpec
    def observe(self, probe: Probe) -> list[Observation]: ...


ALL_OPERATIONS = ("refit", "representation", "relation", "composition", "robustification", "idealization")


def _split(rows, fraction):
    """Hold out whole acquired trajectories for residual model calibration."""
    groups = []
    for row in rows:
        if not row.history or not groups:
            groups.append([])
        groups[-1].append(row)
    if len(groups) < 2:
        cut = max(1, len(rows) - max(1, round(len(rows) * fraction)))
        return rows[:cut], rows[cut:]
    count = max(1, round(len(groups) * fraction))
    return [row for group in groups[:-count] for row in group], [row for group in groups[-count:] for row in group]


class CognitiveAgent:
    def __init__(self, config=None, library=None, proposer=None):
        self.config = config or ResearchConfig()
        self.library = library if library is not None else KnowledgeLibrary()
        self.proposer = proposer

    def run(self, environment: LearningEnvironment):
        config, spec = self.config, environment.spec
        started = perf_counter()
        reserve = min(config.compute_budget // 4, config.observation_budget * (4 + len(spec.feature_names)))
        if spec.decision_costs is not None and config.risk_learning:
            reserve = min(config.compute_budget // 3, 3000)
        compute = SearchBudget(config.compute_budget - reserve)
        learning_limit = compute.limit
        observations, reference, queried = [], [], []
        actions, questions, plan_records, llm_records = [], [], [], []
        llm_probes, observation_cost = [], 0
        calibration_reserve = min(16, config.observation_budget // 4) if spec.decision_costs is not None and config.risk_learning else 0
        observation_limit = config.observation_budget - calibration_reserve
        rng, allocation = Random(config.seed), PlanAllocation()
        proposer = self.proposer
        if proposer is None and config.proposer in {"codex", "hybrid"}:
            from .proposers import CodexProposer
            proposer = CodexProposer(timeout_seconds=config.llm_timeout_seconds, token_budget=config.llm_token_budget,
                                     max_calls=config.llm_calls_per_task, max_history=config.max_history)

        def acquire(probe, turn, reason):
            nonlocal observation_cost
            if probe is None or probe.cost > observation_limit - observation_cost:
                return []
            rows = environment.observe(probe)
            observation_cost += probe.cost
            queried.append(probe)
            actions.append(ActionRecord(turn, "observe", reason, observation_cost=probe.cost,
                                        details={"probe_id": probe.probe_id, "rows": len(rows), "sequence": list(probe.sequence)}))
            return rows

        for probe in spec.initial_probes:
            observations.extend(acquire(probe, 0, "Initial observations under the common task interface"))
        if len(observations) < 3:
            raise ValueError("observation budget must fund at least three initial observations")
        fallback = initial_prediction(observations, spec.risk_quantile)
        initial = Candidate("initial", Program("const", value=fallback), "refit")
        incumbent, parent = initial, None
        if not compute.spend(len(observations)):
            raise ValueError("compute budget cannot fund the initial candidate comparison")
        initial_loss = loss(initial.program, observations, fallback, spec.risk_quantile)
        retrieved = self.library.retrieve(spec) if config.use_memory else []
        if config.use_memory and config.use_abstractions:
            local = SearchBudget(min(2400, compute.remaining // 5))
            retrieved.extend(self.library.propose(spec, observations, local, structural_transfer=config.structural_transfer,
                                                  preserve_parameters=config.transfer_scheduling))
            compute.spend(local.spent)
            actions.append(ActionRecord(0, "adapt_knowledge", "Fit structures ranked by current evidence and transfer experience",
                                        compute_cost=local.spent, details={"candidates": len(retrieved),
                                        "search": self.library.last_search}))
        bank, stop_reason = list(retrieved), "round_budget"
        if retrieved:
            before = compute.spent
            tested = rank_candidates(retrieved, observations, [], fallback=fallback, q=spec.risk_quantile,
                                     budget=compute, complexity_weight=config.complexity_weight)
            actions.append(ActionRecord(0, "retrieve", "Measure transferred candidates before adopting a task explanation",
                                        compute_cost=compute.spent-before, details={"retrieved": [c.memory_id for c in retrieved],
                                        "tested": [{"memory_id": r.candidate.memory_id, "operation": r.candidate.operation,
                                                    "loss": r.evidence_loss} for r in tested], "selected": None}))
            plausible_reuse = bool(tested and tested[0].evidence_loss <= config.minimum_improvement)
            bank = [row.candidate for row in tested[:config.max_candidates]]
        else:
            plausible_reuse = False

        for turn in range(config.max_rounds):
            # Keep another fitting/comparison round affordable after this
            # round's observations; unused work carries into later rounds.
            compute.limit = compute.spent + (learning_limit - compute.spent) // (config.max_rounds - turn)
            if not compute.spend(len(observations)):
                stop_reason = "compute_budget"
                break
            question = diagnose(incumbent.program, observations, fallback, spec.risk_quantile,
                                max_depth=config.max_question_depth)
            questions.append(question)
            actions.append(ActionRecord(turn, "diagnose", question.question, compute_cost=len(observations), details=asdict(question)))
            expected_length = min(spec.max_probe_length, 4 * config.max_history + 4) if config.adaptive_horizon else 8
            per_comparison = len(observations) + len(reference) + 2 * expected_length
            slots = min(config.max_candidates, 8, max(2, compute.remaining // max(1, 5 * per_comparison)))
            comparison_reserve = (slots + 2) * per_comparison
            screening_allowance = min(700, compute.remaining // 7)
            startup_allowance = min(500, compute.remaining // 10)
            probe_allowance = min(500, compute.remaining // 10)
            allowance = max(0, compute.remaining - comparison_reserve - screening_allowance - startup_allowance - probe_allowance)
            proposed, plans = [], []
            if proposer is not None:
                batch = proposer.propose(observations, incumbent.program, spec, compute)
                proposed.extend(batch.candidates)
                llm_probes.extend(batch.probes)
                llm_records.append(batch.usage)
                actions.append(ActionRecord(turn, "llm_proposal", "Execute model-proposed programs under the common task interface",
                                            details={"candidates": len(batch.candidates), "usage": batch.usage}))
            if config.proposer != "codex":
                if config.strategy == "fixed" or (turn == 0 and plausible_reuse):
                    local = SearchBudget(min(allowance, 500))
                    proposed.extend(synthesize(observations, operations=("refit",), budget=local,
                                               incumbent=incumbent.program, risk_quantile=spec.risk_quantile))
                    compute.spend(local.spent)
                    actions.append(ActionRecord(turn, "refit", "Compare inexpensive current-data fits with transferred knowledge", compute_cost=local.spent))
                else:
                    if config.state_search and spec.risk_quantile == 0.5:
                        from .state_search import state_candidates
                        local = SearchBudget(min(allowance // 3, 900))
                        states = state_candidates(observations, budget=local, max_history=config.max_history)
                        proposed.extend(states)
                        compute.spend(local.spent)
                        allowance -= local.spent
                        if local.spent:
                            actions.append(ActionRecord(turn, "state_search", "Learn conditional transitions from observed output states",
                                                compute_cost=local.spent, details={"candidates": len(states)}))
                    before = compute.spent
                    local = SearchBudget(min(allowance * 4 // 5, 2600))
                    if config.risk_learning and spec.risk_quantile != 0.5:
                        from .risk_search import risk_candidates
                        proposed.extend(risk_candidates(observations, budget=local, q=spec.risk_quantile, tail_fit=config.tail_fit))
                    else:
                        proposed.extend(sparse_candidates(observations, budget=local, max_history=config.max_history,
                                                         strategy=config.joint_search))
                    compute.spend(local.spent)
                    allowance -= local.spent
                    actions.append(ActionRecord(turn, "composition", "Select and combine coordinates explaining residual variation",
                                                compute_cost=compute.spent-before, details={"candidates": len(proposed)}))
                    if config.joint_planning and config.strategy == "adaptive":
                        plans = revision_plans(question, depth=config.planning_depth)
                        quotas = allocation.allocate(plans, allowance)
                        for plan, quota in zip(plans, quotas):
                            local = SearchBudget(min(quota, compute.remaining))
                            proposed.extend(execute_revision_plan(plan, observations, incumbent=incumbent.program,
                                            budget=local, risk_quantile=spec.risk_quantile, max_history=config.max_history))
                            compute.spend(local.spent)
                            plan_records.append({"round": turn, "allocated_compute": quota, **plan.to_dict()})
                            for stage in plan.steps:
                                actions.append(ActionRecord(turn, stage["operation"], plan.question,
                                                            compute_cost=stage["compute_cost"], details=stage))
                    else:
                        operations = ALL_OPERATIONS if config.strategy == "search" else tuple(op for op, _ in research_steps(question))
                        local = SearchBudget(min(allowance, compute.remaining))
                        rows = [r for r in observations if len(r.history) >= min(config.max_history, max(map(lambda r: len(r.history), observations))) ]
                        if len(rows) < 4:
                            rows = observations
                        proposed.extend(synthesize(rows, operations=operations, budget=local,
                                                   incumbent=incumbent.program, risk_quantile=spec.risk_quantile,
                                                   max_history=config.max_history))
                        compute.spend(local.spent)
                        actions.append(ActionRecord(turn, "search", "Compare independent revisions in the shared language", compute_cost=local.spent))
            pool = [incumbent, *([parent] if parent else [])]
            for index in range(max(len(bank), len(proposed))):
                if index < len(proposed):
                    pool.append(proposed[index])
                if index < len(bank):
                    pool.append(bank[index])
            if config.use_warm_start and config.strategy != "fixed":
                local = SearchBudget(min(startup_allowance, max(0, compute.remaining - comparison_reserve)))
                warm = warm_start_candidates(pool, observations, budget=local, q=spec.risk_quantile,
                                             complexity_weight=config.complexity_weight)
                compute.spend(local.spent)
                pool.extend(warm)
                actions.append(ActionRecord(turn, "initial_representation", "Learn predictions before the chosen historical representation is available",
                                            compute_cost=local.spent, details={"candidates": len(warm)}))
            before = compute.spent
            local = SearchBudget(min(screening_allowance, max(0, compute.remaining - comparison_reserve)))
            screening = screen_candidates(pool, observations, reference, fallback=fallback, q=spec.risk_quantile,
                                          budget=local, complexity_weight=config.complexity_weight, maximum=slots)
            compute.spend(local.spent)
            actions.append(ActionRecord(turn, "screen_candidates", "Retain promising executable alternatives for a new experiment",
                                        compute_cost=compute.spent-before, details={"candidates": len(screening)}))
            if not screening:
                stop_reason = "compute_budget"
                break
            short = [r.candidate for r in screening[:slots]]
            origins_by_program = {row.candidate.program: row.origins for row in screening}
            for retained in (incumbent, parent):
                if retained and all(c.program != retained.program for c in short):
                    short.append(retained)
            short.sort(key=lambda candidate: candidate.program != incumbent.program)

            representative = representative_probe(spec, seed=config.seed, index=turn,
                            remaining=observation_limit-observation_cost, used=queried,
                            candidates=short, adaptive_horizon=config.adaptive_horizon)
            # An acquired experiment must have a funded comparison. Candidate
            # generation and query selection cannot borrow this allocation.
            per_comparison = len(observations) + len(reference) + 2 * (representative.cost if representative else 0)
            if compute.remaining < per_comparison * len(short):
                stop_reason = "compute_budget"
                continue
            fresh_reference = acquire(representative, turn, "Measure the frozen alternatives on a new representative control trajectory")
            fresh = list(fresh_reference)
            if not fresh:
                stop_reason = "observation_budget"
                break
            before = compute.spent
            prospective = config.selection_policy == "prospective"
            cumulative_risk = bool(config.risk_learning and config.tail_fit and spec.decision_costs is not None)
            fresh_weight = min(0.5, len(fresh_reference) / max(1, len(reference) + len(fresh_reference))) if cumulative_risk else 0.5
            assessed = rank_candidates(short, observations, [*reference, *fresh_reference] if prospective else [],
                        fresh=fresh_reference, fresh_weight=fresh_weight if prospective else 0.0,
                        fallback=fallback, q=spec.risk_quantile,
                        budget=compute, complexity_weight=config.complexity_weight)
            if not assessed:
                observations.extend(fresh)
                reference.extend(fresh_reference)
                stop_reason = "compute_budget"
                break
            for measured in assessed:
                measured.origins = origins_by_program.get(measured.candidate.program, measured.origins)
            best = assessed[0]
            current = next((r for r in assessed if r.candidate.program == incumbent.program), None)
            if current is None:
                observations.extend(fresh)
                reference.extend(fresh_reference)
                stop_reason = "compute_budget"
                break
            support = True
            margin, fresh_gain, evidence_gain = 0.0, None, None
            if prospective and fresh_reference:
                # These differences are already represented in the paid fresh
                # evaluations. The dispersion is a decision heuristic, not a
                # hypothesis-test certificate from correlated trajectory rows.
                differences = [point_loss(predict(incumbent.program, r, fallback), r.target, spec.risk_quantile)
                               - point_loss(predict(best.candidate.program, r, fallback), r.target, spec.risk_quantile)
                               for r in ([*reference, *fresh_reference] if cumulative_risk else fresh_reference)]
                fresh_gain = mean(differences[-len(fresh_reference):])
                evidence_gain = mean(differences)
                margin = 0.5 * pstdev(differences) / sqrt(len(differences))
                support = evidence_gain >= max(config.minimum_improvement, margin) or (
                    abs(evidence_gain) <= config.minimum_improvement and best.candidate.program.size < incumbent.program.size)
                # New interventions may localize a boundary that a fresh
                # representative sample misses. An equally compact correction
                # can use that evidence when its new predictions do not worsen.
                if spec.decision_costs is None and fresh_gain >= -config.minimum_improvement:
                    support = support or (best.evidence_loss + config.minimum_improvement < current.evidence_loss
                                          and best.candidate.program.size <= incumbent.program.size)
            adopt = support and best.score + config.minimum_improvement < current.score
            actions.append(ActionRecord(turn, "compare", "Compare task-representative evidence and prospective predictions",
                                        compute_cost=compute.spent-before, details={"candidates_scored": len(assessed),
                                        "best_expression": best.candidate.program.describe(), "best_loss": best.reference_loss,
                                        "fresh_gain": fresh_gain, "evidence_gain": evidence_gain, "dispersion_margin": margin, "adopt": adopt,
                                        "selection_policy": config.selection_policy,
                                        "decision_evidence": "accumulated_risk_trajectories" if cumulative_risk else "new_trajectory",
                                        "fresh_weight": fresh_weight, "validation_is_adaptive_development": True}))
            if plans and current.fresh_loss is not None:
                allocation.update(plans, assessed, current.fresh_loss)
            if adopt:
                if parent and best.candidate.program == parent.program:
                    actions.append(ActionRecord(turn, "rollback", "The retained previous hypothesis is favored by accumulated representative evidence",
                                                details={"revised_loss": current.fresh_loss, "parent_loss": best.fresh_loss,
                                                "rejected_program": incumbent.program.to_dict(), "restored_program": parent.program.to_dict(),
                                                "rejected_expression": incumbent.program.describe()}))
                    incumbent, parent = best.candidate, None
                else:
                    parent, incumbent = incumbent, best.candidate
                    actions.append(ActionRecord(turn, "revise", "Adopt an executable revision supported by the new experiment",
                                                details={"operation": incumbent.operation, "before": parent.program.describe(),
                                                "after": incumbent.program.describe(), "before_program": parent.program.to_dict(),
                                                "after_program": incumbent.program.to_dict(), "development_loss": best.reference_loss}))
            if parent:
                parent_score = next((r for r in assessed if r.candidate.program == parent.program), None)
                current_score = next((r for r in assessed if r.candidate.program == incumbent.program), None)
                if parent_score and current_score and fresh_reference:
                    actions.append(ActionRecord(turn, "prediction_check", "Keep the previous explanation available without a one-trajectory rollback rule",
                                                details={"revision_loss": current_score.fresh_loss, "parent_loss": parent_score.fresh_loss}))
            for measured in assessed:
                for candidate in measured.origins:
                    if candidate.memory_id and measured.fresh_loss is not None:
                        accepted = candidate.program == incumbent.program
                        self.library.record_transfer_result(candidate, spec, accepted=accepted,
                                      loss=measured.fresh_loss, baseline_loss=current.fresh_loss)
                        actions.append(ActionRecord(turn, "reuse_check", "Measure transferred knowledge on a new representative trajectory",
                                                    details={"loss": measured.fresh_loss, "memory_id": candidate.memory_id,
                                                    "expression": candidate.program.describe(), "accepted": accepted,
                                                    "prediction_shared_by_origins": len(measured.origins)}))
            chosen = next(r for r in assessed if r.candidate.program == incumbent.program)
            before = compute.spent
            # Keep the next representative comparison affordable in observations
            # too. A final-round intervention cannot revise a deterministic model.
            active_remaining = max(0, observation_limit-observation_cost - min(8, spec.max_probe_length))
            if spec.decision_costs is None and turn == config.max_rounds-1:
                active_remaining = 0
            transition_work = SearchBudget(min(probe_allowance, compute.remaining))
            transition, transition_details = None, {}
            if config.transition_probes and config.active_probes and config.strategy != "fixed":
                transition, transition_details = transition_probe(spec, [incumbent, *short[:3]], [*observations, *fresh_reference],
                    budget=transition_work, remaining=active_remaining, round_number=turn, used=queried)
            compute.spend(transition_work.spent)
            fresh_active = []
            reuse_confirmed = bool(config.transfer_scheduling and incumbent.memory_id and transition is None
                and spec.decision_costs is None and fresh_reference
                and chosen.reference_loss <= config.minimum_improvement
                and chosen.evidence_loss <= config.minimum_improvement
                and chosen.fresh_loss <= config.minimum_improvement)
            refit_transfer_first = bool(config.transfer_scheduling and not reuse_confirmed
                and chosen.fresh_loss is not None and chosen.fresh_loss > config.minimum_improvement and any(
                r.candidate.memory_id and r.evidence_loss <= config.minimum_improvement
                and r.fresh_loss is not None and r.fresh_loss > config.minimum_improvement
                for r in assessed))
            if refit_transfer_first:
                actions.append(ActionRecord(turn, "plan_experiment", "Use the new counterexample to refit transferred knowledge before buying another trajectory",
                                            compute_cost=compute.spent-before,
                                            details={"policy": "transfer_refit_first"}))
            elif reuse_confirmed:
                actions.append(ActionRecord(turn, "plan_experiment", "Transferred knowledge explains the existing and newly acquired trajectory",
                                            compute_cost=compute.spent-before,
                                            details={"policy": "supported_transfer_stop", "memory_id": incumbent.memory_id}))
            else:
                affordable = [p for p in [*spec.available_probes, *llm_probes,
                              *(construct_probes(spec, short, round_number=turn, seed=config.seed, max_history=config.max_history)
                                if config.synthesize_probes else [])]
                              if p.cost <= active_remaining and all(p.sequence != old.sequence for old in queried)]
                experiment_details, active = transition_details, transition
                local = SearchBudget(min(probe_allowance-transition_work.spent, compute.remaining))
                if active is None and affordable and config.active_probes and config.strategy != "fixed":
                    if config.experiment_policy == "task_value":
                        active, experiment_details = choose_valuable_probe(affordable, short, fallback, local,
                                     observations=[*observations, *fresh_reference], spec=spec, queried_probes=queried,
                                     max_compute=local.remaining)
                    else:
                        active, value = choose_probe(affordable, short, fallback, local)
                        experiment_details = {"disagreement": value, "policy": "disagreement"}
                elif active is None and affordable:
                    active = rng.choice(affordable)
                    experiment_details = {"policy": "random"}
                compute.spend(local.spent)
                actions.append(ActionRecord(turn, "plan_experiment", "Choose additional observations for task value and unresolved structure",
                                            compute_cost=compute.spent-before, details=experiment_details))
                fresh_active = acquire(active, turn, "Distinguish remaining explanations with an affordable intervention")
                fresh = [*fresh_reference, *fresh_active]
            targeted_support = True
            if fresh_active:
                targeted_support = False
                if compute.spend(len(fresh_active)):
                    targeted_loss = loss(incumbent.program, fresh_active, fallback, spec.risk_quantile)
                    targeted_support = targeted_loss <= config.minimum_improvement
                    actions.append(ActionRecord(turn, "targeted_prediction_check", "Use the new intervention to decide whether another revision round is needed",
                                                compute_cost=len(fresh_active), details={"loss": targeted_loss,
                                                "further_revision_needed": not targeted_support}))
            before_groups = observed_classes(short, observations, fallback, compute)
            observations.extend(fresh)
            reference.extend(fresh_reference)
            after_groups = observed_classes(short, observations, fallback, compute)
            actions.append(ActionRecord(turn, "update_observable_classes", "Revise empirical groups while retaining distinct programs",
                                        details={"before": before_groups, "after": after_groups,
                                        "before_complete": sum(map(len, before_groups)) == len(short),
                                        "after_complete": sum(map(len, after_groups)) == len(short), "programs_retained": len(short)}))
            bank = [r.candidate for r in assessed[:config.max_candidates]
                    if not (config.transfer_scheduling and r.candidate.memory_id
                            and r.evidence_loss > chosen.evidence_loss + config.minimum_improvement
                            and r.fresh_loss is not None and chosen.fresh_loss is not None
                            and r.fresh_loss > chosen.fresh_loss + config.minimum_improvement)]
            transition_unresolved = bool(transition and transition_details.get("relative_width", 0.) > 2 * config.minimum_improvement)
            if reuse_confirmed or (not transition_unresolved and targeted_support and chosen.reference_loss <= config.minimum_improvement and chosen.evidence_loss <= config.minimum_improvement
                    and (turn > 0 or incumbent.program.size <= 3)):
                stop_reason = "knowledge_reused" if incumbent.memory_id else "adequate_and_compact"
                break

        compute.limit = config.compute_budget
        if calibration_reserve:
            train = list(observations)
            observation_limit = config.observation_budget
            validation = []
            for calibration_index in range(2 if config.tail_fit else 1):
                remaining = min(calibration_reserve - len(validation), observation_limit-observation_cost)
                calibration = representative_probe(spec, seed=config.seed, index=config.max_rounds + 1 + calibration_index,
                              remaining=remaining, used=queried, candidates=[incumbent],
                              adaptive_horizon=config.adaptive_horizon if not config.tail_fit else False)
                validation.extend(acquire(calibration, config.max_rounds, "Calibrate decisions after the prediction program has been selected"))
            observations.extend(validation)
        else:
            train, validation = _split(observations, config.validation_fraction)
        measured = compute.spend(2 * len(reference or validation))
        final_rows = reference or validation
        final_loss = loss(incumbent.program, final_rows, fallback, spec.risk_quantile) if measured else None
        baseline_loss = loss(initial.program, final_rows, fallback, spec.risk_quantile) if measured else None
        decision_model = {}
        if spec.decision_costs is not None:
            before = compute.spent
            decision_model = fit_decision_model(incumbent.program, train, validation, fallback=fallback,
                                                costs=spec.decision_costs, budget=compute, adaptive=config.risk_learning, tail_fit=config.tail_fit)
            if calibration_reserve:
                decision_model["calibration_source"] = "new_trajectory_after_program_selection"
            actions.append(ActionRecord(config.max_rounds, "revise_uncertainty", "Learn residual uncertainty under unchanged task costs",
                                        compute_cost=compute.spent-before, details=decision_model))
        events, added = [], []
        if config.use_memory and measured:
            for candidate in retrieved:
                if candidate.operation != "reuse":
                    continue
                if incumbent.operation == "reuse" and incumbent.program == candidate.program:
                    self.library.record_use(candidate.memory_id, spec.task_id, spec=spec, observations=observations)
                    continue
                if not compute.spend(len(final_rows)):
                    break
                old_loss = loss(candidate.program, final_rows, fallback, spec.risk_quantile)
                if old_loss > final_loss + max(0.05, final_loss * 0.25) and compute.spend(len(final_rows)):
                    row = max(final_rows, key=lambda r: abs(r.target-predict(candidate.program, r, fallback)))
                    events.append(self.library.record_counterexample(candidate.memory_id, spec, old_loss, final_loss, row))
            if final_loss + config.minimum_improvement < baseline_loss and incumbent.operation != "reuse":
                identifier, created = self.library.learn(incumbent, spec, observations=observations)
                if created:
                    added.append(identifier)
                    events.append({"action": "learn", "concept_id": identifier, "expression": incumbent.program.describe()})
        called = [r for r in llm_records if r.get("calls")]
        usage = {"calls": sum(r.get("calls", 0) for r in called), "records": llm_records,
                 "wall_time_seconds": sum(r.get("wall_time_seconds", 0) for r in called),
                 "costs_complete": all(r.get("total_tokens") is not None for r in called)}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] = sum(r.get(key, 0) or 0 for r in called) if usage["costs_complete"] else None
        return EpisodeResult(spec.task_id, incumbent.program, fallback, observations, actions, questions,
                             observation_cost, compute.spent, final_loss, initial_loss, incumbent.operation,
                             incumbent.memory_id, added, events, stop_reason, plans=plan_records,
                             wall_time_seconds=perf_counter()-started, llm_usage=usage,
                             uncertainty_model=decision_model, validation_rows=len(final_rows) if measured else 0)
