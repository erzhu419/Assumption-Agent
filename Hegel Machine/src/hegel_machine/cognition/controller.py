"""Discover models, compare prospective predictions, and revise reusable knowledge."""
from __future__ import annotations

from dataclasses import asdict, replace
from random import Random
from statistics import mean
from time import perf_counter
from typing import Protocol

from .diagnosis import diagnose, research_steps
from .action_policy import ActionOption, ResearchActionPolicy
from .continuous_state import continuous_state_candidates
from .evaluation import initial_prediction, loss, predict
from .candidate_comparison import compare_candidate_adoption
from .experiment_knowledge import research_context
from .method_conditions import method_profile, method_applicability
from .language import EvaluationContext, Program
from .knowledge_evolution import _calls, _mechanisms, _opened
from .memory import KnowledgeLibrary
from .models import ActionRecord, Candidate, EpisodeResult, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from .planning import RevisionPlan, execute_revision_plan, revision_plans
from .research_graph import ResearchGraph
from .probing import (choose_probe, choose_valuable_probe, choose_question_probe, construct_probes,
                      observed_classes, transition_probe, transition_outcome, _trajectory_predictions)
from .experiment_simulator import ExperimentSimulator
from .selection import (PlanAllocation, rank_candidates, representative_probe, warm_start_candidates,
                        screen_candidates, retain_history_frontier, retain_research_comparison_pool)
from .sparse_search import sparse_candidates
from .synthesis import _terms, synthesize
from .uncertainty import action_and_interval, decision_loss, fit_decision_model, select_decision_plan, risk_calibration_quote
from .decision_evidence import begin_decision_evidence, extend_decision_evidence, decision_evidence_quote
from .decision_probing import decision_probe_condition, plan_decision_probes
from .branch_support import history_branch_support
from .workspace import ResearchWorkspace
from .research_frontier import plan_experiment_frontier, frontier_probe, frontier_is_payable


class LearningEnvironment(Protocol):
    spec: TaskSpec
    def observe(self, probe: Probe) -> list[Observation]: ...


ALL_OPERATIONS = ("refit", "representation", "relation", "composition", "robustification", "idealization")


def _research_comparison(assessed, task_candidates, spec, observations, *, seed, turn,
                         remaining, maximum, fallback, budget, used=(), prediction_sink=None):
    """Fund a public forecast panel for experiment design, separate from adoption."""
    panel = representative_probe(spec, seed=seed, index=turn, remaining=remaining,
                                 candidates=task_candidates, adaptive_horizon=False, used=used)
    predictions = {}
    if panel is not None:
        for row in assessed:
            program = row.candidate.program.expanded()
            if program in predictions:
                continue
            if not budget.spend(panel.cost):
                break
            predictions[program] = tuple(_trajectory_predictions(row.candidate, panel, fallback, observations))
    if prediction_sink is not None:
        prediction_sink.update(predictions)
    retained, details = retain_research_comparison_pool(assessed, predictions, maximum=maximum)
    details.update(task_candidate_ids=[candidate.candidate_id for candidate in task_candidates],
                   panel_probe=asdict(panel) if panel else None,
                   compute_cost=budget.spent,
                   use='representative_experiment_design', adoption_changed=False)
    # Preserve every candidate required by the unchanged task comparison.
    candidates = list(task_candidates)
    for row in retained:
        if all(row.candidate.program != candidate.program for candidate in candidates):
            candidates.append(row.candidate)
    return candidates, details


def _refresh_active_research_pool(task_assessed, candidates, public_predictions,
                                  observations, reference, fresh, *, budget, maximum,
                                  fallback, q, complexity_weight, minimum_improvement,
                                  fresh_weight=.5):
    """Review design alternatives on the actual paid comparison, not screening samples.

    Task assessments are reused without modifying their ranking or adoption.
    Missing full assessments stay UNKNOWN and cannot vote in this research pool.
    The loss tolerance only qualifies this finite comparison, not a scientific
    family or a permanent transfer refutation.
    """
    start = budget.spent
    measured = {row.candidate.program.expanded(): row for row in task_assessed}
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate.program.expanded(), candidate)
    extras = [candidate for program, candidate in unique.items() if program not in measured]
    checked = rank_candidates(extras, observations, reference, fresh=fresh,
        fresh_weight=fresh_weight, fallback=fallback, q=q, budget=budget,
        complexity_weight=complexity_weight)
    measured.update((row.candidate.program.expanded(), row) for row in checked)
    complete = [measured[program] for program in unique if program in measured]
    fields = ('evidence_loss', 'reference_loss', 'fresh_loss')
    best = {field: min((getattr(row, field) for row in complete
                       if getattr(row, field) is not None), default=None) for field in fields}
    compatible, failures = [], []
    scope = {'observations': len(observations), 'reference': len(reference), 'fresh': len(fresh)}
    for row in complete:
        violated = [field for field in fields if getattr(row, field) is not None
                    and getattr(row, field) > best[field] + minimum_improvement]
        if violated:
            failures.append({'candidate_id': row.candidate.candidate_id,
                'status': 'outside_research_loss_tolerance', 'violated_losses': violated,
                'losses': {field: getattr(row, field) for field in fields},
                'evidence_scope': scope, 'scientific_family_refuted': False})
        else:
            compatible.append(row)
    retained, retention = retain_research_comparison_pool(
        compatible, public_predictions, maximum=maximum)
    details = {'status': 'complete' if len(complete) == len(unique) else 'partial_unknown',
        'input_candidate_ids': [candidate.candidate_id for candidate in unique.values()],
        'reused_task_candidate_ids': [row.candidate.candidate_id for row in complete
            if row in task_assessed],
        'fully_reviewed_candidate_ids': [row.candidate.candidate_id for row in complete],
        'unknown_candidate_ids': [candidate.candidate_id for program, candidate in unique.items()
                                  if program not in measured],
        'scoped_failures': failures, 'loss_tolerance': minimum_improvement,
        'criterion': 'finite_research_loss_heuristic_best_plus_minimum_improvement',
        'statistical_refutation_established': False,
        'prediction_semantics': 'native_predict_on_acquired_row_history',
        'best_losses': best, 'evidence_scope': scope, 'retention': retention,
        'compute_cost': budget.spent-start, 'task_adoption_changed': False}
    return [row.candidate for row in retained], {row.candidate.program: row for row in retained}, details


def _check_active_research_predictions(candidates, fresh, *, observation_offset, budget,
                                       fallback, q, minimum_improvement):
    """Check the acquired intervention, keeping scoped failures and incomplete work."""
    start = budget.spent
    checked = rank_candidates(candidates, fresh, [], fallback=fallback, q=q,
        budget=budget, complexity_weight=0.) if fresh else []
    best = min((row.evidence_loss for row in checked), default=None)
    values = {row.candidate.program.expanded(): row for row in checked}
    result = {'status': 'complete' if len(values) == len(candidates) else 'partial_unknown',
        'evidence_ref': {'field': 'observations',
            'row_indices': list(range(observation_offset, observation_offset+len(fresh)))},
        'loss_tolerance': minimum_improvement, 'best_loss': best,
        'criterion': 'finite_research_loss_heuristic_best_plus_minimum_improvement',
        'statistical_refutation_established': False,
        'prediction_semantics': 'native_predict_on_acquired_row_history',
        'checks': [], 'compute_cost': budget.spent-start, 'task_adoption_changed': False}
    for candidate in candidates:
        row = values.get(candidate.program.expanded())
        result['checks'].append({'candidate_id': candidate.candidate_id,
            'status': ('UNKNOWN' if row is None else 'outside_research_loss_tolerance'
                       if row.evidence_loss > best + minimum_improvement else 'within_research_loss_tolerance'),
            'loss': row.evidence_loss if row else None, 'scientific_family_refuted': False})
    return result


def _representation_work(observations, spec, max_history):
    """Fund complete sparse initialization, including broad-fit pruning.

    Current columns are shared by all depths. Each historical extension adds
    one column, a broad fit, and (when wider than four terms) a pruning fit.
    The quote uses the actual observable column inventory of sparse search.
    """
    if spec.risk_quantile != .5 or all(row.target in (0., 1.) for row in observations):
        return 0
    current, _ = _terms(observations, 0)
    width = len(current)
    current_terms = 2 * width + width * (width - 1) // 2
    history_rows = {depth: sum(len(row.history) >= depth for row in observations)
                    for depth in range(1, max_history + 1)}
    deepest = max((depth for depth, count in history_rows.items() if count >= 6), default=0)
    historical = _terms([row for row in observations if len(row.history) >= deepest], deepest)[1] if deepest else []
    return len(observations) * (current_terms + 1 + (current_terms > 4)) + sum(
        history_rows[term.lag] * (2 + (current_terms + 1 > 4)) for term in historical)


def _continuous_state_work(observations, spec):
    """Price a complete pole search and nonlinear readout on acquired rows."""
    if (spec.risk_quantile != .5 or spec.decision_costs is not None or len(observations) < 6
            or len({row.target for row in observations}) <= 2
            or max((len(row.history) for row in observations), default=0) < 2):
        return 0
    return 450 * len(observations)


def _refit_transferred(library, incumbent, spec, observations, budget, config, fallback):
    """Price and fit this explanation's source on the evidence already acquired.

    The returned fits still need the ordinary subsequent trajectory comparison.
    They test whether changed parameters suffice before buying another structure.
    """
    fitting = SearchBudget(max(0, budget.remaining - 2 * len(observations)))
    candidates = library.propose(spec, observations, fitting,
        structural_transfer=config.structural_transfer, preserve_parameters=False,
        source_concept_ids={incumbent.memory_id})
    budget.spend(fitting.spent)
    return rank_candidates(candidates, observations, [], fallback=fallback,
        q=spec.risk_quantile, budget=budget, complexity_weight=config.complexity_weight)


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

    def run(self, environment: LearningEnvironment, *, research_state=None, research_selection=None):
        config, spec = self.config, environment.spec
        if research_selection is not None and (not config.active_probes or config.strategy == 'fixed'
                                               or spec.decision_costs is not None):
            raise ValueError('native PR2 requires the active scalar-response research path')
        workspace = (ResearchWorkspace.from_dict(research_state) if research_state is not None
                     else ResearchWorkspace(spec.task_id))
        if workspace.task_id != spec.task_id:
            raise ValueError("research evidence belongs to a different task")
        simulator = ExperimentSimulator(workspace.simulator_state)
        resumed = workspace.incumbent is not None
        execution_plan = workspace.research_intent.get("execution_plan", {})
        graph = ResearchGraph.from_dict(workspace.graph)
        evidence_champion = workspace.evidence_champion
        started = perf_counter()
        reserve = min(config.compute_budget // 4, config.observation_budget * (4 + len(spec.feature_names)))
        if resumed:
            reserve = min(config.compute_budget // 3, max(reserve,
                2 * len(workspace.reference) + 4 * config.observation_budget))
        if spec.decision_costs is not None and config.risk_learning:
            reserve = min(config.compute_budget // 3, 3000)
        if execution_plan:
            reserve = min(config.compute_budget, execution_plan["report_compute_budget"])
        compute = SearchBudget(config.compute_budget - reserve)
        learning_limit = compute.limit
        observations, reference, queried = workspace.observations, workspace.reference, workspace.queried
        actions, questions, plan_records, llm_records = [], [], [], []
        llm_probes, observation_cost = [], 0
        calibration_reserve = 0
        if spec.decision_costs is not None and config.risk_learning:
            depth = max((candidate.program.history_depth for candidate in
                         [workspace.incumbent, *workspace.bank] if candidate is not None), default=0)
            risk_quote = risk_calibration_quote(training_rows=0, validation_rows=0,
                training_trajectories=0, feature_count=len(spec.feature_names),
                history_depth=depth, max_probe_length=spec.max_probe_length, tail_fit=config.tail_fit)
            calibration_reserve = min(config.observation_budget // 2,
                risk_quote["comparison_observation_cost"] + risk_quote["followup_observation_cost"])
        observation_limit = config.observation_budget - calibration_reserve
        rng, allocation = Random(config.seed), PlanAllocation()
        proposer = self.proposer
        if proposer is None and config.proposer in {"codex", "hybrid"}:
            from .proposers import CodexProposer
            proposer = CodexProposer(timeout_seconds=config.llm_timeout_seconds, token_budget=config.llm_token_budget,
                                     max_calls=config.llm_calls_per_task, max_history=config.max_history)

        def acquire(probe, turn, reason, design=None):
            nonlocal observation_cost
            if probe is None or probe.cost > observation_limit - observation_cost:
                return []
            rows = environment.observe(probe)
            observation_cost += probe.cost
            queried.append(probe)
            actions.append(ActionRecord(turn, "observe", reason, observation_cost=probe.cost,
                                        details={"probe_id": probe.probe_id, "rows": len(rows), "sequence": list(probe.sequence)}))
            if design and design.get("boundary_targets"):
                outcome = transition_outcome(design, rows)
                design["outcome"] = outcome
                actions.append(ActionRecord(turn, "measure_experiment_coverage",
                    "Use observed histories to determine which planned transition conditions were reached",
                    details=outcome))
                if pending_actions:
                    pending_actions[-1]["experiment_outcome"] = outcome
            resolved = graph.supply_measurements(rows)
            if resolved:
                actions.append(ActionRecord(turn, "supply_research_measurements", "Deliver acquired measurements to their dependent questions",
                    details={"questions": resolved, "rows": len(rows)}))
            return rows

        if observations:
            actions.append(ActionRecord(workspace.rounds_completed, "resume_research",
                "Continue the saved question using its acquired evidence and executable alternatives",
                details={"restored_observations": len(observations), "restored_reference_rows": len(reference),
                         "restored_candidates": len(workspace.bank), "initial_queries_repeated": False}))
        else:
            for probe in spec.initial_probes:
                observations.extend(acquire(probe, 0, "Initial observations under the common task interface"))
        if len(observations) < 3:
            raise ValueError("observation budget must fund at least three initial observations")
        if config.use_memory and config.learn_research_policy:
            allocation.remember(observations, spec, self.library.research_experience)
            actions.append(ActionRecord(0, "recall_research", "Recall observed returns from earlier research plans",
                                        details={"context": allocation.context,
                                                 "experience": dict(allocation.experience)}))
        fallback = workspace.fallback if resumed else initial_prediction(observations, spec.risk_quantile)
        initial = Candidate("initial", Program("const", value=fallback), "refit")
        incumbent, parent = (workspace.incumbent, workspace.parent) if resumed else (initial, None)
        visit_candidate = incumbent
        visit_incumbent = incumbent.program
        visit_reference_start = len(reference)
        visit_uncertainty = workspace.uncertainty_model
        if resumed:
            initial_loss = workspace.initial_loss
        else:
            if not compute.spend(len(observations)):
                raise ValueError("compute budget cannot fund the initial candidate comparison")
            initial_loss = loss(initial.program, observations, fallback, spec.risk_quantile)
        action_policy = ResearchActionPolicy(observations, spec,
            self.library.action_experience if config.use_memory and config.learn_research_policy else {})
        pending_actions = workspace.pending_actions
        method_trace = workspace.method_trace
        for pending in pending_actions:
            pending["start_compute"], pending["start_observations"] = 0, 0

        def objective_loss(program, rows, model=None):
            if spec.decision_costs is None:
                return loss(program, rows, fallback, spec.risk_quantile)
            return mean(decision_loss(action_and_interval(program, row, fallback, model or {}, spec.decision_costs)[0],
                                      row.target, spec.decision_costs) for row in rows)

        def retained_decision_model(program):
            return visit_uncertainty if program == visit_incumbent else {}

        def credit_actions(rows, turn, names=None):
            for pending in list(pending_actions):
                if names is not None and pending["name"] not in names:
                    continue
                if not rows or not compute.spend(3 * len(rows)):
                    break
                prior = Program.from_dict(pending["program"])
                after = Program.from_dict(pending["after_program"]) if "after_program" in pending else incumbent.program
                before_loss = objective_loss(prior, rows, pending.get("before_model"))
                after_loss = objective_loss(after, rows, pending.get("after_model", retained_decision_model(after)))
                normalizer = objective_loss(initial.program, rows) if spec.decision_costs else initial_loss
                feedback = action_policy.update(pending["name"], before_loss=before_loss, after_loss=after_loss,
                    observation_cost=pending["observation_cost"] + observation_cost - pending["start_observations"],
                    compute_cost=pending["compute_cost"] + compute.spent - pending["start_compute"], baseline_loss=normalizer,
                    context=pending.get("context"))
                state_steps = [action.details for action in actions if action.round == turn and action.action == "state_search"]
                needs_state_fit = (len({row.target for row in observations}) == 2 and
                                   pending["name"] in {"transition", "representative", "task_value", "state_transition"})
                execution_complete = not needs_state_fit or any(step.get("basic_fit_funded") for step in state_steps)
                actions.append(ActionRecord(turn, "action_feedback", "Measure the previous research action on a subsequent common trajectory",
                    compute_cost=3 * len(rows), details={"action": pending["name"], "context": pending.get("context", action_policy.context),
                                                       "reference_rows": len(rows), "execution_complete": execution_complete, **feedback}))
                if config.use_memory and config.learn_research_policy:
                    step = {"action": pending["name"], "context": pending.get("context", action_policy.context),
                            "gain": feedback["last_gain"], "compute_cost":
                                pending["compute_cost"] + compute.spent - pending["start_compute"],
                            "observation_cost": pending["observation_cost"] + observation_cost - pending["start_observations"],
                            "method_id": pending.get("method_id"), "profile": pending.get("profile"),
                            "after_program": after.to_dict(), "before_loss": before_loss, "after_loss": after_loss,
                            "execution_complete": execution_complete}
                    method_trace.append(step)
                    for key in ("requires_actions", "experiment_outcome", "execution_profile"):
                        if pending.get(key):
                            step[key] = pending[key]
                    method_id = self.library.methods.observe(step["context"], method_trace,
                        condition=pending.get("method_condition", research_context(spec, observations)), source_task=spec.task_id)
                    if method_id:
                        step["produced_method_id"] = method_id
                        actions.append(ActionRecord(turn, "revise_method", "Revise a research sequence using its measured subsequent return",
                            details={"method_id": method_id, "revision": self.library.methods.revisions[-1]}))
                pending_actions.remove(pending)

        def begin_action(name, *, start_compute, start_observations, before_model=None,
                         after_model=None, method_id=None, profile=None):
            pending = {"name": name, "context": action_policy.context, "program": incumbent.program.to_dict(),
                       "before_model": retained_decision_model(incumbent.program) if before_model is None else before_model,
                       "compute_cost": 0, "observation_cost": 0,
                       "start_compute": start_compute, "start_observations": start_observations}
            pending.update(method_id=method_id, method_condition=research_context(spec, observations))
            pending["profile"] = profile or method_profile(spec, observations, incumbent, method_trace,
                baseline_loss=initial_loss, minimum_improvement=config.minimum_improvement)
            if after_model is not None:
                pending.update(after_program=incumbent.program.to_dict(), after_model=after_model)
            pending_actions.append(pending)

        frontier = workspace.experiment_frontier
        if resumed and frontier and frontier.get("evidence_rows") == len(observations):
            if (workspace.research_intent.get("kind", "experiment") == "experiment"
                    and frontier_is_payable(frontier, observation_limit-observation_cost)):
                begin_action("transition", start_compute=compute.spent, start_observations=observation_cost,
                    profile=method_profile(spec, observations, incumbent, method_trace,
                        baseline_loss=initial_loss, minimum_improvement=config.minimum_improvement,
                        boundary_evidence=frontier["design"]))
                probe = frontier_probe(frontier)
                observations.extend(acquire(probe, workspace.rounds_completed,
                    "Continue the saved distinguishing experiment before revising the explanation",
                    design=frontier["design"]))
                actions.append(ActionRecord(workspace.rounds_completed, "execute_research_question",
                    frontier["question"], details={"probe_id": probe.probe_id,
                        "prior_uncertainty": frontier["uncertainty"], "design": frontier["design"]}))
                workspace.experiment_frontier = {}
        elif frontier:
            # New external evidence changes the question; the final planning
            # step will rebuild its actual experiment from those observations.
            workspace.experiment_frontier = {}
        request = workspace.requested_method
        if request and request["name"] in {"refit", "representation", "continuous_state", "state_transition"}:
            pending = next((item for item in pending_actions if item["name"] == request["name"]), None)
            if pending is None:
                begin_action(request["name"], start_compute=compute.spent, start_observations=observation_cost,
                             method_id=request["method_id"])
                pending = pending_actions[-1]
            pending.update(method_id=request["method_id"], context=request["context"],
                           method_condition=request["condition"])
            if request.get("profile"):
                pending["profile"] = request["profile"]
            actions.append(ActionRecord(workspace.rounds_completed, "execute_method_step",
                "Execute the concrete research step selected across tasks", details=dict(request)))
            workspace.requested_method = {}
        elif request and request["name"] in {"representative", "transition", "task_value"}:
            # A resumed method can acquire its planned evidence before fitting.
            # One research round still leaves a new common comparison afterward.
            remaining = max(0, observation_limit-observation_cost-min(8, spec.max_probe_length))
            work = SearchBudget(min(500, compute.remaining // 10))
            alternatives = list(dict.fromkeys([incumbent, *workspace.bank]))[:8]
            probe, details = None, {}
            if request["name"] == "representative":
                probe = representative_probe(spec, seed=config.seed, index=workspace.rounds_completed + 2000,
                    remaining=remaining, used=queried, candidates=alternatives, adaptive_horizon=config.adaptive_horizon)
            elif request["name"] == "transition":
                followup = representative_probe(spec, seed=config.seed, index=workspace.rounds_completed,
                    remaining=observation_limit-observation_cost, used=queried, candidates=alternatives,
                    adaptive_horizon=config.adaptive_horizon)
                followup_cost = followup.cost if followup else 0
                remaining = max(0, observation_limit-observation_cost-followup_cost)
                probe, details = transition_probe(spec, alternatives, observations, budget=work,
                    remaining=remaining, round_number=workspace.rounds_completed, used=queried,
                    joint_controls=config.joint_state_probes, followup_observation_cost=followup_cost,
                    remaining_rounds=config.max_rounds)
            else:
                offers = self.library.experiments.propose(spec, alternatives, observations,
                    fallback=fallback, budget=work, remaining=remaining, used=queried)
                offers = [probe for probe in [*offers, *spec.available_probes,
                    *construct_probes(spec, alternatives, round_number=workspace.rounds_completed, seed=config.seed,
                                      max_history=config.max_history)] if probe.cost <= remaining]
                probe, details = choose_valuable_probe(offers, alternatives, fallback, work,
                    observations=observations, spec=spec, queried_probes=queried, max_compute=work.remaining, simulator=simulator)
            start_compute = compute.spent
            compute.spend(work.spent)
            if probe is not None:
                begin_action(request["name"], start_compute=start_compute, start_observations=observation_cost,
                             method_id=request["method_id"])
                pending_actions[-1].update(context=request["context"], method_condition=request["condition"])
                actions.append(ActionRecord(workspace.rounds_completed, "execute_method_step",
                    "Acquire the scheduled method's evidence before continuing its fit",
                    compute_cost=work.spent, details={**request, "probe_id": probe.probe_id, "experiment": details}))
                observations.extend(acquire(probe, workspace.rounds_completed, "Execute the experiment selected across tasks", design=details))
                workspace.requested_method = {}
        if config.use_memory:
            self.library.begin_transfer_episode(spec, observations)
        retrieved = self.library.retrieve(spec, observations=observations) if config.use_memory else []
        # An allocated continuation already promises one fit and its new
        # comparison. Growing memory can use the remaining startup allocation;
        # it cannot spend that fit on evaluating more inherited candidates.
        reserved_learning = (execution_plan["requested_fit_compute"]
            + execution_plan["comparison_compute_budget"] + len(observations) + 1700
            if execution_plan else 0)
        if config.use_memory and config.use_abstractions:
            local = SearchBudget(min(2400, compute.remaining // 5,
                                    max(0, compute.remaining-reserved_learning)))
            retrieved.extend(self.library.propose(spec, observations, local, structural_transfer=config.structural_transfer,
                                                  preserve_parameters=config.transfer_scheduling))
            compute.spend(local.spent)
            actions.append(ActionRecord(0, "adapt_knowledge", "Fit structures ranked by current evidence and transfer experience",
                                        compute_cost=local.spent, details={"candidates": len(retrieved),
                                        "search": self.library.last_search}))
        bank, stop_reason = [*workspace.bank, *retrieved], "round_budget"
        if retrieved:
            retrieval = SearchBudget(max(0, compute.remaining-reserved_learning))
            tested = rank_candidates(retrieved, observations, [], fallback=fallback, q=spec.risk_quantile,
                                     budget=retrieval, complexity_weight=config.complexity_weight)
            compute.spend(retrieval.spent)
            actions.append(ActionRecord(0, "retrieve", "Measure transferred candidates before adopting a task explanation",
                                        compute_cost=retrieval.spent, details={"retrieved": [c.memory_id for c in retrieved],
                                        "allocated_compute": retrieval.limit,
                                        "reserved_learning_compute": reserved_learning,
                                        "tested": [{"memory_id": r.candidate.memory_id, "operation": r.candidate.operation,
                                                    "loss": r.evidence_loss} for r in tested], "selected": None}))
            plausible_reuse = bool(tested and tested[0].evidence_loss <= config.minimum_improvement)
            retained, retention = retain_history_frontier(tested, observations, maximum=config.max_candidates)
            bank = [*workspace.bank, *(row.candidate for row in retained)]
            if retention["history_frontier"]:
                actions.append(ActionRecord(0, "retain_history_alternatives",
                    "Keep inherited mature mechanisms that initial evidence has not distinguished", details=retention))
        else:
            plausible_reuse = False

        completed_rounds = workspace.rounds_completed
        attempted_decision_conditions = list(workspace.decision_probe_conditions)
        for local_turn in range(config.max_rounds):
            turn = workspace.rounds_completed + local_turn
            decision_experiment = {}
            # A cheap refit cannot distinguish a static rule from a response
            # feedback rule when their histories have not been varied. Prepare
            # that concrete question before fitting, then compare on two new
            # independent representative trajectories after candidates freeze.
            if spec.decision_costs is not None and config.active_probes and config.strategy != "fixed":
                historical = bool(history_branch_support(incumbent.program, observations)["required_depth"])
                alternative = next((candidate for candidate in [evidence_champion, *bank]
                    if candidate is not None and candidate.program != incumbent.program
                    and bool(history_branch_support(candidate.program, observations)["required_depth"]) != historical), None)
                if alternative is not None:
                    policies = [(incumbent, retained_decision_model(incumbent.program)),
                                (alternative, retained_decision_model(alternative.program))]
                    question_condition = decision_probe_condition(spec, policies)
                    if question_condition not in attempted_decision_conditions:
                        work = SearchBudget(min(500, max(0, learning_limit-compute.spent) // 10))
                        experiment_start = compute.spent
                        decision_experiment = plan_decision_probes(spec, policies, observations, fallback, work,
                            remaining=observation_limit-observation_cost, round_number=turn,
                            seed=config.seed, used=queried)
                        compute.limit = learning_limit
                        compute.spend(work.spent)
                        if work.spent:
                            attempted_decision_conditions.append(question_condition)
                        if decision_experiment:
                            new_rows = len(observations) + sum(probe.cost for probe in decision_experiment["targeted_probes"])
                            new_reference = sum(probe.cost for probe in decision_experiment["reference_probes"])
                            comparison_quote = (min(config.max_candidates, 8) + 3) * (
                                new_rows + len(reference) + 2 * new_reference)
                            if comparison_quote + 2 * new_rows > learning_limit - compute.spent:
                                decision_experiment = {}
                            else:
                                decision_experiment["comparison_reserve"] = comparison_quote
                        if decision_experiment:
                            old_refits = [pending for pending in pending_actions if pending["name"] == "refit"]
                            for pending in old_refits:
                                pending_actions.remove(pending)
                            old_methods = {pending.get("method_id") for pending in old_refits}
                            for action in actions:
                                if (action.round == turn and action.action == "execute_method_step"
                                        and action.details.get("name") == "refit"
                                        and action.details.get("method_id") in old_methods):
                                    action.action = "defer_method_step"
                                    action.reason = "Prepare new distinguishing evidence before testing a revised research sequence"
                            begin_action("task_value", start_compute=experiment_start,
                                         start_observations=observation_cost)
                            actions.append(ActionRecord(turn, "plan_experiment",
                                "Vary the history that separates the fitted decision explanations before refitting",
                                compute_cost=work.spent, details={**decision_experiment["design"],
                                    "observation_cost": decision_experiment["observation_cost"],
                                    "reference_probe_ids": [probe.probe_id for probe in decision_experiment["reference_probes"]],
                                    "reference_observation_cost": sum(probe.cost for probe in decision_experiment["reference_probes"]),
                                    "action_policy": {"selected_action": "task_value", "reason": "untried_history_contrast"}}))
                            experiment_earlier = list(observations)
                            acquired_preparations = []
                            for probe in decision_experiment["targeted_probes"]:
                                rows = acquire(probe, turn,
                                    "Acquire the distinguishing preparation before revising the decision explanation")
                                observations.extend(rows)
                                acquired_preparations.append((probe, rows))
                            workspace.decision_probe_conditions.append(question_condition)
                            if config.use_memory:
                                experiment_memory = SearchBudget(min(96, max(0, learning_limit-compute.spent
                                    - decision_experiment["comparison_reserve"] - 2 * len(observations))))
                                for probe, rows in acquired_preparations:
                                    memory_start = experiment_memory.spent
                                    learned = self.library.experiments.record(spec,
                                        [candidate for candidate, _ in policies], experiment_earlier, probe, rows,
                                        fallback=fallback, budget=experiment_memory, design=decision_experiment["design"])
                                    if learned:
                                        actions.append(ActionRecord(turn, "learn_experiment",
                                            "Retain a history intervention that actually separated the frozen explanations",
                                            compute_cost=experiment_memory.spent-memory_start, details=learned))
                                compute.spend(experiment_memory.spent)
                            if old_refits:
                                begin_action("refit", start_compute=compute.spent, start_observations=observation_cost)
                                pending_actions[-1]["requires_actions"] = ["task_value"]
            # Keep another fitting/comparison round affordable after this
            # round's observations; unused work carries into later rounds.
            representation_work = _representation_work(observations, spec, config.max_history)
            representation_requested = any(p["name"] == "representation" for p in pending_actions)
            continuous_work = _continuous_state_work(observations, spec)
            continuous_requested = any(p["name"] == "continuous_state" for p in pending_actions)
            from .state_search import state_search_work
            state_work = state_search_work(observations) if config.state_search else 0
            state_requested = any(p["name"] == "state_transition" for p in pending_actions)
            round_work = (learning_limit - compute.spent) // (config.max_rounds - local_turn)
            if decision_experiment:
                round_work = max(round_work, decision_experiment["comparison_reserve"] + len(observations) + 1700)
            expected_length = min(spec.max_probe_length, 4 * config.max_history + 4) if config.adaptive_horizon else 8
            if decision_experiment:
                expected_length = max(expected_length,
                    sum(probe.cost for probe in decision_experiment["reference_probes"]))
            per_comparison = len(observations) + len(reference) + 2 * expected_length
            champion_extra = bool(evidence_champion and evidence_champion.program != incumbent.program
                                  and (parent is None or evidence_champion.program != parent.program))
            # Retaining more evidence increases the cost of an executable fit.
            # A continued representation must receive a complete search and
            # comparison opportunity rather than five unusable small quotas.
            if resumed or representation_requested:
                if representation_work:
                    complete_comparison = (min(config.max_candidates, 8) + 2 + champion_extra) * per_comparison
                    round_work = max(round_work, representation_work + complete_comparison + len(observations) + 1700)
                else:
                    round_work = max(round_work, 4 * per_comparison + 1800)
            if continuous_requested:
                expected = min(spec.max_probe_length, 4 * config.max_history + 4)
                round_work = max(round_work, continuous_work +
                    (min(config.max_candidates, 8) + 3) * (len(observations) + len(reference) + 2 * expected) + 2000)
            if state_work and (resumed or state_requested):
                round_work = max(round_work, state_work +
                    (min(config.max_candidates, 8) + 2 + champion_extra) * per_comparison + len(observations) + 1700)
            compute.limit = compute.spent + min(learning_limit - compute.spent, round_work)
            if not compute.spend(len(observations)):
                stop_reason = "compute_budget"
                break
            question = diagnose(incumbent.program, observations, fallback, spec.risk_quantile,
                                max_depth=config.max_question_depth)
            graph.update(question, evidence_count=len(observations), joint=config.joint_planning,
                         depth=config.planning_depth)
            questions.append(question)
            actions.append(ActionRecord(turn, "diagnose", question.question, compute_cost=len(observations), details=asdict(question)))
            slots = min(config.max_candidates, 8, max(2, compute.remaining // max(1, 5 * per_comparison)))
            comparison_reserve = (slots + 2 + champion_extra) * per_comparison
            if decision_experiment:
                comparison_reserve = max(comparison_reserve, decision_experiment["comparison_reserve"])
            screening_allowance = min(700, compute.remaining // 7)
            startup_allowance = min(500, compute.remaining // 10)
            probe_allowance = min(500, compute.remaining // 10)
            allowance = max(0, compute.remaining - comparison_reserve - screening_allowance - startup_allowance - probe_allowance)
            proposed, plans, research_plans, executed_plans = [], [], [], []
            callable_atoms = []
            calls = list(dict.fromkeys(_calls(_opened(incumbent.program)))) if config.use_memory else []
            known_computation = bool(calls) or (config.use_memory and incumbent.program.op == "call")
            residual = question.evidence.get("squared_residual", 0.) > config.minimum_improvement
            local_repair = residual and known_computation and len(_mechanisms(_opened(incumbent.program))) > 1
            split_merge = residual and bool(calls) and bool(self.library.evolution.merges)
            merge_calls = not residual and len(calls) > 1
            recursive_work = bool(graph.ready_questions(observations))
            automatic_residual = bool(residual and known_computation and config.strategy == "adaptive" and len(observations) >= 8
                and not any(node.residual_base == incumbent.program for node in graph.nodes.values()))
            extension_quota = min(2400, allowance // 2) if (
                local_repair or split_merge or merge_calls or recursive_work or automatic_residual) else 0

            transfer_repaired = False
            if (residual and incumbent.memory_id and config.use_memory and config.use_abstractions
                    and config.strategy == "adaptive" and config.proposer != "codex"
                    and spec.decision_costs is None and spec.risk_quantile == .5):
                repair_start = compute.spent
                repair_profile = method_profile(spec, observations, incumbent, method_trace,
                    residual_loss=question.evidence["squared_residual"], baseline_loss=initial_loss,
                    minimum_improvement=config.minimum_improvement)
                local = SearchBudget(min(500, allowance))
                exhausted = (repair_profile.get("source_revision") or {}).get("status") == "parameter_repair_exhausted"
                repaired = ([] if exhausted else _refit_transferred(self.library, incumbent, spec, observations,
                                             local, config, fallback))
                compute.spend(local.spent)
                allowance -= local.spent
                proposed.extend(row.candidate for row in repaired)
                transfer_repaired = any(row.evidence_loss <= config.minimum_improvement for row in repaired)
                method_trace.append({"action": "refit_transfer", "context": action_policy.context,
                    "gain": 0., "compute_cost": local.spent, "observation_cost": 0,
                    "execution_complete": bool(repaired), "profile": repair_profile,
                    "source_id": incumbent.memory_id, "candidate_count": len(repaired),
                    "residual_resolved": transfer_repaired, "training_rows": len(observations),
                    "fit_gain": question.evidence["squared_residual"] - min(
                        (row.evidence_loss for row in repaired), default=question.evidence["squared_residual"]),
                    "evidence": "acquired_data_fit"})
                actions.append(ActionRecord(turn, "defer_source_refit" if exhausted else "refit_transfer",
                    "Measure whether current evidence repairs the transferred structure before expanding it",
                    compute_cost=local.spent, details={"memory_id": incumbent.memory_id,
                        "training_rows": len(observations), "residual_resolved": transfer_repaired,
                        "candidates": [{"expression": row.candidate.program.describe(),
                                        "loss": row.evidence_loss} for row in repaired]}))
                if transfer_repaired:
                    deferred = [pending for pending in pending_actions if pending["name"] in
                                {"refit", "representation", "continuous_state", "state_transition"}]
                    for pending in deferred:
                        pending_actions.remove(pending)
                    deferred_steps = {(pending["name"], pending.get("method_id")) for pending in deferred}
                    for action in actions:
                        if (action.round == turn and action.action == "execute_method_step"
                                and (action.details.get("name"), action.details.get("method_id")) in deferred_steps):
                            action.action = "defer_method_step"
                            action.reason = "Defer the scheduled method while comparing an already fitted repair"
                    begin_action("refit", start_compute=repair_start, start_observations=observation_cost)
                    actions.append(ActionRecord(turn, "revise_research_step",
                        "Compare the concrete parameter repair before executing a broader research method",
                        details={"selected_action": "refit", "memory_id": incumbent.memory_id,
                            "deferred": [{"action": pending["name"], "method_id": pending.get("method_id")}
                                         for pending in deferred]}))

            def select_calls(local_budget, baseline_program):
                start = local_budget.spent
                selected = self.library.select_callable_atoms(spec, observations, local_budget,
                    structural_transfer=config.structural_transfer, limit=4, baseline_program=baseline_program)
                callable_atoms.extend(selected)
                actions.append(ActionRecord(turn, "select_concepts", "Select callable concepts by their measured contribution to current residuals",
                    compute_cost=local_budget.spent - start, details=dict(self.library.last_callable_selection)))
                return selected

            select_additional_terms = select_calls if config.use_memory and config.use_abstractions else None
            if proposer is not None:
                batch = proposer.propose(observations, incumbent.program, spec, compute)
                proposed.extend(batch.candidates)
                llm_probes.extend(batch.probes)
                llm_records.append(batch.usage)
                actions.append(ActionRecord(turn, "llm_proposal", "Execute model-proposed programs under the common task interface",
                                            details={"candidates": len(batch.candidates), "usage": batch.usage}))
            if config.proposer != "codex" and not transfer_repaired:
                refit_requested = not (representation_requested or continuous_requested or state_requested) and any(p["name"] == "refit" for p in pending_actions)
                if continuous_requested:
                    local = SearchBudget(min(allowance, continuous_work))
                    generated = continuous_state_candidates(observations, budget=local,
                        seed_programs=tuple(dict.fromkeys([incumbent.program, *(candidate.program for candidate in bank)])))
                    proposed.extend(generated)
                    compute.spend(local.spent)
                    state_plan = RevisionPlan("continuous_state", "Infer a continuous state and its observed readout",
                        ("continuous_state",), steps=[{"operation": "continuous_state", "compute_cost": local.spent}],
                        next_step=1, frontier=[candidate.program for candidate in generated], status="resolved")
                    research_plans.append(state_plan)
                    plan_records.append({"round": turn, "allocated_compute": local.limit, **state_plan.to_dict()})
                    actions.append(ActionRecord(turn, "continuous_state", "Fit continuous memory from recorded trajectories",
                        compute_cost=local.spent, details={"candidates": len(generated), "allocated_compute": local.limit,
                        "training_rows": len(observations), "expressions": [candidate.program.describe() for candidate in generated]}))
                elif config.strategy == "fixed" or refit_requested or (local_turn == 0 and plausible_reuse and not representation_requested):
                    local = SearchBudget(min(allowance, 1600 if refit_requested else 500))
                    if refit_requested:
                        execution_profile = method_profile(spec, observations, incumbent, method_trace,
                            residual_loss=question.evidence["squared_residual"], baseline_loss=initial_loss,
                            minimum_improvement=config.minimum_improvement)
                        for pending in pending_actions:
                            if pending["name"] == "refit":
                                pending["execution_profile"] = execution_profile
                        from .abstractions import extract_templates, fit_template
                        templates = extract_templates(incumbent.program, "current_explanation")
                        if templates:
                            template = templates[0]
                            fitted = fit_template(template, dict(zip(template.roles, template.source_features)),
                                observations, local, risk_quantile=spec.risk_quantile)
                            proposed.extend(Candidate(f"incumbent_refit:{index}", program, "refit")
                                            for index, program in enumerate(fitted))
                    proposed.extend(synthesize(observations, operations=("refit",), budget=local,
                                               incumbent=incumbent.program, risk_quantile=spec.risk_quantile))
                    compute.spend(local.spent)
                    actions.append(ActionRecord(turn, "refit", "Compare inexpensive current-data fits with transferred knowledge", compute_cost=local.spent))
                else:
                    if config.state_search and spec.risk_quantile == 0.5:
                        from .state_search import state_candidates
                        state_plan = RevisionPlan("state_transition", "Learn a coordinate and its state conditions", ("state_search",))
                        coordinate_plan = RevisionPlan("joint_coordinates", "Learn a numerical coordinate combination", ("composition",))
                        ordinary_reserve = min(allowance, representation_work) if resumed or representation_requested else 0
                        state_quota = allocation.allocate([state_plan, coordinate_plan], allowance - ordinary_reserve,
                                                          base_weights=(0.6, 0.4))[0]
                        if state_work:
                            state_quota = max(state_quota, state_work)
                        local = SearchBudget(min(allowance, max(2200, state_work), state_quota))
                        states = state_candidates(observations, budget=local, max_history=config.max_history,
                                                  joint_coordinates=config.joint_state_coordinates)
                        proposed.extend(replace(candidate, candidate_id=f"state_transition:{candidate.candidate_id}")
                                        for candidate in states)
                        compute.spend(local.spent)
                        allowance -= local.spent
                        if local.spent:
                            state_plan.steps.append({"operation": "state_search", "compute_cost": local.spent})
                            research_plans.append(state_plan)
                            actions.append(ActionRecord(turn, "state_search", "Learn conditional transitions from observed output states",
                                                compute_cost=local.spent, details={"candidates": len(states),
                                                "allocated_compute": local.limit, "complete_basic_fit_cost": state_work,
                                                "basic_fit_funded": local.limit >= state_work,
                                                "research_experience": dict(allocation.experience)}))
                    before = compute.spent
                    local = SearchBudget(min(allowance, max(2600, representation_work)) if resumed or representation_requested
                                         else min(allowance * 4 // 5, 2600))
                    if config.risk_learning and spec.risk_quantile != 0.5:
                        from .risk_search import risk_candidates
                        generated = risk_candidates(observations, budget=local, q=spec.risk_quantile, tail_fit=config.tail_fit)
                    else:
                        generated = sparse_candidates(observations, budget=local, max_history=config.max_history,
                                                      strategy=config.joint_search, select_additional_terms=select_additional_terms)
                    proposed.extend(replace(candidate, candidate_id=f"joint_coordinates:{candidate.candidate_id}")
                                    for candidate in generated)
                    research_plans.append(RevisionPlan("joint_coordinates", "Fit executable coordinate combinations", ("composition",),
                                                      steps=[{"operation": "composition", "compute_cost": local.spent}]))
                    compute.spend(local.spent)
                    allowance -= local.spent
                    actions.append(ActionRecord(turn, "composition", "Select and combine coordinates explaining residual variation",
                                                compute_cost=compute.spent-before, details={"candidates": len(proposed),
                                                    "callable_atoms": [atom.describe() for atom in callable_atoms]}))
                    if config.joint_planning and config.strategy == "adaptive":
                        plans = graph.pending_plans()
                        quotas = allocation.allocate(plans, max(0, allowance - extension_quota))
                        actions.append(ActionRecord(turn, "allocate_research", "Allocate remaining search work using measured plan experience",
                                                    details={"context": allocation.context,
                                                             "experience": dict(allocation.experience),
                                                             "quotas": {plan.plan_id: quota for plan, quota in zip(plans, quotas)}}))
                        for plan, quota in zip(plans, quotas):
                            local = SearchBudget(min(quota, compute.remaining))
                            first_step = len(plan.steps)
                            proposed.extend(execute_revision_plan(plan, observations, incumbent=incumbent.program,
                                            budget=local, risk_quantile=spec.risk_quantile, max_history=config.max_history))
                            graph.record(plan)
                            new_steps = plan.steps[first_step:]
                            executed_plans.append(replace(plan, steps=new_steps))
                            compute.spend(local.spent)
                            plan_records.append({"round": turn, "allocated_compute": quota, **plan.to_dict()})
                            for stage in new_steps:
                                actions.append(ActionRecord(turn, stage["operation"], plan.question,
                                                            compute_cost=stage["compute_cost"], details=stage))
                    else:
                        operations = ALL_OPERATIONS if config.strategy == "search" else tuple(op for op, _ in research_steps(question))
                        local = SearchBudget(min(max(0, allowance - extension_quota), compute.remaining))
                        rows = [r for r in observations if len(r.history) >= min(config.max_history, max(map(lambda r: len(r.history), observations))) ]
                        if len(rows) < 4:
                            rows = observations
                        proposed.extend(synthesize(rows, operations=operations, budget=local,
                                                   incumbent=incumbent.program, risk_quantile=spec.risk_quantile,
                                                   max_history=config.max_history))
                        compute.spend(local.spent)
                        actions.append(ActionRecord(turn, "search", "Compare independent revisions in the shared language", compute_cost=local.spent))
            # Ordinary refits and complete current/history coordinates run first.
            # Only the remaining discretionary search work funds graph/evolution.
            evolved = []
            available = max(0, compute.remaining - comparison_reserve - screening_allowance - startup_allowance - probe_allowance)
            extra = SearchBudget(min(extension_quota, available))
            if split_merge and extra.remaining:
                start = extra.spent
                evolved.extend(self.library.evolution.reopen_merges(incumbent.program, spec, observations, extra))
                actions.append(ActionRecord(turn, "split_concepts", "Restore distinct mechanisms contradicted by new measurements",
                    compute_cost=extra.spent-start, details=dict(self.library.evolution.last_event)))
            if local_repair and not evolved and extra.remaining:
                local = SearchBudget(extra.remaining * (2 if recursive_work else 3) // 3)
                repaired = self.library.evolution.local_revisions(incumbent.program, spec, observations, local,
                    support_observations=self.library.supporting_observations(incumbent.program, spec), limit=2)
                extra.spend(local.spent)
                evolved.extend(repaired)
                actions.append(ActionRecord(turn, "repair_mechanism", "Fit the failed component while retaining successful mechanisms",
                    compute_cost=local.spent, details=dict(self.library.evolution.last_event)))
            if merge_calls and extra.remaining:
                start = extra.spent
                evolved.extend(self.library.evolution.merge_equivalent(incumbent.program, spec, observations, extra, limit=1))
                actions.append(ActionRecord(turn, "merge_concepts", "Propose executable sharing of observationally equivalent calls",
                    compute_cost=extra.spent-start, details=dict(self.library.evolution.last_event)))
            if automatic_residual and not evolved and extra.remaining >= 8 * len(observations):
                node = graph.add_residual_question(f"residual:{turn}:{len(graph.nodes)}", incumbent.program, observations)
                actions.append(ActionRecord(turn, "create_subproblem", "Preserve the current computation and investigate its remaining error",
                    details={"question_id": node.question_id, "base_program": incumbent.program.to_dict()}))
            if graph.ready_questions(observations) and extra.remaining:
                start = extra.spent
                answers = graph.execute_ready(observations, extra, incumbent=incumbent.program,
                    risk_quantile=spec.risk_quantile, max_history=config.max_history, max_nodes=4, max_candidates=6)
                dependencies = {key for node in graph.nodes.values() for key in node.depends_on}
                conclusions = [answer for answer in answers if answer.candidate_id.removeprefix("answer:") not in dependencies]
                evolved.extend(conclusions)
                actions.append(ActionRecord(turn, "execute_research_graph", "Use measured child answers in dependent executable explanations",
                    compute_cost=extra.spent-start, details={"answers": [answer.candidate_id for answer in answers],
                    "conclusions": [answer.candidate_id for answer in conclusions], "measurement_requests": graph.measurement_requests()}))
            compute.spend(extra.spent)
            equivalents = []
            simplified = []
            for candidate in [*evolved, *proposed]:
                program = candidate.program.simplified()
                if program != candidate.program:
                    equivalents.append({"candidate_id": candidate.candidate_id,
                        "before": candidate.program.describe(), "after": program.describe(),
                        "nodes_removed": candidate.program.size-program.size})
                simplified.append(replace(candidate, program=program))
            evolved = simplified[:len(evolved)]
            proposed = simplified
            if equivalents:
                actions.append(ActionRecord(turn, "simplify_equivalent", "Remove executable redundancy before comparing and retaining new explanations",
                    details={"candidates": equivalents}))
            pool = [incumbent, *([parent] if parent else []), *([evidence_champion] if evidence_champion else [])]
            for index in range(max(len(bank), len(proposed))):
                if index < len(proposed):
                    pool.append(proposed[index])
                if index < len(bank):
                    pool.append(bank[index])
            if config.use_warm_start and config.strategy != "fixed":
                local = SearchBudget(min(startup_allowance, max(0, compute.remaining - comparison_reserve)))
                from .state_search import startup_revision_candidates
                repair_seeds = list(dict.fromkeys([incumbent.program, *(candidate.program for candidate in bank[:2])]))
                repaired = startup_revision_candidates(repair_seeds, observations, budget=local)
                if local.spent:
                    actions.append(ActionRecord(turn, "repair_startup", "Revise startup predictions while retaining the learned mature mechanism",
                        compute_cost=local.spent, details={"candidates": len(repaired),
                        "expressions": [candidate.program.describe() for candidate in repaired]}))
                repair_cost = local.spent
                warm = warm_start_candidates(pool, observations, budget=local, q=spec.risk_quantile,
                                             complexity_weight=config.complexity_weight)
                compute.spend(local.spent)
                pool.extend([*repaired, *warm])
                actions.append(ActionRecord(turn, "initial_representation", "Learn predictions before the chosen historical representation is available",
                                            compute_cost=local.spent-repair_cost, details={"candidates": len(warm)}))
            before = compute.spent
            local = SearchBudget(min(screening_allowance, max(0, compute.remaining - comparison_reserve)))
            research_assessed = []
            screening = screen_candidates(pool, observations, reference, fallback=fallback, q=spec.risk_quantile,
                                          budget=local, complexity_weight=config.complexity_weight, maximum=slots,
                                          incumbent=incumbent.program, assessment_sink=research_assessed)
            compute.spend(local.spent)
            actions.append(ActionRecord(turn, "screen_candidates", "Retain promising executable alternatives for a new experiment",
                                        compute_cost=compute.spent-before, details={"candidates": len(screening)}))
            if not screening:
                stop_reason = "compute_budget"
                break
            short = [r.candidate for r in screening[:slots]]
            # A scoped shared call has a larger static fallback AST. Give the
            # real rewrite a funded comparison before judging execution savings.
            for candidate in reversed(evolved[:min(2, slots)]):
                if all(row.program != candidate.program for row in short):
                    short = [candidate, *short[:max(0, slots-1)]]
            origins_by_program = {row.candidate.program: row.origins for row in screening}
            for retained in (incumbent, parent, evidence_champion):
                if retained and all(c.program != retained.program for c in short):
                    short.append(retained)
            short.sort(key=lambda candidate: candidate.program != incumbent.program)

            design_candidates, design_predictions = short, {}
            research_design = False
            if (config.active_probes and config.strategy != 'fixed'
                    and spec.decision_costs is None and not decision_experiment):
                design_work = SearchBudget(min(probe_allowance // 2,
                    max(0, compute.remaining - comparison_reserve)))
                design_candidates, design_details = _research_comparison(
                    research_assessed, short, spec, observations, seed=config.seed, turn=turn,
                    remaining=observation_limit-observation_cost, maximum=slots,
                    fallback=fallback, budget=design_work, used=queried,
                    prediction_sink=design_predictions)
                research_design = True
                compute.spend(design_work.spent)
                probe_allowance -= design_work.spent
                actions.append(ActionRecord(turn, 'research_comparison_pool',
                    'Retain evidence, complexity and public prediction alternatives for experiment design',
                    compute_cost=design_work.spent, details=design_details))
            representative = (decision_experiment["reference_probes"][0] if decision_experiment else representative_probe(spec, seed=config.seed, index=turn,
                            remaining=observation_limit-observation_cost, used=queried,
                            candidates=design_candidates, adaptive_horizon=config.adaptive_horizon))
            # An acquired experiment must have a funded comparison. Candidate
            # generation and query selection cannot borrow this allocation.
            reference_cost = (sum(probe.cost for probe in decision_experiment["reference_probes"])
                              if decision_experiment else representative.cost if representative else 0)
            per_comparison = len(observations) + len(reference) + 2 * reference_cost
            if compute.remaining < per_comparison * len(short):
                stop_reason = "compute_budget"
                continue
            fresh_reference = []
            channel_candidate = screening[0].candidate
            channel_residual = screening[0].evidence_loss
            for probe in (decision_experiment["reference_probes"] if decision_experiment else [representative]):
                forecast = None
                channel_work = SearchBudget(min(probe_allowance,
                    max(0, compute.remaining-per_comparison*len(short))))
                forecast_cost = (probe.cost * (1 + max(1, 2 * (len(simulator.candidates) + bool(simulator.current))))
                                 if probe else 0)
                if (probe and channel_residual is not None and channel_residual > config.minimum_improvement
                        and channel_work.remaining >= forecast_cost):
                    channel_work.spend(probe.cost)
                    forecast = simulator.forecast(probe, _trajectory_predictions(channel_candidate, probe, fallback, observations),
                        baseline_variance=channel_residual, budget=channel_work)
                    compute.spend(channel_work.spent)
                    probe_allowance -= channel_work.spent
                    actions.append(ActionRecord(turn, "forecast_experiment_observations",
                        "Freeze observation distributions before the representative experiment",
                        compute_cost=channel_work.spent, details={"probe_id": probe.probe_id, "forecast": forecast}))
                acquired = acquire(probe, turn, "Measure the frozen alternatives on a new representative control trajectory")
                fresh_reference.extend(acquired)
                if forecast and acquired:
                    simulator.observe(forecast, acquired, budget=SearchBudget(0))
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
            current = next((r for r in assessed if r.candidate.program == incumbent.program), None)
            if current is None:
                observations.extend(fresh)
                reference.extend(fresh_reference)
                stop_reason = "compute_budget"
                break
            comparison = compare_candidate_adoption(assessed, incumbent, observations,
                reference, fresh_reference, fallback=fallback, spec=spec, config=config)
            best, current = comparison.best, comparison.current
            supported_candidates, adopt = comparison.supported_candidates, comparison.adopt
            margin, fresh_gain, evidence_gain = comparison.margin, comparison.fresh_gain, comparison.evidence_gain
            support_unit, counterexample_gain = comparison.support_unit, comparison.counterexample_gain
            known_response_gain = comparison.known_response_gain
            cumulative_prediction_support = comparison.cumulative_prediction_support
            execution_gain = None
            if not adopt and spec.decision_costs is None:
                for merged in (row for row in assessed if row.candidate.operation == "concept_merge"):
                    if any(new > old + 1e-8 for new, old in ((merged.evidence_loss, current.evidence_loss),
                            (merged.reference_loss, current.reference_loss), (merged.fresh_loss, current.fresh_loss))):
                        continue
                    rows = [*observations, *reference, *fresh_reference]
                    if not compute.spend(2 * len(rows)):
                        break
                    costs = []
                    for program in (incumbent.program, merged.candidate.program):
                        steps = 0
                        for row in rows:
                            if len(row.history) >= program.minimum_history_depth:
                                context = EvaluationContext(row)
                                program.predict(row, context)
                                steps += context.primitive_steps + context.call_evaluations
                        costs.append(steps)
                    if costs[1] < costs[0]:
                        execution_gain = {"before": costs[0], "after": costs[1], "rows": len(rows)}
                        best, adopt = merged, True
                        fresh_gain = current.fresh_loss - merged.fresh_loss
                        for record in self.library.evolution.merges:
                            if record["merge_id"] == merged.candidate.candidate_id:
                                record["status"] = "adopted"
                        break
            actions.append(ActionRecord(turn, "compare", "Compare task-representative evidence and prospective predictions",
                                        compute_cost=compute.spent-before, details={"candidates_scored": len(assessed),
                                        "best_expression": best.candidate.program.describe(), "best_loss": best.reference_loss,
                                        "fresh_gain": fresh_gain, "evidence_gain": evidence_gain, "dispersion_margin": margin, "adopt": adopt,
                                        "support_unit": support_unit,
                                        "counterexample_gain": counterexample_gain,
                                        "known_response_gain": known_response_gain,
                                        "cumulative_prediction_support": cumulative_prediction_support,
                                        "execution_gain": execution_gain,
                                        "selection_policy": config.selection_policy,
                                        "decision_evidence": ("observed_counterexample_with_nonworse_reference" if counterexample_gain is not None
                                                              else "accumulated_prediction_with_nonworse_new_trajectory" if cumulative_prediction_support
                                                              else "accumulated_risk_trajectories" if cumulative_risk else "new_trajectory"),
                                        "fresh_weight": fresh_weight, "validation_is_adaptive_development": True}))
            if (executed_plans or research_plans) and current.fresh_loss is not None:
                allocation.update([*research_plans, *executed_plans], supported_candidates, current.fresh_loss)
                actions.append(ActionRecord(turn, "research_feedback", "Update research tactics from new predictions and actual search cost",
                                            details={"context": allocation.context,
                                                     "experience": dict(allocation.experience)}))
            previous_program = incumbent.program
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
            for event in self.library.evolution.record_comparison(
                    previous_program, incumbent, assessed, spec, len(fresh_reference)):
                actions.append(ActionRecord(turn, event["operation"],
                    "Retain the concept comparison and its actual selected explanation", details=event))
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
                            loss=measured.fresh_loss, baseline_loss=current.fresh_loss,
                            evidence_rows=len(fresh_reference), comparison_id=turn,
                            comparison_compute=len(observations) + len(fresh_reference)
                                + (len(reference) + len(fresh_reference) if prospective else 0),
                            observations=observations)
                        actions.append(ActionRecord(turn, "reuse_check", "Measure transferred knowledge on a new representative trajectory",
                                                    details={"loss": measured.fresh_loss, "memory_id": candidate.memory_id,
                                                    "expression": candidate.program.describe(), "accepted": accepted,
                                                    "prediction_shared_by_origins": len(measured.origins)}))
            chosen = next(r for r in assessed if r.candidate.program == incumbent.program)
            evidence_champion = min(assessed, key=lambda row: (
                .8 * row.reference_loss + .2 * row.evidence_loss + config.complexity_weight * row.candidate.program.size,
                row.candidate.program.size)).candidate
            credit_actions(fresh_reference, turn)
            before = compute.spent
            # Keep the next representative comparison affordable in observations
            # too. A final-round intervention cannot revise a deterministic model.
            active_remaining = max(0, observation_limit-observation_cost - len(fresh_reference))
            if spec.decision_costs is None and local_turn == config.max_rounds-1:
                active_remaining = 0
            # Task ranking/adoption above keeps its original shortlist. Only
            # fully checked paid-evidence alternatives may design interventions.
            active_candidates, active_assessed, reviewed_research_pool = short, assessed, None
            task_tail_reserve = len(short) * (2 * len(observations) + len(fresh_reference))
            if research_design and active_remaining:
                review_work = SearchBudget(min(probe_allowance,
                    max(0, compute.remaining-task_tail_reserve)))
                active_candidates, reviews, review_details = _refresh_active_research_pool(
                    assessed, design_candidates, design_predictions, observations,
                    [*reference, *fresh_reference] if prospective else [], fresh_reference,
                    budget=review_work, maximum=min(4, slots), fallback=fallback,
                    q=spec.risk_quantile, complexity_weight=config.complexity_weight,
                    minimum_improvement=config.minimum_improvement,
                    fresh_weight=fresh_weight if prospective else 0.)
                compute.spend(review_work.spent)
                probe_allowance -= review_work.spent
                active_assessed = list(reviews.values())
                reviewed_research_pool = active_candidates
                actions.append(ActionRecord(turn, 'review_active_research_pool',
                    'Check research alternatives against the actual paid comparison before planning interventions',
                    compute_cost=review_work.spent, details=review_details))
            before = compute.spent
            transition_work = SearchBudget(min(probe_allowance // 2,
                max(0, compute.remaining-task_tail_reserve) if research_design else compute.remaining))
            transition, transition_details = None, {}
            probe_evidence = [*observations, *fresh_reference]
            probe_losses = {row.candidate.program:
                (row.evidence_loss * len(observations) + row.fresh_loss * len(fresh_reference))
                / len(probe_evidence) for row in active_assessed}
            if config.transition_probes and config.active_probes and config.strategy != "fixed":
                transition, transition_details = transition_probe(spec,
                    active_candidates if research_design else [incumbent, *short[:3]], probe_evidence,
                    budget=transition_work, remaining=active_remaining, round_number=turn, used=queried,
                    joint_controls=config.joint_state_probes,
                    evidence_losses=probe_losses, followup_observation_cost=len(fresh_reference),
                    remaining_rounds=config.max_rounds-local_turn-1)
            compute.spend(transition_work.spent)
            fresh_active = []
            reuse_confirmed = bool(config.transfer_scheduling and incumbent.memory_id and transition is None
                and spec.decision_costs is None and fresh_reference
                and chosen.reference_loss <= config.minimum_improvement
                and chosen.evidence_loss <= config.minimum_improvement
                and chosen.fresh_loss <= config.minimum_improvement)
            if reuse_confirmed and not workspace.requested_method and research_selection is None:
                actions.append(ActionRecord(turn, "plan_experiment", "Transferred knowledge explains the existing and newly acquired trajectory",
                                            compute_cost=compute.spent-before,
                                            details={"policy": "supported_transfer_stop", "memory_id": incumbent.memory_id}))
            else:
                recalled_probes = []
                if config.use_memory and config.active_probes and config.strategy != "fixed":
                    recall_work = SearchBudget(min(160, max(0, probe_allowance-transition_work.spent),
                        max(0, compute.remaining-task_tail_reserve) if research_design else compute.remaining))
                    recalled_probes = self.library.experiments.propose(spec, active_candidates, probe_evidence,
                        fallback=fallback, budget=recall_work, remaining=active_remaining, used=queried,
                        evidence_losses=probe_losses, boundary_proposal=(transition, transition_details))
                    compute.spend(recall_work.spent)
                    probe_allowance -= recall_work.spent
                    if recalled_probes:
                        actions.append(ActionRecord(turn, "recall_experiment", "Rebind previously useful interventions and test their current distinguishing power",
                            compute_cost=recall_work.spent, details={"experiments": self.library.experiments.last_recall}))
                affordable = [p for p in [*recalled_probes, *spec.available_probes, *llm_probes,
                              *(construct_probes(spec, active_candidates, round_number=turn, seed=config.seed, max_history=config.max_history)
                                if config.synthesize_probes else [])]
                              if p.cost <= active_remaining and all(p.sequence != old.sequence for old in queried)]
                extra = (representative_probe(spec, seed=config.seed, index=turn + 1000,
                    remaining=active_remaining, used=queried, candidates=[incumbent],
                    adaptive_horizon=config.adaptive_horizon)
                    if config.active_probes and config.strategy != 'fixed' else None)
                experiment_details, active, question_selection = {}, None, None
                intervention_check_reserve = (len(active_candidates) * max((probe.cost for probe in
                    [*affordable, *([transition] if transition else []), *([extra] if extra else [])]), default=0)
                    if research_design else 0)
                local = SearchBudget(max(0, min(probe_allowance-transition_work.spent-intervention_check_reserve,
                    compute.remaining-task_tail_reserve-intervention_check_reserve
                    if research_design else compute.remaining)))
                if affordable and config.active_probes and config.strategy != "fixed":
                    if research_selection is not None:
                        active, question_selection = choose_question_probe(
                            affordable, active_candidates, fallback, local, spec=spec,
                            selection_contract=research_selection, observation_budget=active_remaining,
                            public_prediction_rows=getattr(environment, 'prediction_rows', None))
                        experiment_details = {'policy': 'pr2_question_selection'}
                    elif config.experiment_policy == "task_value":
                        active, experiment_details = choose_valuable_probe(affordable, active_candidates, fallback, local,
                                     observations=[*observations, *fresh_reference], spec=spec, queried_probes=queried,
                                     max_compute=local.remaining, simulator=simulator,
                                     reviewed_research_pool=reviewed_research_pool)
                    else:
                        active, value = choose_probe(affordable, active_candidates, fallback, local,
                            observations=probe_evidence if research_design else observations)
                        experiment_details = {"disagreement": value, "policy": "disagreement"}
                elif affordable and research_selection is None:
                    active = rng.choice(affordable)
                    experiment_details = {"policy": "random"}
                if active is not None:
                    recalled = next((item for item in self.library.experiments.last_recall
                                     if item["probe_id"] == active.probe_id), None)
                    if recalled and recalled.get("design"):
                        experiment_details = {**experiment_details, **recalled["design"]}
                compute.spend(local.spent)
                if config.active_probes and config.strategy != "fixed":
                    # These are quotes, not debits. The original action policy
                    # must leave the selected intervention's actual checks funded.
                    check_units = len(active_candidates)+len(short)+1 if research_design else 1
                    action_compute_remaining = (max(0, compute.remaining-task_tail_reserve)
                        if research_design else max(0, learning_limit-compute.spent))
                    transferred_conflict = any(r.candidate.memory_id and r.evidence_loss <= config.minimum_improvement
                        and r.fresh_loss is not None and r.fresh_loss > config.minimum_improvement for r in assessed)
                    condition = ("transfer_counterexample" if transferred_conflict else
                                 "model_residual" if chosen.evidence_loss > config.minimum_improvement else
                                 "new_counterexample" if chosen.fresh_loss > config.minimum_improvement else "adequate")
                    action_policy.set_condition(condition)
                    options = []
                    if transition is not None and research_selection is None:
                        options.append(ActionOption("transition", transition.cost, transition.cost*check_units,
                                                    (transition, transition_details)))
                    if active is not None:
                        options.append(ActionOption("task_value", active.cost, active.cost*check_units, (active, experiment_details)))
                    if extra is not None and research_selection is None:
                        options.append(ActionOption("representative", extra.cost, extra.cost*check_units, (extra, {})))
                    if local_turn < config.max_rounds - 1 and max(chosen.evidence_loss, chosen.fresh_loss) > config.minimum_improvement:
                        options.append(ActionOption("refit", 0, 2 * len(observations)))
                        if representation_work:
                            options.append(ActionOption("representation", 0, representation_work))
                        if state_work:
                            options.append(ActionOption("state_transition", 0, state_search_work([*observations, *fresh_reference])))
                        next_rows = [*observations, *fresh_reference]
                        next_continuous_work = _continuous_state_work(next_rows, spec)
                        next_comparison = (min(config.max_candidates, 8) + 3) * (
                            len(next_rows) + len(reference) + len(fresh_reference) + 2 * expected_length) + 2000
                        if (next_continuous_work and config.proposer != "codex"
                                and observation_limit - observation_cost >= min(8, spec.max_probe_length)
                                and next_continuous_work + next_comparison <= learning_limit - compute.spent):
                            options.append(ActionOption("continuous_state", 0, next_continuous_work))
                    if research_design:
                        available_checks = max(0, probe_allowance-transition_work.spent-local.spent)
                        unfunded = [option for option in options if option.name in
                            {'transition', 'task_value', 'representative'}
                            and option.observation_cost*len(active_candidates) > available_checks]
                        options = [option for option in options if option not in unfunded]
                        if unfunded:
                            actions.append(ActionRecord(turn, 'defer_research_intervention',
                                'Leave interventions with unfunded research prediction checks unexecuted',
                                details={'status': 'UNKNOWN', 'available_compute': available_checks,
                                    'plans': [{'action': option.name,
                                        'required_compute': option.observation_cost*len(active_candidates)}
                                        for option in unfunded]}))
                    selected_method = None
                    selection_profile = method_profile(spec, observations, incumbent, method_trace,
                        residual_loss=chosen.evidence_loss, reference_loss=chosen.fresh_loss,
                        baseline_loss=initial_loss, minimum_improvement=config.minimum_improvement,
                        boundary_evidence=transition_details)
                    options = [option for option in options if method_applicability(None, selection_profile, option.name)["can_try"]]
                    ordinary = action_policy.choose(options, observation_remaining=active_remaining,
                        compute_remaining=action_compute_remaining,
                        minimum_improvement=config.minimum_improvement)
                    request = workspace.requested_method
                    if request:
                        selected_method = next((option for option in options if option.name == request["name"]
                            and option.observation_cost <= active_remaining
                            and option.compute_cost <= action_compute_remaining), None)
                        if selected_method:
                            self.library.methods.last_choice = dict(request)
                            workspace.requested_method = {}
                    if config.use_memory and config.learn_research_policy:
                        selected_method = selected_method or self.library.methods.choose(action_policy.context, research_context(spec, observations),
                            options, method_trace, task_id=spec.task_id, observation_remaining=active_remaining,
                            compute_remaining=action_compute_remaining,
                            profile=selection_profile, ordinary=ordinary,
                            ordinary_evidence=action_policy.last_decision)
                    selected = selected_method or ordinary
                    revision_options = [option for option in options if option.observation_cost <= active_remaining
                        and option.compute_cost <= action_compute_remaining]
                    method_revision = self.library.methods.revise_plan(selection_profile, revision_options, trace=method_trace)
                    if selected and method_revision and selected.name in method_revision["excluded_actions"]:
                        successor = next(option for option in options if option.name == method_revision["action"])
                        selected_method, selected = None, successor
                        action_policy.last_decision = {"selected_action": successor.name,
                            "reason": method_revision["reason"], "method_revision": method_revision}
                    if method_revision:
                        actions.append(ActionRecord(turn, "revise_research_method",
                            method_revision["reason"], details={**method_revision,
                                "selected_action": selected.name if selected else None}))
                    if selected_method:
                        action_policy.last_decision = {"context": action_policy.context,
                            "selected_action": selected.name, "reason": "learned_research_sequence",
                            "eligible_actions": [option.name for option in options],
                            "quoted_observation_cost": selected.observation_cost,
                            "quoted_compute_cost": selected.compute_cost,
                            "observation_remaining": active_remaining,
                            "compute_remaining": action_compute_remaining}
                    active, experiment_details = selected.payload if selected and selected.payload else (None, {})
                    experiment_details = {**experiment_details, "action_policy": dict(action_policy.last_decision)}
                    if selected_method:
                        experiment_details["learned_method"] = dict(self.library.methods.last_choice)
                    if selected:
                        begin_action(selected.name, start_compute=before, start_observations=observation_cost,
                                     method_id=self.library.methods.last_choice["method_id"] if selected_method else None,
                                     profile=selection_profile)
                        preparation = next((step for step in reversed(method_trace) if step["action"] in
                            {"transition", "representative", "task_value"} and step.get("execution_complete", True)), None)
                        if preparation and selected.name in {"representation", "continuous_state", "state_transition"}:
                            previous_data = (preparation.get("profile") or {}).get("data", {})
                            current_data = selection_profile["data"]
                            if (min(config.max_history, current_data["max_history"]) >
                                    min(config.max_history, previous_data.get("max_history", 0))
                                    or current_data["coverage"] != previous_data.get("coverage")):
                                pending_actions[-1]["requires_actions"] = [preparation["action"]]
                if research_selection is not None:
                    experiment_details['policy'] = 'pr2_question_selection'
                    experiment_details['question_selection'] = question_selection or {
                        'status': 'scoped_unknown', 'reason': 'no_affordable_native_intervention',
                        'compute_cost': 0, 'evidence': [], 'support_updated': False}
                    planned = experiment_details['question_selection'].get('selected_experiment')
                    experiment_details['question_selection']['native_execution'] = {
                        'status': ('prepared' if active is not None else 'prepared_not_executed')
                            if planned else 'not_prepared',
                        'actual_action': action_policy.last_decision.get('selected_action'),
                        'observation_cost': 0}
                actions.append(ActionRecord(turn, "plan_experiment", "Choose additional observations for task value and unresolved structure",
                                            compute_cost=compute.spent-before, details=experiment_details))
                # Freeze the observation-channel forecast only from the unused
                # experiment-design allocation. Core fitting/report work keeps
                # its existing reservation, including on small continuations.
                simulation_forecast = None
                simulation_before = local.spent
                simulation_work = (SearchBudget(min(local.remaining,
                    max(0, compute.remaining-task_tail_reserve-active.cost*check_units)))
                    if research_design and active is not None else local)
                if active is not None and simulation_work.remaining >= active.cost * (1 + max(1, 2 * (len(simulator.candidates) + bool(simulator.current)))):
                    simulation_work.spend(active.cost)
                    simulation_forecast = simulator.forecast(active,
                        _trajectory_predictions(incumbent, active, fallback, observations),
                        baseline_variance=max(chosen.evidence_loss, 1e-8), budget=simulation_work)
                    if simulation_work is not local:
                        local.spend(simulation_work.spent)
                    compute.spend(local.spent-simulation_before)
                    actions.append(ActionRecord(turn, "forecast_experiment_observations",
                        "Predict the real trajectory with the current and proposed observation models",
                        compute_cost=local.spent-simulation_before,
                        details={"probe_id": active.probe_id, "forecast": simulation_forecast}))
                fresh_active = acquire(active, turn, "Distinguish remaining explanations with an affordable intervention", design=experiment_details)
                if research_selection is not None and fresh_active:
                    experiment_details['question_selection']['native_execution'].update(
                        status='executed', probe_id=active.probe_id, observation_cost=active.cost,
                        evidence_ref={'field': 'observations', 'row_indices': list(range(
                            len(observations)+len(fresh_reference), len(observations)+len(fresh_reference)+len(fresh_active)))})
                if research_design and fresh_active:
                    check_work = SearchBudget(len(active_candidates)*len(fresh_active))
                    checks = _check_active_research_predictions(active_candidates, fresh_active,
                        observation_offset=len(observations)+len(fresh_reference), budget=check_work,
                        fallback=fallback, q=spec.risk_quantile,
                        minimum_improvement=config.minimum_improvement)
                    compute.spend(check_work.spent)
                    actions.append(ActionRecord(turn, 'check_active_research_predictions',
                        'Retain actual intervention losses and scoped research-pool failures',
                        compute_cost=check_work.spent, details=checks))
                if simulation_forecast and fresh_active:
                    simulator.observe(simulation_forecast, fresh_active, budget=SearchBudget(0))
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
            if config.use_memory:
                # The ordinary candidate comparison and grouping retain priority;
                # spare work turns a useful intervention into future knowledge.
                memory_work = SearchBudget(min(96, max(0, compute.remaining -
                    len(short) * (len(observations) + len(fresh)) - len(observations) - len(fresh))))
                learned_experiment = self.library.experiments.record(spec,
                    active_candidates if research_design and fresh_active else short, observations,
                    active if fresh_active else representative,
                    fresh_active or (fresh_reference[:representative.cost] if decision_experiment else fresh_reference),
                    fallback=fallback, budget=memory_work,
                    design=experiment_details if fresh_active else None)
                compute.spend(memory_work.spent)
                if fresh_active and active:
                    self.library.experiments.feedback(active, supported=learned_experiment is not None)
                if learned_experiment:
                    actions.append(ActionRecord(turn, "learn_experiment", "Retain an acquired experiment that separated competing explanations",
                        compute_cost=memory_work.spent, details=learned_experiment))
            observations.extend(fresh)
            reference.extend(fresh_reference)
            after_groups = observed_classes(short, observations, fallback, compute)
            actions.append(ActionRecord(turn, "update_observable_classes", "Revise empirical groups while retaining distinct programs",
                                        details={"before": before_groups, "after": after_groups,
                                        "before_complete": sum(map(len, before_groups)) == len(short),
                                        "after_complete": sum(map(len, after_groups)) == len(short), "programs_retained": len(short)}))
            retained, retention = retain_history_frontier(assessed, probe_evidence, maximum=config.max_candidates)
            bank = [r.candidate for r in retained
                    if not (config.transfer_scheduling and r.candidate.memory_id
                            and r.evidence_loss > chosen.evidence_loss + config.minimum_improvement
                            and r.fresh_loss is not None and chosen.fresh_loss is not None
                            and r.fresh_loss > chosen.fresh_loss + config.minimum_improvement)]
            if retention["history_frontier"]:
                actions.append(ActionRecord(turn, "retain_history_alternatives",
                    "Carry still indistinguishable historical mechanisms into subsequent research", details=retention))
            completed_rounds = turn + 1
            transition_unresolved = bool(transition and (
                transition_details.get("relative_width", 0.) > 2 * config.minimum_improvement
                or transition_details.get("candidate_disagreement", 0.) > config.minimum_improvement))
            if reuse_confirmed or (not transition_unresolved and targeted_support and chosen.reference_loss <= config.minimum_improvement and chosen.evidence_loss <= config.minimum_improvement
                    and (turn > 0 or incumbent.program.size <= 3)):
                stop_reason = "knowledge_reused" if incumbent.memory_id else "adequate_and_compact"
                break

        compute.limit = config.compute_budget
        train, validation = _split(observations, config.validation_fraction)
        decision_previous_candidate = (visit_candidate if resumed and visit_uncertainty else incumbent)
        decision_previous_model = (dict(visit_uncertainty) if resumed and visit_uncertainty
                                   else {"kind": "nominal"})
        decision_model = {}
        proposed_decision_model = None
        decision_evidence = None
        calibration_choice = None
        calibration_followup = []
        calibration_compute_start, calibration_observation_start = compute.spent, observation_cost
        if spec.decision_costs is not None:
            # Freeze the predictor before residual fitting. The last two trials
            # are acquired only after both complete decision plans are frozen.
            proposed_candidate = incumbent
            depth = max(decision_previous_candidate.program.history_depth, incumbent.program.history_depth)
            risk_quote = risk_calibration_quote(training_rows=len(train), validation_rows=len(validation),
                training_trajectories=sum(not row.history for row in train),
                feature_count=len(spec.feature_names), history_depth=depth,
                max_probe_length=spec.max_probe_length, tail_fit=config.tail_fit)
            compare_complete = bool(config.risk_learning or (resumed and visit_uncertainty))
            reporting = 2 * len(reference or validation)
            if calibration_reserve:
                action_policy.set_condition("decision_calibration")
                calibration_choice = action_policy.choose(
                    [ActionOption("calibration", 0, risk_quote["fit_compute_cost"])],
                    observation_remaining=config.observation_budget-observation_cost,
                    compute_remaining=max(0, compute.remaining-reporting-risk_quote["comparison_compute_cost"]))
                actions.append(ActionRecord(completed_rounds, "plan_calibration",
                    "Price residual fitting while preserving fresh complete-decision comparison",
                    details={**action_policy.last_decision, "risk_quote": risk_quote}))
            comparison_reserve = 0
            if compare_complete:
                comparison_reserve = risk_quote["comparison_compute_cost"]
                if config.observation_budget-observation_cost >= (risk_quote["comparison_observation_cost"]
                        + risk_quote["followup_observation_cost"]):
                    comparison_reserve += risk_quote["followup_compute_cost"]
            feedback_reserve = 3 * risk_quote["comparison_observation_cost"] if calibration_choice else 0
            fitting = SearchBudget(max(0, compute.remaining-reporting-comparison_reserve-feedback_reserve))
            calibration_before = dict(decision_previous_model)
            if calibration_choice or not config.risk_learning:
                decision_model = fit_decision_model(incumbent.program, train, validation, fallback=fallback,
                    costs=spec.decision_costs, budget=fitting, adaptive=config.risk_learning, tail_fit=config.tail_fit)
            elif resumed and incumbent.program == visit_incumbent:
                decision_model = dict(decision_previous_model)
            else:
                decision_model = {"kind": "nominal"}
            fit_cost = fitting.spent
            compute.spend(fit_cost)
            new_decision = bool(calibration_choice or incumbent.program != decision_previous_candidate.program)
            if compare_complete and not new_decision and resumed:
                actions.append(ActionRecord(completed_rounds, "compare_calibration",
                    "Retain the current decision because recalibration was not selected",
                    details={"selected": "previous", "adopted": False, "rows": 0,
                        "trajectories": 0, "compute_cost": 0, "reason": "calibration_not_selected"}))
            if compare_complete and new_decision:
                proposed_decision_model = dict(decision_model)
                observation_limit = config.observation_budget
                # Buying a partial pair cannot support adoption. Keep the fitted
                # alternative pending when its complete first comparison cannot
                # be afforded, instead of deploying it from its fitting loss.
                if (observation_limit-observation_cost >= risk_quote["comparison_observation_cost"]
                        and compute.remaining >= reporting+risk_quote["comparison_compute_cost"]):
                    followup_probes = []
                    for index in range(2):
                        probe = representative_probe(spec, seed=config.seed,
                            index=completed_rounds + 100 + index, remaining=risk_quote["trajectory_length"],
                            used=[*queried, *followup_probes],
                            candidates=[decision_previous_candidate, proposed_candidate], adaptive_horizon=True)
                        if probe is not None:
                            followup_probes.append(probe)
                    if len(followup_probes) == 2:
                        calibration_followup = [row for probe in followup_probes for row in acquire(probe,
                            completed_rounds, "Compare complete decision plans on subsequent trajectories")]
                        observations.extend(calibration_followup)
                selected_program, decision_model, comparison = select_decision_plan(
                    decision_previous_candidate.program, decision_previous_model,
                    proposed_candidate.program, proposed_decision_model,
                    calibration_followup, fallback=fallback, costs=spec.decision_costs,
                    budget=compute, minimum_improvement=config.minimum_improvement)
                decision_evidence = begin_decision_evidence(
                    decision_previous_candidate.program, decision_previous_model,
                    proposed_candidate.program, proposed_decision_model, fallback=fallback,
                    costs=spec.decision_costs, minimum_improvement=config.minimum_improvement,
                    comparison=comparison)
                changed_predictor = proposed_candidate.program != decision_previous_candidate.program
                action = ("compare_decision_plan" if changed_predictor else "compare_calibration") if resumed else "compare_initial_calibration"
                actions.append(ActionRecord(completed_rounds, action,
                    "Compare the frozen proposed decision with the previous deployment or initial nominal action",
                    compute_cost=comparison["compute_cost"], details={**comparison,
                        "previous_program": decision_previous_candidate.program.to_dict(),
                        "previous_model": decision_previous_model,
                        "candidate_program": proposed_candidate.program.to_dict(),
                        "candidate_model": proposed_decision_model}))
                if decision_evidence["status"] == "continue" and comparison["trajectories"] >= 2:
                    quote = decision_evidence_quote(decision_evidence,
                        trajectory_length=risk_quote["trajectory_length"],
                        observation_remaining=config.observation_budget-observation_cost,
                        compute_remaining=max(0, compute.remaining-reporting))
                    actions.append(ActionRecord(completed_rounds, "plan_decision_followup",
                        "Price another pair of trajectories for the still conflicting frozen decisions",
                        details={**quote, "gain": comparison["gain"],
                                 "support_margin": comparison["support_margin"]}))
                    if quote["payable"]:
                        extra_probes = []
                        for index, length in enumerate(quote["probe_lengths"]):
                            probe = representative_probe(spec, seed=config.seed,
                                index=completed_rounds + 200 + index, remaining=length,
                                used=[*queried, *extra_probes],
                                candidates=[decision_previous_candidate, proposed_candidate], adaptive_horizon=True)
                            if probe is not None:
                                extra_probes.append(probe)
                        if len(extra_probes) == 2:
                            extra_rows = [row for probe in extra_probes for row in acquire(probe,
                                completed_rounds, "Resolve conflicting complete decisions with additional representative evidence")]
                            observations.extend(extra_rows)
                            calibration_followup.extend(extra_rows)
                            decision_evidence = extend_decision_evidence(decision_evidence, extra_rows, budget=compute)
                            comparison = decision_evidence["comparison"]
                            actions.append(ActionRecord(completed_rounds, "extend_decision_comparison",
                                "Combine the newly measured trajectories with the already paid frozen comparison",
                                compute_cost=decision_evidence["last_compute_cost"], details=comparison))
                            if comparison["adopted"]:
                                selected_program, decision_model = proposed_candidate.program, proposed_decision_model
                if selected_program == decision_previous_candidate.program and changed_predictor:
                    incumbent, parent = decision_previous_candidate, proposed_candidate
                    bank = list(dict.fromkeys([proposed_candidate, *bank]))[:config.max_candidates]
                    actions.append(ActionRecord(completed_rounds, "retain_decision_plan",
                        "Keep the previous executable decision when the proposed plan lacks new support",
                        details={"rejected_program": proposed_candidate.program.to_dict(),
                                 "retained_program": decision_previous_candidate.program.to_dict(),
                                 "reason": comparison.get("reason"), "gain": comparison.get("gain")}))
                    proposed_decision_model = None
                elif decision_evidence["status"] != "continue":
                    proposed_decision_model = None
            actions.append(ActionRecord(config.max_rounds, "revise_uncertainty", "Learn residual uncertainty under unchanged task costs",
                                        compute_cost=fit_cost, details=decision_model))
            if (calibration_choice and validation and decision_model.get("status") != "insufficient_budget"
                    and not (decision_evidence and decision_evidence["status"] == "continue")):
                begin_action("calibration", start_compute=calibration_compute_start,
                             start_observations=calibration_observation_start, before_model=calibration_before,
                             after_model=decision_model)
                credit_actions(calibration_followup, completed_rounds, names={"calibration"})
        measured = compute.spend(2 * len(reference or validation))
        final_rows = reference or validation
        final_loss = loss(incumbent.program, final_rows, fallback, spec.risk_quantile) if measured else None
        baseline_loss = loss(initial.program, final_rows, fallback, spec.risk_quantile) if measured else None
        if resumed and decision_evidence and decision_evidence["comparison"]["trajectories"] >= 2:
            compared = decision_evidence["comparison"]
            before_loss = compared["before_loss"]
            after_loss = compared["after_loss"] if compared["adopted"] else before_loss
            actions.append(ActionRecord(completed_rounds, "continuation_feedback",
                "Use the paid complete-decision comparison as the actual continuation return",
                details={"before_loss": before_loss, "after_loss": after_loss,
                    "gain": before_loss-after_loss, "normalized_gain": (before_loss-after_loss)/max(initial_loss, 1e-8),
                    "reference_rows": compared["rows"], "program_changed": visit_incumbent != incumbent.program,
                    "execution_complete": decision_evidence["status"] != "continue",
                    "observation_cost": observation_cost, "compute_cost": compute.spent,
                    "evidence": "accumulated_frozen_complete_decisions"}))
        elif resumed and decision_evidence and decision_evidence["status"] == "continue":
            actions.append(ActionRecord(completed_rounds, "continuation_feedback",
                "Keep the unfunded complete-decision comparison open for a subsequent visit",
                details={"before_loss": None, "after_loss": None, "gain": None,
                    "normalized_gain": None, "reference_rows": 0,
                    "program_changed": visit_incumbent != incumbent.program,
                    "execution_complete": False, "observation_cost": observation_cost,
                    "compute_cost": compute.spent, "evidence": "pending_frozen_complete_decisions"}))
        elif resumed:
            # Compare both saved and final programs on the same newly acquired
            # trajectories. A changing validation denominator is not a return.
            new_reference = reference[visit_reference_start:]
            if new_reference and compute.spend(2 * len(new_reference)):
                before_loss = objective_loss(visit_incumbent, new_reference, visit_uncertainty)
                after_loss = objective_loss(incumbent.program, new_reference, decision_model)
                actions.append(ActionRecord(completed_rounds, "continuation_feedback",
                    "Measure the saved and continued explanations on the same new observations",
                    compute_cost=2 * len(new_reference), details={"before_loss": before_loss, "after_loss": after_loss,
                        "gain": before_loss - after_loss, "normalized_gain": (before_loss-after_loss) / max(initial_loss, 1e-8),
                        "reference_rows": len(new_reference), "program_changed": visit_incumbent != incumbent.program,
                        "observation_cost": observation_cost, "compute_cost": compute.spent,
                        "evidence": "adaptive_learning_trajectories"}))
        events, added = [], []
        if config.use_memory and measured:
            # A whole learned call can arrive through the ordinary candidate
            # bank as well as startup retrieval. Credit the actual retained
            # source once, including this subsequent invocation path.
            used_source = None
            if incumbent.operation == "reuse" and incumbent.memory_id:
                concept = next((item for item in self.library.concepts
                                if item.concept_id == incumbent.memory_id), None)
                if concept and incumbent.program.expanded().simplified() == concept.program.expanded().simplified():
                    used_source = incumbent.memory_id
                    self.library.record_use(used_source, spec.task_id, spec=spec, observations=observations)
                    for binding in self.library.evolution.bind_concept(used_source, incumbent.program):
                        actions.append(ActionRecord(completed_rounds, "bind_evolution_knowledge",
                            "Associate selected concept structure with reusable knowledge", details=binding))
            for candidate in retrieved:
                if candidate.operation != "reuse":
                    continue
                if incumbent.operation == "reuse" and incumbent.program == candidate.program:
                    if candidate.memory_id != used_source:
                        self.library.record_use(candidate.memory_id, spec.task_id, spec=spec, observations=observations)
                        used_source = candidate.memory_id
                    for binding in self.library.evolution.bind_concept(candidate.memory_id, incumbent.program):
                        actions.append(ActionRecord(completed_rounds, "bind_evolution_knowledge",
                            "Associate selected concept structure with reusable knowledge", details=binding))
                    continue
                if not compute.spend(len(final_rows)):
                    break
                old_loss = loss(candidate.program, final_rows, fallback, spec.risk_quantile)
                if old_loss > final_loss + max(0.05, final_loss * 0.25) and compute.spend(len(final_rows)):
                    row = max(final_rows, key=lambda r: abs(r.target-predict(candidate.program, r, fallback)))
                    before = compute.spent
                    events.append(self.library.record_counterexample(candidate.memory_id, spec, old_loss, final_loss, row,
                                  budget=compute, supporting_observations=observations))
                    actions.append(ActionRecord(completed_rounds, "revise_knowledge_conditions",
                        "Use previous successful observations and new counterexamples to revise applicability",
                        compute_cost=compute.spent-before, details=events[-1]))
            if final_loss + config.minimum_improvement < baseline_loss and incumbent.operation != "reuse":
                identifier, created = self.library.learn(incumbent, spec, observations=observations)
                for binding in self.library.evolution.bind_concept(identifier, incumbent.program):
                    actions.append(ActionRecord(completed_rounds, "bind_evolution_knowledge",
                        "Associate selected concept structure with reusable knowledge", details=binding))
                production_methods = self.library.record_knowledge_production(identifier, spec, method_trace)
                if production_methods:
                    actions.append(ActionRecord(completed_rounds, "link_knowledge_method",
                        "Associate the retained explanation with the measured research steps that produced it",
                        details={"concept_id": identifier, "production_methods": production_methods}))
                if created:
                    added.append(identifier)
                    events.append({"action": "learn", "concept_id": identifier, "expression": incumbent.program.describe()})
        called = [r for r in llm_records if r.get("calls")]
        usage = {"calls": sum(r.get("calls", 0) for r in called), "records": llm_records,
                 "wall_time_seconds": sum(r.get("wall_time_seconds", 0) for r in called),
                 "costs_complete": all(r.get("total_tokens") is not None for r in called)}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] = sum(r.get(key, 0) or 0 for r in called) if usage["costs_complete"] else None
        workspace.incumbent, workspace.parent, workspace.bank = incumbent, parent, bank
        workspace.fallback, workspace.initial_loss = fallback, initial_loss
        workspace.rounds_completed = completed_rounds
        workspace.uncertainty_model = decision_model
        if compute.spend(len(observations)):
            graph.update(diagnose(incumbent.program, observations, fallback, spec.risk_quantile,
                                 max_depth=config.max_question_depth), evidence_count=len(observations),
                         joint=config.joint_planning, depth=config.planning_depth)
            actions.append(ActionRecord(completed_rounds, "update_research_frontier",
                "Carry still-relevant questions and their executable intermediates into the next visit",
                compute_cost=len(observations), details={"pending_plans": [p.plan_id for p in graph.pending_plans()]}))
        workspace.graph = graph.to_dict()
        frontier_work = SearchBudget(min(800, compute.remaining))
        workspace.experiment_frontier = plan_experiment_frontier(spec,
            list(dict.fromkeys([incumbent, *bank[:2]])), observations, queried, config,
            frontier_work, turn=completed_rounds)
        compute.spend(frontier_work.spent)
        if workspace.experiment_frontier:
            actions.append(ActionRecord(completed_rounds, "retain_research_question",
                workspace.experiment_frontier["question"], compute_cost=frontier_work.spent,
                details=dict(workspace.experiment_frontier)))
        if spec.decision_costs is not None:
            from .decision_frontier import plan_decision_frontier, decision_evidence_frontier
            decision_work = SearchBudget(min(1200, compute.remaining))
            workspace.evidence_champion = evidence_champion
            workspace.decision_frontier = (decision_evidence_frontier(spec, workspace, config,
                decision_previous_candidate, proposed_candidate, decision_evidence)
                if decision_evidence and decision_evidence["status"] == "continue" else {})
            if not workspace.decision_frontier:
                workspace.decision_frontier = plan_decision_frontier(spec, workspace, config,
                    decision_work, proposed_model=proposed_decision_model)
            compute.spend(decision_work.spent)
            if workspace.decision_frontier:
                actions.append(ActionRecord(completed_rounds, "retain_decision_question",
                    workspace.decision_frontier["question"], compute_cost=decision_work.spent,
                    details=workspace.decision_frontier))
        simulation_work = SearchBudget(min(600, compute.remaining))
        simulation_update = simulator.flush(budget=simulation_work)
        compute.spend(simulation_work.spent)
        workspace.simulator_state = simulator.to_dict()
        if simulation_work.spent:
            actions.append(ActionRecord(completed_rounds, "revise_experiment_simulator",
                "Compare frozen forecasts with acquired observations and revise the executable observation channel",
                compute_cost=simulation_work.spent, details=simulation_update))
        if config.use_memory:
            knowledge_returns = self.library.finish_transfer_episode(spec, final_candidate=incumbent,
                final_loss=final_loss, baseline_loss=baseline_loss, compute_cost=compute.spent,
                observation_cost=observation_cost, validation_rows=len(final_rows) if measured else 0,
                final_status=("completed" if stop_reason in {"knowledge_reused", "adequate_and_compact"}
                              else "unresolved"))
            if knowledge_returns.get("returns"):
                actions.append(ActionRecord(completed_rounds, "knowledge_return",
                    "Update knowledge and its producing methods from measured downstream returns and costs",
                    details=knowledge_returns))
        for pending in pending_actions:
            pending["compute_cost"] += compute.spent - pending.pop("start_compute")
            pending["observation_cost"] += observation_cost - pending.pop("start_observations")
        workspace.pending_actions = pending_actions
        workspace.method_trace = method_trace
        workspace.research_intent = {}
        workspace.evidence_champion = evidence_champion
        return EpisodeResult(spec.task_id, incumbent.program, fallback, observations, actions, questions,
                             observation_cost, compute.spent, final_loss, initial_loss, incumbent.operation,
                             incumbent.memory_id, added, events, stop_reason, plans=plan_records,
                             wall_time_seconds=perf_counter()-started, llm_usage=usage,
                             uncertainty_model=decision_model, validation_rows=len(final_rows) if measured else 0,
                             research_state=workspace.to_dict())
