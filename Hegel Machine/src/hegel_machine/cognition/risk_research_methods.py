"""Reuse paid risk research computations under the unchanged public quantile."""

from copy import copy, deepcopy
from math import ceil
from statistics import mean

from .action_policy import ResearchActionPolicy
from .continuous_state import _linear_fit, _sum_terms
from .evaluation import initial_prediction, loss, predict
from .language import Program
from .method_conditions import _representation
from .models import Candidate
from .research_drafts import _response_terms
from .research_programs import _Fitter, _sources
from .risk_search import empirical_quantile, trajectory_blocks


METHOD_ACTION = "successor_partial_method_visit"


def _scale_kind(selected):
    kind = selected.get("kind", "")
    return "absolute" if "absolute_scale" in kind else "quadratic" if "quadratic_scale" in kind else "global"


def refit_risk_method(recipe, spec, rows, budget, *, mechanism=None):
    """Refit the recorded mechanism coordinates, then its recorded tail family.

    Location coefficients propose a mechanism; their residual empirical
    quantile and conditional scale are fitted at spec.risk_quantile. Neither
    location fitting nor its training score certifies a risk predictor.
    """
    rows = list(rows)
    if not rows:
        return None
    base = mechanism or Program.from_dict(recipe["mechanism_program"])
    if mechanism is None:
        terms = tuple(dict.fromkeys(_response_terms(base)))
        if terms:
            fitted = _Fitter(rows, budget, spec.risk_quantile).fit(terms)
            if fitted is None:
                return None
            base = fitted["program"]
        elif not budget.spend(len(rows)):
            return None
        else:
            base = Program("const", value=mean(row.target for row in rows))
    fallback = initial_prediction(rows, spec.risk_quantile)
    if not budget.spend(len(rows)):
        return None
    residuals = [row.target - predict(base, row, fallback) for row in rows]
    kind = recipe.get("uncertainty_kind", "global")
    if kind == "global":
        if not budget.spend(len(rows)):
            return None
        correction = Program("const", value=empirical_quantile(residuals, spec.risk_quantile))
    else:
        names = [name for name in spec.feature_names if name in base.required_features()]
        terms = [Program("abs", children=(Program("feature", name=name),)) if kind == "absolute"
                 else Program("mul", children=(Program("feature", name=name),) * 2) for name in names]
        if not budget.spend(len(rows) * (len(terms) + 5)):
            return None
        center = mean(residuals)
        magnitudes = [abs(value - center) for value in residuals]
        coefficients = _linear_fit([[term.predict(row) for row in rows] for term in terms], magnitudes)
        spread = Program("max", children=(Program("const", value=max(1e-12, .05 * mean(magnitudes))),
            _sum_terms(terms, coefficients[1:], coefficients[0])))
        standardized = [(value - center) / spread.predict(row) for value, row in zip(residuals, rows)]
        correction = Program("add", children=(Program("const", value=center), Program("mul", children=(
            Program("const", value=empirical_quantile(standardized, spec.risk_quantile)), spread))))
    if not budget.spend(len(rows)):
        return None
    program = Program("add", children=(base, correction)).simplified()
    method = {**deepcopy(recipe), "mechanism_program": base.to_dict(), "response_program": program.to_dict()}
    return {"candidate": Candidate(f"risk_method:{budget.spent}", program, "risk_method_replay"),
        "kind": f"residual_{kind}_scale_quantile" if kind != "global" else "residual_global_quantile",
        "family_id": "T22", "fallback": fallback, "fixed_quantile": spec.risk_quantile,
        "training_loss": loss(program, rows, fallback, spec.risk_quantile), "risk_method": method}


def remember_risk_method(library, spec, selected, origin, costs, comparison, confirmed):
    """Save a concrete executed recipe and its actual, possibly failed return."""
    recipe = deepcopy(selected.get("risk_method") or {})
    recipe.update(mechanism_program=recipe.get("mechanism_program", origin["baseline_program"]),
        response_program=selected["candidate"].program.to_dict(),
        uncertainty_kind=recipe.get("uncertainty_kind", _scale_kind(selected)),
        steps=list(origin["method_steps"]))
    parent_id = origin.get("source_draft_id")
    record = next((item for item in library.research_drafts
        if item.get("risk_method") == recipe and item["source_task_id"] == spec.task_id
        and item.get("source_seed") == origin["source_seed"]), None)
    if record is None:
        index = 1 + max((int(item["draft_id"].rsplit("_", 1)[1]) for item in library.research_drafts), default=0)
        parent = next((item for item in library.research_drafts if item["draft_id"] == parent_id), {})
        record = {"draft_id": f"research_draft_{index:04d}", "source_task_id": spec.task_id,
            "risk_quantile": spec.risk_quantile, "source_seed": origin["source_seed"],
            "source_features": list(spec.feature_names),
            "control_features": list(spec.feature_bounds),
            "observed_features": [name for name in spec.feature_names if name not in spec.feature_bounds],
            "risk_method": recipe, "parent_draft_id": parent_id, "outcomes": [], "status": "unresolved",
            "source_ids": list(dict.fromkeys([*_sources(Program.from_dict(recipe["mechanism_program"])),
                *_sources(selected["candidate"].program), *parent.get("source_ids", ())]))}
        library.research_drafts.append(record)
    record["status"] = "supported" if confirmed else "unresolved"
    outcome = {"fresh_comparison": deepcopy(comparison), "confirmed": confirmed,
        "whole_visit_costs": dict(costs), "evidence_condition": origin["evidence_condition"]}
    record["outcomes"].append(outcome)
    selected.update(risk_method=deepcopy(recipe), research_draft_id=record["draft_id"])
    return record


def _binding(record, spec):
    commands = list(spec.feature_bounds)
    readings = [name for name in spec.feature_names if name not in spec.feature_bounds]
    mapping = {}
    for source, target in ((record["control_features"], commands), (record["observed_features"], readings)):
        if len(source) > len(target):
            return None
        mapping.update({name: name for name in source if name in target})
        available = [name for name in target if name not in mapping.values()]
        mapping.update(zip((name for name in source if name not in mapping), available))
    return mapping


def _applicability(recipe, spec, reason):
    program = Program.from_dict(recipe["mechanism_program"])
    return {"method_steps": list(recipe["steps"]), "uncertainty_kind": recipe["uncertainty_kind"],
        "response_structure": _representation(program, spec.feature_names)["structure"],
        "risk_quantile": spec.risk_quantile, "evidence_condition": reason}


def _visit_policy(library, spec, rows, applicability):
    policy = ResearchActionPolicy(rows, spec, library.action_experience)
    policy.set_condition("successor_risk_method_paid_failure")
    policy.set_action_knowledge(METHOD_ACTION, applicability)
    return policy


def replay_risk_methods(agent, spec, training, validation, budget, actions, draft_ids=None):
    """Execute retained coordinates and tail fits before a new grammar search."""
    checked = []
    for record in agent.library.research_drafts:
        if (not record.get("risk_method") or record["risk_quantile"] != spec.risk_quantile
                or (draft_ids is not None and record["draft_id"] not in draft_ids)
                or record.get("source_seed") == agent.config.seed):
            continue
        mapping = _binding(record, spec)
        if mapping is None:
            continue
        recipe = deepcopy(record["risk_method"])
        recipe["mechanism_program"] = Program.from_dict(recipe["mechanism_program"]).remap_features(mapping).to_dict()
        start = budget.spent
        claim = refit_risk_method(recipe, spec, training, budget)
        scored = claim is not None and budget.spend(len(validation))
        if budget.spent == start:
            continue
        measured = (loss(claim["candidate"].program, validation, claim["fallback"], spec.risk_quantile)
                    if scored else None)
        actions.append({"action": "replay_risk_method", "draft_id": record["draft_id"],
            "source_task_id": record["source_task_id"], "task_id": spec.task_id,
            "training_loss": claim["training_loss"] if claim is not None else None, "validation_loss": measured,
            "fitting_rows": len(training), "validation_rows": len(validation),
            "fixed_quantile": spec.risk_quantile, "compute_cost": budget.spent - start,
            "status": "requires_fresh_confirmation" if scored else "no_scored_candidate"})
        if not scored:
            continue
        claim["research_draft_id"] = record["draft_id"]
        claim["validation_loss"] = measured
        checked.append((claim, record))
    return checked


def offer_risk_method_experiment(agent, laboratory, previous, draft_ids, budget, *, actions=None):
    """Price a retained risk recipe from fitting and scoring paid trajectories."""
    from .unified_continuation import paid_view

    if previous["status"] != "unresolved" or laboratory.spec.risk_quantile == .5 or not agent.config.risk_learning:
        return None
    channels = tuple(channel.name for channel in laboratory.channels)
    spec = agent._spec(laboratory, channels)
    rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
    blocks = trajectory_blocks(rows)
    if len(blocks) < 2 or len(rows) < 32:
        return None
    training, validation = [row for block in blocks[:-1] for row in block], blocks[-1]
    old_channels = tuple(previous["selected_channels"])
    old_rows = [row for batch in paid_view(laboratory, previous, old_channels) for row in batch]
    if not budget.spend(len(old_rows)):
        return None
    baseline = loss(Program.from_dict(previous["program"]), old_rows, previous["fallback"], spec.risk_quantile)
    checked = replay_risk_methods(agent, spec, training, validation, budget,
        actions if actions is not None else [], draft_ids)
    if not checked:
        return None
    chosen, record = min(checked, key=lambda item: (item[0]["validation_loss"], item[0]["candidate"].program.size))
    if not budget.spend(3 * len(validation)):
        return None
    old_validation = old_rows[-len(validation):]
    separation = mean((predict(chosen["candidate"].program, row, chosen["fallback"])
        - predict(Program.from_dict(previous["program"]), old, previous["fallback"])) ** 2
        for row, old in zip(validation, old_validation))
    if separation <= agent.config.minimum_improvement:
        return None
    reason = (previous.get("noise_assessment") or {}).get("reason", "unassessed_residual")
    applicability = _applicability(chosen["risk_method"], spec, reason)
    experience_library = copy(agent.library)
    experience_library.action_experience = deepcopy(agent.library.action_experience)
    policy = _visit_policy(experience_library, spec, rows, applicability)
    experience = policy.action_record(METHOD_ACTION)
    estimates = policy._estimated_costs(experience)
    length = min(24, laboratory.spec.max_probe_length)
    observation_quote = max(4 * length, ceil(estimates["observation_cost"]))
    rate = sum(channel.cost_per_row for channel in laboratory.channels)
    measurement_quote = max(observation_quote * rate, ceil(estimates["measurement_cost"]))
    compute_quote = (max(12000 + 8 * len(rows) + 32 * length,
        ceil(estimates["compute_cost"]) - budget.spent) if experience.get("trials", 0)
        + experience.get("unscored_attempts", 0) else agent.config.compute_budget)
    gain = baseline - chosen["validation_loss"]
    expected = max(0., gain) / max(baseline, agent.config.minimum_improvement) + .5 * separation / (separation + baseline)
    if experience.get("trials"):
        expected = max(0., max(0., gain) / max(baseline, agent.config.minimum_improvement) + experience["expected_gain"])
    if expected <= 0.:
        return None
    opportunity = {"draft_ids": [record["draft_id"]], "source_task_ids": [record["source_task_id"]],
        "paid_counterexample_loss": baseline, "candidate_paid_loss": chosen["validation_loss"],
        "predicted_separation": separation, "planning_compute_cost": budget.spent,
        "method_applicability": applicability, "whole_visit_compute_quote": compute_quote,
        "observed_visit_costs": estimates, "observed_visit_trials": experience.get("trials", 0),
        "observed_expected_gain": experience.get("expected_gain"), "requires_fresh_confirmation": True}
    share = (observation_quote / agent.config.observation_budget
        + (compute_quote + budget.spent) / agent.config.compute_budget
        + measurement_quote / max(1, agent.measurement_budget))
    return {"candidate": {"program": chosen["candidate"].program.to_dict(), "channels": list(channels),
        "fallback": chosen["fallback"], "kind": "successor_risk_method", "family_id": "T22",
        "risk_method": chosen["risk_method"], "research_draft_id": record["draft_id"],
        "source_concept_ids": [], "method_opportunity": opportunity},
        "basis": "new_partial_method_distinguishing_experiment", "old_evidence_gain": gain,
        "old_evidence_rows": len(rows), "predicted_separation": separation,
        "relative_gain": expected / max(share, 1e-12), "minimum_resources": {"observation_cost": observation_quote,
            "measurement_cost": measurement_quote, "compute_cost": compute_quote}}


def record_risk_method_visit(library, spec, previous, report, opportunity, screening_compute_cost):
    """Credit the complete visit only from its last independent paired score."""
    costs = {key: report["visit_costs"][key] for key in ("observation_cost", "measurement_cost", "compute_cost")}
    costs["compute_cost"] += screening_compute_cost
    policy = _visit_policy(library, spec, [], opportunity["method_applicability"])
    comparisons = report.get("continuation", {}).get("method_fresh_comparisons", [])
    paired = comparisons[-1] if comparisons else None
    experience = (policy.update(METHOD_ACTION, before_loss=paired["before_loss"], after_loss=paired["after_loss"],
        **costs) if paired else policy.record_unscored_attempt(METHOD_ACTION, status="unresolved", **costs))
    outcome = {"task_id": spec.task_id, "whole_visit_costs": costs, "screening_compute_cost": screening_compute_cost,
        "fresh_comparison": deepcopy(paired), "status": report["status"],
        "method_context": policy.action_context(METHOD_ACTION),
        "method_applicability": deepcopy(opportunity["method_applicability"])}
    report["continuation"]["method_visit_outcome"] = deepcopy(outcome)
    for record in library.research_drafts:
        if record["draft_id"] in opportunity["draft_ids"]:
            record["outcomes"].append(deepcopy(outcome))
    return experience
