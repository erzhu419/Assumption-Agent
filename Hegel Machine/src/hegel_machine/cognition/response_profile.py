"""Active order/curvature research with a revisable sparse response regime."""

from dataclasses import replace
from time import perf_counter

from .continuous_state import _linear_fit, _sum_terms
from .language import Program
from .memory import KnowledgeLibrary
from .models import Candidate, EpisodeResult, Observation, Probe, ResearchConfig, SearchBudget
from .workspace import ResearchWorkspace


def profile_prediction(report, features, *, evidence=()):
    """Answer only where surviving threshold programs agree, or from a reading.

    The returned probe can be funded and executed by the caller. Passing that
    actual reading back answers this input without pretending its neighboring
    unmeasured positions have also been settled.
    """
    for row in evidence:
        if row.features == features:
            return {"status": "measured", "value": row.target}
    alternatives = report["discovery"].get("alternative_programs", [report["episode"]["program"]])
    row = Observation(features, 0.)
    values = [Program.from_dict(payload).predict(row) for payload in alternatives]
    if max(values) - min(values) <= 1e-6:
        return {"status": "agreed", "value": sum(values) / len(values)}
    return {"status": "unknown", "prediction_interval": [min(values), max(values)],
            "probe": {"probe_id": "profile:decision-boundary", "sequence": [features]}}


def run_profile_research(laboratory, config=None, library=None, measurement_budget=4096):
    from .mechanism_discovery import _call, _Unfunded

    config = config or ResearchConfig(observation_budget=96, compute_budget=24000)
    library = library if library is not None else KnowledgeLibrary()
    initial, started = laboratory.costs, perf_counter()
    budget = SearchBudget(config.compute_budget)
    rows, queried, experiments, assessments = [], [], [], []
    discovery, prior, episode = {}, None, None
    axis = Program("feature", name="x0")
    terms = [axis, Program("mul", children=(axis, axis))]

    def pay(units):
        if not budget.spend(units):
            raise _Unfunded

    def acquire(values, name):
        values = list(values)
        if len(rows) + len(values) > min(config.observation_budget, measurement_budget):
            raise _Unfunded
        probe = Probe(name, tuple(laboratory.point([value, 0., 0., 0.]) for value in values))
        batch = laboratory.acquire(probe)
        rows.extend(batch.observations)
        queried.append(probe)
        return batch.observations

    def predicted(program, observations):
        pay(len(observations) * (2 * program.size + 1))
        return [program.predict(row) for row in observations]

    def fit(observations):
        pay(len(observations) * 3 * 3 + 3 ** 3)
        values = [row.features["x0"] for row in observations]
        coefficients = _linear_fit([values, [value**2 for value in values]], [row.target for row in observations])
        return _sum_terms(terms, coefficients[1:], coefficients[0]), coefficients

    try:
        pilot = acquire([index / 10 for index in range(8)], "profile:fit")
        base, coefficients = fit(pilot)
        sources = [record for record in library.structural_priors
                   if record.get("kind") == "mechanism" and record["family"] == "order"]
        source = sources[-1] if sources else None
        base_id = None
        if source:
            old = Program.from_dict(source["base_program"])
            errors = [(value - row.target)**2 for value, row in zip(predicted(old, pilot), pilot)]
            retained = max(errors) < 1e-10
            assessments.append({"record_id": source["record_id"], "status": "retained" if retained else "revised",
                                "loss": sum(errors) / len(errors), "retained_source_support": source["support_rows"]})
            if retained:
                base, base_id = old, source["base_concept_id"]

        # Rank actual proposed interventions by the learned order margin. A
        # concave profile directs research to its smallest marginal return.
        offers = []
        for index in range(8, 21):
            right = index / 20
            points = [right - .05, right - .025, right]
            simulated = [Observation(laboratory.point([value, 0., 0., 0.]), 0.) for value in points]
            predictions = predicted(base, simulated)
            offers.append({"points": points, "predicted_order_margin": predictions[-1] - predictions[0]})
        offer = min(offers, key=lambda item: abs(item["predicted_order_margin"]))
        tested = acquire(offer["points"], "profile:least-order-margin")
        prediction = predicted(base, tested)
        errors = [abs(value - row.target) for value, row in zip(prediction, tested)]
        changed = max(errors) > 1e-6
        experiments.append({"kind": "order_discrimination", **offer,
            "predictions": prediction, "observed": [row.target for row in tested],
            "status": "refuted" if changed else "supported"})
        base_call = _call(base, base_id, f"{base_id}:profile-base", laboratory.feature_names) if base_id else base
        program, boundary, modified, repeats = base_call, None, None, []
        if changed:
            # A reproducible response at the same input earns a local mechanism
            # revision; a fluctuating response would leave this hypothesis open.
            first = next(row for row, error in zip(tested, errors) if error > 1e-6)
            repeats = acquire([first.features["x0"]] * 3 + [pilot[-1].features["x0"]] * 3,
                              "profile:repeat-regime")
            variances = []
            for group in (repeats[:3], repeats[3:]):
                mean = sum(row.target for row in group) / len(group)
                variances.append(sum((row.target - mean)**2 for row in group) / len(group))
            experiments.append({"kind": "exception_or_noise", "repeat_variances": variances,
                                "status": "stable_conditional_exception" if max(variances) < 1e-12 else "unknown"})
            if max(variances) >= 1e-12:
                discovery["unknown"] = "repeated outcomes do not support a deterministic exception"
                raise _Unfunded
            lower, upper = pilot[-1].features["x0"], first.features["x0"]
            while upper - lower > 1e-5:
                middle = (lower + upper) / 2
                fresh = acquire([middle], "profile:change-boundary")
                error = abs(predicted(base, fresh)[0] - fresh[0].target)
                if error > 1e-6:
                    upper = middle
                else:
                    lower = middle
            boundary = {"lower": lower, "upper": upper, "status": "interval",
                        "uncommitted_interval": [lower, upper]}
            observed_error = [abs(value - row.target) for value, row in zip(predicted(base, rows), rows)]
            changed_rows = [row for row, error in zip(rows, observed_error) if error > 1e-6]
            modified, local_coefficients = fit(changed_rows)
            threshold = (lower + upper) / 2
            program = Program("if", children=(Program("ge", children=(axis, Program("const", value=threshold))),
                                               modified, base_call))
            experiments.append({"kind": "local_regime_revision", "boundary": boundary,
                                "retained_program": base.to_dict(), "replacement_program": modified.to_dict(),
                                "replacement_rows": len(changed_rows)})

        # Fresh positions include both sides outside the unresolved threshold
        # interval; none of these outcomes participate in fitting the program.
        probe = laboratory.random_probe(config.seed + 7919, 12, "profile:prospective")
        validation_values = [point["x0"] for point in probe.sequence]
        if boundary:
            width = boundary["upper"] - boundary["lower"]
            validation_values += [boundary["lower"] - width, boundary["upper"] + width]
        validation = acquire(validation_values, "profile:prospective")
        values = predicted(program, validation)
        error = sum((value-row.target)**2 for value, row in zip(values, validation)) / len(values)
        if error > 1e-8:
            discovery["unknown"] = "revised profile failed new positions"
            raise _Unfunded
        ordered = sorted(pilot + tested, key=lambda row: row.features["x0"])
        order_violation = next(({"lower": {"x": a.features["x0"], "y": a.target},
                                 "upper": {"x": b.features["x0"], "y": b.target}}
                                for a, b in zip(ordered, ordered[1:]) if b.target < a.target - 1e-8), None)
        slopes = [(b.target-a.target) / (b.features["x0"]-a.features["x0"])
                  for a, b in zip(tested, tested[1:])]
        diminishing = slopes[1] <= slopes[0] + 1e-7
        discovery = {"hypotheses": [
            {"kind": "monotonicity", "status": "refuted" if order_violation else "supported",
             "counterexample": order_violation},
            {"kind": "diminishing_returns", "status": "supported" if diminishing else "refuted",
             "measured_adjacent_slopes": slopes}],
            "base_coefficients": coefficients, "base_program": base.to_dict(),
            "regimes": 2 if changed else 1, "change_boundary": boundary,
            "smooth_regions": [[0., boundary["lower"]], [boundary["upper"], 1.]] if boundary else [[0., 1.]],
            "exception_fraction_of_control_domain": 1. - boundary["upper"] if boundary else 0.,
            "uncertainty": {"boundary": boundary, "unmeasured_noise_model": "unknown"},
            "raw_dimension": 4, "effective_dimension": 1,
            "triangulation": {"channels": ["profile_fit", "ordered_intervention", "repeat_response", "new_positions"],
                              "prospective_loss": error},
            "applied_concept_ids": [base_id] if base_id else []}
        alternatives = [program]
        if boundary:
            alternatives = [Program("if", children=(Program("ge", children=(axis, Program("const", value=threshold))),
                                                       modified, base_call))
                            for threshold in (boundary["lower"], boundary["upper"])]
            discovery["alternative_programs"] = [item.to_dict() for item in alternatives]
        concepts = []
        if base_id is None:
            base_id, created = library.learn(Candidate("profile:base", base, "profile_discovery"),
                replace(laboratory.spec, task_id=f"{laboratory.spec.task_id}:base"), pilot)
            if created:
                concepts.append(base_id)
        candidate = Candidate("profile:task", program, "local_mechanism_revision" if changed else
                              "mechanism_reuse" if source else "profile_discovery", parents=(base_id,),
                              memory_id=base_id if source else None)
        concept_id, created = library.learn(candidate, laboratory.spec, rows)
        if created:
            concepts.append(concept_id)
        workspace = ResearchWorkspace(laboratory.spec.task_id, observations=rows, reference=validation,
            queried=queried, incumbent=candidate, evidence_champion=candidate,
            bank=[candidate, *[Candidate(f"profile:boundary:{index}", alternative, "scope_revision")
                               for index, alternative in enumerate(alternatives) if alternative != program]],
            fallback=0., initial_loss=sum(row.target**2 for row in rows) / len(rows))
        prior = {"kind": "mechanism", "record_id": f"mechanism-prior:{len(library.structural_priors)}",
                 "family": "order", "base_program": base.to_dict(), "base_concept_id": base_id,
                 "program": program.to_dict(), "discovery": discovery, "support_rows": len(rows),
                 "revises": [source["record_id"]] if source and changed else []}
        library.structural_priors.append(prior)
        episode = EpisodeResult(laboratory.spec.task_id, program, 0., rows, [], [], len(rows), budget.spent,
            error, workspace.initial_loss, candidate.operation, candidate.memory_id, concepts,
            [{"kind": "component_reuse", "concept_id": base_id}] if source else [],
            "profile_prospectively_applied", validation_rows=len(validation), research_state=workspace.to_dict()).to_dict()
    except _Unfunded:
        workspace = ResearchWorkspace(laboratory.spec.task_id, observations=rows, queried=queried,
            experiment_frontier={"kind": "response_profile", "reason": discovery.get("unknown", "additional funded evidence required")})
        episode = {"task_id": laboratory.spec.task_id, "program": Program("const", value=0.).to_dict(),
                   "research_state": workspace.to_dict(), "validation_loss": None, "validation_rows": 0,
                   "concepts_added": [], "stop_reason": workspace.experiment_frontier["reason"]}
    costs = laboratory.costs
    return {"task_id": laboratory.spec.task_id, "status": "completed" if prior else "unresolved",
            "episode": episode, "discovery": discovery, "experiments": experiments,
            "prior_assessments": assessments, "structural_prior": prior,
            "costs": {"observation_cost": costs["observation_cost"] - initial["observation_cost"],
                      "measurement_cost": costs["measurement_cost"] - initial["measurement_cost"],
                      "compute_cost": budget.spent, "wall_time_seconds": perf_counter() - started}}
