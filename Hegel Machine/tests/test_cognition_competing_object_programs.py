"""Old observational agreement must not erase executable competitors."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.competing_object_programs import research_competing_object_programs
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


def _spec():
    quantities = ["amount_a", "amount_b"]
    model = {"model_id": "observed_partial_objects", "complete": False, "schema": None,
        "focus_control": "select", "unresolved": ["a:x", "b:x"],
        "groups": [{"object_id": name, "q": f"amount_{name}", "u": f"drive_{name}",
                    "f": f"selector_{name}", "controls": {"q": f"set_q_{name}", "u": f"set_u_{name}"}}
                   for name in ("a", "b")],
        "position_candidates": {f"position_{name}": {"quantities": quantities,
            "control": f"place_{name}"} for name in ("a", "b")},
        "focus_candidates": {f"selector_{name}": {"quantities": [f"amount_{name}"]}
            for name in ("a", "b")}}
    names = tuple(f"{role}_{name}" for name in ("a", "b")
                  for role in ("amount", "drive", "position", "selector"))
    return TaskSpec("ambiguous_observation", "An observed focal response", names, (), (),
                    context={"object_model": model})


def _rows(seed, count, *, separate=False):
    rng, rows = Random(seed), []
    for _ in range(count):
        point = {f"{role}_{name}": rng.uniform(-1., 1.)
                 for name in ("a", "b") for role in ("amount", "drive", "position")}
        if not separate:
            point["position_b"] = point["position_a"]
        focus = rng.choice(("a", "b"))
        point.update({f"selector_{name}": float(name == focus) for name in ("a", "b")})
        target = 1.2 * point[f"amount_{focus}"] + .7 * point[f"drive_{focus}"] \
            + 2.1 * point[f"position_{focus}"]
        rows.append(Observation(point, target))
    return rows


def test_independent_fits_keep_non_equivalent_laws_that_agree_on_actual_old_rows():
    spec, training = _spec(), _rows(71, 36)
    before = deepcopy(spec.context)
    budget = SearchBudget(60000)
    result = research_competing_object_programs(spec, training, budget)
    state, entries = result["competition"], result["entries"]
    assert state["exhaustive"] and len(state["schemas"]) == 2
    assert state["status"] == "competing" and state["program_classes"] >= 2
    assert {json.dumps(entry["schema"], sort_keys=True) for entry in entries} == {
        json.dumps(schema, sort_keys=True) for schema in state["schemas"]}
    assert all(entry["training_loss"] < 1e-7 for entry in entries)
    fresh = _rows(97, 48, separate=True)
    errors = [loss(entry["candidate"].program, fresh, 0., .5) for entry in entries]
    assert min(errors) < 1e-7 and max(errors) > .1
    assert state["compute_cost"] == budget.spent <= budget.limit
    assert all(search["compute_cost"] > 0 for search in state["searches"])
    assert spec.context == before and spec.context["object_model"]["schema"] is None
    restored = json.loads(json.dumps(state))
    assert all(Program.from_dict(item["program"]) for item in restored["hypotheses"])


def test_actual_new_observations_refute_old_programs_and_refit_without_deciding_identity():
    spec, training, budget = _spec(), _rows(71, 36), SearchBudget(160000)
    original = research_competing_object_programs(spec, training, budget)["competition"]
    frozen = json.loads(json.dumps(original))
    actual = _rows(103, 48, separate=True)
    updated = research_competing_object_programs(spec, training + actual, budget,
        previous=frozen, new_observations=actual)
    state = updated["competition"]
    assert state["counterexamples"]
    refuted = [item for item in state["counterexamples"] if item["status"] == "refuted"]
    assert refuted and all(item["assessment_loss"] > .1 for item in refuted)
    assert all(item["counterexamples"][-1]["observation_count"] == len(actual) for item in refuted)
    assert all(item["hypothesis_id"] in {old["hypothesis_id"] for old in original["hypotheses"]}
               for item in refuted)
    assert updated["entries"]
    fresh = _rows(809, 64, separate=True)
    assert all(loss(entry["candidate"].program, fresh, 0., .5) < 1e-7 for entry in updated["entries"])
    assert all(item["parent_hypothesis_id"] for item in state["hypotheses"])
    assert len(state["schemas"]) == 2 and spec.context["object_model"]["schema"] is None
    assert frozen == json.loads(json.dumps(original))


def test_counterexamples_survive_when_no_current_program_is_supported():
    spec, training = _spec(), _rows(71, 36)
    original = research_competing_object_programs(spec, training, SearchBudget(60000))["competition"]
    actual = [Observation(row.features, row.target + 100.) for row in _rows(79, 12, separate=True)]
    # This budget buys evaluation of the frozen hypotheses, then no useful
    # structural refit. Failure still needs a returned research state.
    budget = SearchBudget(12 * len(original["hypotheses"]) + 10)
    result = research_competing_object_programs(spec, training + actual, budget,
        previous=original, new_observations=actual)
    assert result["entries"] == []
    assert result["competition"]["counterexamples"]
    assert result["competition"]["status"] == "no_supported_program"
    assert budget.spent <= budget.limit


def test_assignment_search_budget_is_never_a_unique_object_claim():
    spec, budget = _spec(), SearchBudget(2)
    result = research_competing_object_programs(spec, _rows(7, 12), budget)
    assert not result["entries"] and not result["competition"]["exhaustive"]
    assert budget.spent == 2 and spec.context["object_model"]["schema"] is None


def test_later_agreement_cannot_readopt_a_program_refuted_before_refit_budget_ran_out():
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    spec, training = _spec(), _rows(71, 36)
    original = research_competing_object_programs(spec, training, SearchBudget(60000))
    fresh = _rows(103, 48, separate=True)
    rejected = max(original["entries"], key=lambda item: loss(item["candidate"].program, fresh, 0., .5))
    # Start from an unfinished search that retained this one executable law.
    previous = deepcopy(original["competition"])
    previous["hypotheses"] = [item for item in previous["hypotheses"]
                              if item["hypothesis_id"] == rejected["hypothesis_id"]]
    rejected["object_competition"] = previous
    # This later trajectory returns to the old operating regime. It agrees
    # perfectly with the rejected law but cannot answer the observed failure.
    later = _rows(809, 48)
    assert loss(rejected["candidate"].program, later, 0., .5) < 1e-7
    agent = UnifiedResearchAgent()
    allowance = SearchBudget(len(fresh) + 10)
    result = agent._object_explanations(spec, training + fresh, allowance, actions=[],
        parent=rejected, new_observations=fresh)
    assert not result
    assert rejected["object_competition"]["counterexamples"]
    assert agent._object_program_refuted(rejected, spec)
    # A later search uses the saved counterexamples and finds a supported law.
    repaired = agent._object_explanations(spec, training + fresh, SearchBudget(160000),
        actions=[], parent=rejected, new_observations=fresh)
    assert repaired and all(not agent._object_program_refuted(item, spec) for item in repaired)


def test_supported_frozen_programs_survive_when_budget_cannot_buy_another_fit():
    spec, training = _spec(), _rows(71, 36)
    initial = research_competing_object_programs(spec, training, SearchBudget(60000))["competition"]
    actual = _rows(81, 12)
    rows = training + actual
    # Enumeration and evaluation are affordable, another structural fit is not.
    budget = SearchBudget((len(actual) + len(rows)) * len(initial["hypotheses"]) + 10)
    result = research_competing_object_programs(spec, rows, budget,
        previous=initial, new_observations=actual)
    assert len(result["entries"]) == len(initial["hypotheses"])
    assert all(item["structure_search"] == "retained_frozen_law" for item in result["entries"])
    assert len({json.dumps(item["schema"], sort_keys=True) for item in result["entries"]}) == 2
    assert all(item["training_rows"] == len(rows) for item in result["competition"]["hypotheses"])
    assert budget.spent <= budget.limit


def test_interrupted_assignment_search_preserves_prior_hypotheses_without_refuting_them():
    spec, training = _spec(), _rows(71, 36)
    initial = research_competing_object_programs(spec, training, SearchBudget(60000))["competition"]
    interrupted = research_competing_object_programs(spec, training, SearchBudget(2), previous=initial)
    state = interrupted["competition"]
    assert state["counterexamples"] == [] and not interrupted["entries"]
    assert len(state["unassessed_hypotheses"]) == len(initial["hypotheses"])
    resumed = research_competing_object_programs(spec, training, SearchBudget(1000), previous=state)
    assert resumed["entries"] and not resumed["competition"]["counterexamples"]


def test_joint_family_mismatch_leaves_budget_to_find_a_different_law():
    spec = _spec()
    old_rows = []
    for row in _rows(7109, 48, separate=True):
        p = row.features
        selected = "a" if p["selector_a"] else "b"
        other = "b" if selected == "a" else "a"
        exchange = p[f"amount_{other}"] - p[f"amount_{selected}"] \
            if abs(p["position_a"] - p["position_b"]) < .5 else 0.
        target = 1.2 * p[f"amount_{selected}"] + .7 * p[f"drive_{selected}"] + 1.9 * exchange
        old_rows.append(Observation(p, target))
    initial = research_competing_object_programs(spec, old_rows, SearchBudget(80000),
        max_candidates=2)["competition"]
    assert len(initial["hypotheses"]) == 2
    assert all(item["structure"]["fits"][0]["family"] == "local_exchange"
               for item in initial["hypotheses"])
    current = [Observation(row.features, row.features["amount_a"] + row.features["amount_b"]
        + .4 * (row.features["drive_a"] + row.features["drive_b"]))
        for row in _rows(9127, 48, separate=True)]
    budget = SearchBudget(3000)
    result = research_competing_object_programs(spec, current, budget,
        previous=initial, new_observations=current, max_candidates=2)
    assert result["competition"]["counterexamples"]
    assert len(result["entries"]) == 2
    assert all(item["structure"]["fits"][0]["family"] == "conserved_total"
               and item["training_loss"] < 1e-7 for item in result["entries"])
    assert budget.spent <= budget.limit


def test_long_prefix_cannot_dilute_the_new_counterexample_back_into_support():
    spec, training = _spec(), _rows(71, 240)
    initial = research_competing_object_programs(spec, training, SearchBudget(250000),
        max_candidates=2)["competition"]
    actual = [Observation(row.features, row.target + .002) for row in _rows(79, 4)]
    for item in initial["hypotheses"]:
        program = Program.from_dict(item["program"])
        assert loss(program, training + actual, 0., .5) <= 1e-7
        assert loss(program, actual, 0., .5) > 1e-7
    revised = research_competing_object_programs(spec, training + actual, SearchBudget(160000),
        previous=initial, new_observations=actual, max_candidates=2)
    assert len(revised["competition"]["counterexamples"]) == 2
    assert not revised["entries"]


def test_fitted_competitors_choose_and_buy_a_separating_experiment_before_revision():
    from hegel_machine.cognition.competing_object_experiments import (
        assess_competing_object_probe, plan_competing_object_probe,
    )

    spec, training, work = _spec(), _rows(71, 36), SearchBudget(120000)
    initial = research_competing_object_programs(spec, training, work)
    # The public selector command is converted to complementary binary readings;
    # other measured coordinates are directly controllable in this fixture.
    bounds = {name: (-1., 1.) for name in spec.feature_names if not name.startswith("selector_")}
    command = replace(spec, feature_names=tuple(bounds) + ("select",),
        feature_bounds={**bounds, "select": (0., 1.)})
    selector = Program("lt", children=(Program("feature", name="select"), Program("const", value=.5)))
    binding = {"selector_a": selector, "selector_b": Program("sub", children=(
        Program("const", value=1.), selector))}
    plan = plan_competing_object_probe(command, initial["entries"], work, binding=binding)
    assert plan["forecast_status"] == "predicted_distinction" and plan["probe"] is not None
    pairs = []
    for point in plan["probe"].sequence:
        selected = "a" if point["select"] < .5 else "b"
        measured = {**point, "selector_a": float(selected == "a"), "selector_b": float(selected == "b")}
        target = 1.2 * measured[f"amount_{selected}"] + .7 * measured[f"drive_{selected}"] \
            + 2.1 * measured[f"position_{selected}"]
        pairs.append((Observation(point, target), Observation(measured, target)))
    assessment = assess_competing_object_probe(plan, pairs, work)
    assert assessment["status"] == "distinguished"
    refuted_ids = {item["hypothesis_id"] for item in assessment["hypotheses"]
                   if item["status"] == "counterexample"}
    actual = [measured for _, measured in pairs]
    revised = research_competing_object_programs(spec, training + actual, work,
        previous=initial["competition"], new_observations=actual)
    assert refuted_ids <= {item["hypothesis_id"] for item in revised["competition"]["counterexamples"]}
    assert revised["entries"] and work.spent <= work.limit


def _boundary_lab():
    from hegel_machine.cognition.models import Probe
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab

    lab = UnifiedObjectLab(objects=2, readout="focus", apparatus_seed=71)
    measured = lab.spec_for(("fine",))
    schema = json.loads(measured.context["object_schema"])
    fields, controls = schema["fields"], schema["controls"]
    quantities = [item["q"] for item in fields.values()]
    model = {"model_id": "partially_observed", "focus_control": schema["focus_control"],
        "complete": False, "schema": None, "unresolved": ["x"],
        "groups": [{"object_id": obj, **{role: item[role] for role in ("q", "u", "f")},
                    "controls": {role: controls[obj][role] for role in ("q", "u")}}
                   for obj, item in fields.items()],
        "position_candidates": {item["x"]: {"quantities": quantities, "control": controls[obj]["x"]}
                                for obj, item in fields.items()},
        "focus_candidates": {item["f"]: {"quantities": [item["q"]]} for item in fields.values()}}
    positions = [item["x"] for item in controls.values()]
    original = lab.random_probe(971, length=48)
    probe = Probe("observed_near_and_far", tuple({**point, positions[0]: 0.,
        positions[1]: .2 if index % 2 else 1.8} for index, point in enumerate(original.sequence)))
    batch = lab.acquire(probe, ("fine",))
    return lab, replace(measured, context={"object_model": model}), list(batch.observations), positions


def test_actual_boundary_feedback_updates_a_supported_frozen_law_and_next_probe():
    from hegel_machine.cognition.unified_structure import structural_probe

    lab, spec, rows, positions = _boundary_lab()
    work = SearchBudget(120000)
    result = research_competing_object_programs(spec, rows, work, max_candidates=2)
    original_program = result["entries"][0]["candidate"].program
    widths, distances = [], []
    for serial in range(2):
        entry = result["entries"][0]
        before = entry["structure"]["radius_interval"]
        widths.append(before[1] - before[0])
        probe = structural_probe(entry, lab.spec, serial, 4)
        distances.append(probe.sequence[0][positions[1]] - probe.sequence[0][positions[0]])
        batch = lab.acquire(probe, ("fine",))
        rows.extend(batch.observations)
        before_compute = work.spent
        result = research_competing_object_programs(spec, rows, work,
            previous=result["competition"], new_observations=batch.observations,
            transitions=batch.transitions, max_candidates=2)
        assert result["entries"][0]["candidate"].program == original_program
        assert all(item["structure_search"] == "retained_boundary_update" for item in result["entries"])
        assert work.spent - before_compute < 500
        interval = result["entries"][0]["structure"]["radius_interval"]
        assert interval[1] - interval[0] < widths[-1]
    assert distances[1] < distances[0]


def test_unresolved_radius_provides_paid_validated_executable_competitors_and_a_real_probe():
    from hegel_machine.cognition.competing_object_experiments import (
        assess_competing_object_probe, plan_competing_object_probe,
    )
    from hegel_machine.cognition.instrument_models import fit_instrument_model, rebind_instrument_model
    from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab

    lab, spec, rows, _ = _boundary_lab()
    work = SearchBudget(160000)
    result = research_competing_object_programs(spec, rows, work)
    entries = result["entries"]
    variants = [item for item in entries if item["structure_search"] == "radius_alternative"]
    assert variants and len(entries) <= 8
    assert len({json.dumps(item["schema"], sort_keys=True) for item in entries}) == 2
    assert len({item["structure"]["fits"][0]["radius"] for item in entries}) >= 3
    assert all(loss(item["candidate"].program, rows, 0., .5) < 1e-7 for item in entries)
    for item in variants:
        radius = item["structure"]["fits"][0]["radius"]
        low, high = item["structure"]["radius_interval"]
        assert low < radius < high

    # Calibration is reusable apparatus knowledge purchased on a previous
    # total-response task. Its response labels never fit the current focus law.
    warmup = UnifiedObjectLab(objects=2, readout="total", apparatus_seed=71)
    calibration_probe = warmup.random_probe(491, length=48)
    calibration_rows = warmup.acquire(calibration_probe, ("fine",)).observations
    pairs = [(Observation(point, row.target), row)
             for point, row in zip(calibration_probe.sequence, calibration_rows)]
    calibration_spec = warmup.spec_for(("fine",))
    calibration = fit_instrument_model(warmup.spec, calibration_spec, pairs, work)
    binding = rebind_instrument_model(calibration, calibration_spec)
    plan = plan_competing_object_probe(lab.spec, entries, work, binding=binding)
    assert plan["forecast_status"] == "predicted_distinction"
    before = lab.observation_cost
    actual = lab.acquire(plan["probe"], ("fine",))
    assert lab.observation_cost - before == plan["probe"].cost
    assessment = assess_competing_object_probe(plan, [(Observation(point, row.target), row)
        for point, row in zip(plan["probe"].sequence, actual.observations)], work)
    assert assessment["status"] == "distinguished"
    refuted = {item["hypothesis_id"] for item in assessment["hypotheses"] if item["status"] == "counterexample"}
    revised = research_competing_object_programs(spec, rows + actual.observations, work,
        previous=result["competition"], new_observations=actual.observations)
    assert refuted <= {item["hypothesis_id"] for item in revised["competition"]["counterexamples"]}
    assert revised["entries"] and work.spent <= work.limit
