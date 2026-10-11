"""A dense archive must not acquire extra votes for one physical forecast."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_experiment_choice import select_experiment
from hegel_machine.cognition.world_response_consequences import _contrast_order
from hegel_machine.cognition.world_response_transfer import reuse_response_knowledge
from test_world_response_hypotheses import PATH
from test_world_response_ordering import alternatives


def _linear(slope):
    x = Program("feature", name="measurement")
    return Program("add", children=(Program("const", value=10. - slope),
        Program("mul", children=(Program("const", value=slope), x)))).to_dict()


def _frozen_case(copies=0):
    """All laws fit x=1; four different success regions remain at x=5."""
    measurement = {"instrument": "signal meter",
                   "pattern": r"signal magnitude = ([\d.]+) units", "group": 1}
    estimators = [{"id": "linear", "basis": [Program("feature", name="measurement").to_dict()],
                   "measurements": {"measurement": measurement}}]
    models = [{"program": _linear(slope), "lower": lower, "upper": upper,
               "family": "linear", "source_id": source, "source_question": source + "_response"}
              for source, slope, lower, upper in (
                  ("a", 22.5, 0., 1.), ("b", 22.5, 0., 2.),
                  ("c", 25., 0., .4), ("d", 27.5, 0., .4))]
    # These are different numerical laws and bands, not repeated object IDs.
    # Integer controls cannot distinguish them from c. Finer controls can.
    models += [{"program": _linear(25. + i * .01), "lower": -i * .05,
                "upper": .4 - i * .01, "family": "linear",
                "source_id": "c_copy_" + str(i), "source_question": "copy_response_" + str(i)}
               for i in range(1, copies + 1)]
    state = {"research_table": {
        "references": [{"crystal": 1, "measurements": {"measurement": {"value": 1., "evidence": 1}},
                        "frequency": {"value": 10., "evidence": 1}}],
        "targets": [{"crystal": 5, "measurements": {"measurement": {"value": 5., "evidence": 2}}}]},
        "response_settings": [{"crystal": 1, "frequency": 10., "activated": True, "evidence": [1]}],
        "target_numbers": [5], "completed": [],
        "control_grid": {"step": 1., "origin": 0., "lower": 0., "upper": 10000.},
        "frozen_response_knowledge": {"source_id": "mixed_sources",
            "measurements": {"measurement": deepcopy(measurement)}, "models": models}}
    return state, estimators


def _sources(result):
    candidates = {candidate["id"]: candidate for candidate in result["candidates"]}
    return {hypothesis["id"]: candidates[hypothesis["candidate_id"]]["fit_sources"][0]["source_id"]
            for hypothesis in result["hypotheses"]}


def test_equivalent_numerical_laws_do_not_vote_extra_or_change_physical_choice():
    ordinary, estimators = _frozen_case()
    crowded, _ = _frozen_case(copies=6)
    before = deepcopy(crowded)

    original = reuse_response_knowledge(ordinary, estimators, PATH)
    duplicated = reuse_response_knowledge(crowded, estimators, PATH)

    assert crowded == before
    assert original["selected_number"] == duplicated["selected_number"] == 5
    assert original["selected_setting"] == duplicated["selected_setting"] == 100.
    assert original["selection"]["predicted_separation"] == duplicated["selection"]["predicted_separation"]
    assert original["selection"]["compute_cost"] == duplicated["selection"]["compute_cost"]
    assert len(duplicated["hypotheses"]) == 10
    assert duplicated["selection"]["hypotheses"] == duplicated["hypotheses"]
    assert sorted(map(len, duplicated["selection"]["forecast_classes"])) == [1, 1, 1, 7]
    identifiers = {hypothesis["id"] for hypothesis in duplicated["hypotheses"]}
    assert all(set(experiment["predictions"]) == identifiers for experiment in duplicated["experiments"])
    assert set(duplicated["selection"]["selected_experiment"]["predictions"]) == identifiers
    # Demonstrate the old defect on these very same real, compatible models.
    legacy = select_experiment({"hypotheses": duplicated["hypotheses"],
                                "experiments": duplicated["experiments"]})
    assert legacy["selected_experiment"]["prediction"] == 110.


def test_new_executable_measurements_split_a_previous_forecast_class():
    state, estimators = _frozen_case(copies=6)
    coarse = reuse_response_knowledge(state, estimators, PATH)
    state["control_grid"]["step"] = .1

    finer = reuse_response_knowledge(state, estimators, PATH)

    assert len(coarse["selection"]["forecast_classes"]) == 4
    assert len(finer["selection"]["forecast_classes"]) > 4
    assert finer["hypotheses"] == coarse["hypotheses"]
    sources = _sources(finer)
    classes = [{sources[identifier] for identifier in group}
               for group in finer["selection"]["forecast_classes"]]
    assert not any({"c", "c_copy_6"} <= group for group in classes)


def test_paid_refutation_distinguishes_sources_that_previously_shared_a_class():
    state, estimators = _frozen_case(copies=6)
    first = reuse_response_knowledge(state, estimators, PATH)
    sources = _sources(first)
    assert any({"c", "c_copy_6"} <= {sources[identifier] for identifier in group}
               for group in first["selection"]["forecast_classes"])
    archive = deepcopy(state["frozen_response_knowledge"])
    state["control_grid"]["step"] = .1
    state["response_settings"].append({"crystal": 5, "frequency": 110.5,
                                       "activated": False, "evidence": [3]})

    result = reuse_response_knowledge(state, estimators, PATH)

    counts = result["frozen_transfer"]["source_counts"]
    assert counts["c"]["retained_count"] == 1
    assert counts["c_copy_3"]["retained_count"] == 1
    assert all(counts["c_copy_" + str(i)]["rejected_count"] == 1 for i in (4, 5, 6))
    assert result["frozen_transfer"]["retained_count"] == 7
    assert result["frozen_transfer"]["rejected_count"] == 3
    assert state["frozen_response_knowledge"] == archive
    assert 3 in result["evidence"]


def test_single_forecast_class_can_confirm_an_unfinished_target():
    state, estimators = _frozen_case(copies=6)
    state["frozen_response_knowledge"]["models"] = [model
        for model in state["frozen_response_knowledge"]["models"] if model["source_id"].startswith("c")]

    result = reuse_response_knowledge(state, estimators, PATH)

    assert result["status"] == "frozen_target_confirmation"
    assert result["selected_setting"] == 110.
    assert result["selection"]["predicted_separation"] == 0.
    assert len(result["selection"]["forecast_classes"]) == 1
    assert len(result["selection"]["hypotheses"]) == 7
    assert all(result["selection"]["selected_experiment"]["predictions"].values())


def test_single_forecast_class_without_target_support_or_contrast_stops():
    state, estimators = _frozen_case(copies=6)
    state["frozen_response_knowledge"]["models"] = [model
        for model in state["frozen_response_knowledge"]["models"] if model["source_id"].startswith("c")]
    state["control_grid"].update(lower=200., upper=210.)
    state["control_grids"] = {"1": {"step": 1., "origin": 0., "lower": 0., "upper": 100.}}

    result = reuse_response_knowledge(state, estimators, PATH)

    assert result["status"] == "no_untried_supported_setting"
    assert result["frozen_transfer"]["status"] == "exhausted"
    assert result["selection"] is None
    assert result["selected_setting"] is None
    assert len(result["hypotheses"]) == 7
    assert result["experiments"]
    assert all(not any(experiment["predictions"].values()) for experiment in result["experiments"])


def test_existing_panel_contrast_order_keeps_its_physical_coverage():
    pool = alternatives()
    crowded = deepcopy(pool)
    for experiment in crowded:
        experiment["predictions"].update({"copy_" + str(i): experiment["predictions"]["member_0"]
                                          for i in range(10)})

    ordered, cost = _contrast_order(pool)
    repeated, repeated_cost = _contrast_order(crowded)

    assert [item["prediction"] for item in ordered[:4]] == [10, 40, 200, 900]
    assert [item["id"] for item in repeated] == [item["id"] for item in ordered]
    assert {item["id"] for item in ordered} == {item["id"] for item in pool}
    assert repeated_cost == cost
