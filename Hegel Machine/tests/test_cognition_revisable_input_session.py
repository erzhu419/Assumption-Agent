"""Actual acquired evidence separates calibration repair from mechanism change."""

import json
from math import isfinite

import pytest

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession


class ObservedNoiseResearcher(UnifiedResearchAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_evidence = []

    def _support(self, program, training, validation, spec, budget):
        self.noise_evidence.append((program, list(training), list(validation), spec))
        return super()._support(program, training, validation, spec, budget)


def check_inherited_noise_method_uses_paired_commands(knowledge, config, source):
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(knowledge)))
    prior = next(prior for prior in library.meta_priors if prior["source_concept_id"] == source)
    # This is an actual persisted v1 condition: a systematic model mismatch was
    # once accepted as noise. Its successor must acquire discriminating fields.
    prior["research_method"]["noise_supported"] = True
    researcher = ObservedNoiseResearcher(config, library=library, measurement_budget=4608)
    with UnifiedObjectLab(seed=80127, objects=4, readout="window_total_event", variant="nonlocal",
                          apparatus_seed=9013, actuator_curvature=.39, rename_prefix="noisy_successor_") as lab:
        report = researcher.run(lab)
        commands, measured = set(lab.spec.feature_names), set(lab.spec_for(("fine",)).feature_names)
        trials = [event for event in report["actions"] if event["action"] == "test_learned_acquisition_method"]
        assert trials and trials[0]["pilot_channels"] == ["fine"]
        assert trials[0]["selected_channels"] == []
        assert researcher.noise_evidence
        program, training, validation, spec = researcher.noise_evidence[0]
        assert set(spec.feature_names) == commands | measured
        assert {name for name, _ in program.observed_variables()} <= commands | {"__target__"}

        paid = []
        for batch in report["batches"]:
            if batch["channels"] == ["fine"]:
                sequence = batch["probe"]["sequence"]
                paid.extend((row, point, list(sequence[:index]))
                            for index, (row, point) in enumerate(zip(batch["observations"], sequence)))
        for index, row in enumerate([*training, *validation]):
            assert set(row.features) == commands | measured
            assert isfinite(program.predict(row))
            matching = [(point, history) for acquired, point, history in paid
                        if row.target == acquired["target"] and
                        {name: row.features[name] for name in measured} == acquired["features"]]
            observed_commands = {name: row.features[name] for name in commands}
            observed_history = [{name: past[name] for name in commands} for past in row.history]
            assert matching, (index, "measured row absent from paid batch")
            assert any(observed_commands == point for point, _ in matching), (index, "command mismatch", observed_commands, matching)
            assert any(observed_commands == point and observed_history == history for point, history in matching), (
                index, "history mismatch", len(observed_history), observed_history, matching)
        measured_loss = sum((program.predict(row) - row.target) ** 2 for row in validation) / len(validation)
        assert trials[0]["validation_loss"] == pytest.approx(measured_loss, abs=1e-10)


def test_window_reuses_its_mechanism_after_calibration_change_but_rejects_a_changed_relation():
    common = {"environment": "objects", "objects": 4, "variant": "exchange", "apparatus_seed": 9013}
    tasks = [{**common, "seed": seed, "readout": readout}
             for seed, readout in ((10078, "total"), (20085, "event"),
                                   (30092, "total_event"), (50106, "window_total_event"))]
    tasks.extend([
        {**common, "seed": 60113, "readout": "window_total_event", "actuator_curvature": .39,
         "rename_prefix": "recalibrated_"},
        {**common, "seed": 70120, "readout": "window_total_event", "actuator_curvature": .39,
         "variant": "sink", "rename_prefix": "mechanism_"},
    ])
    config = ResearchConfig(observation_budget=384, compute_budget=360000, max_candidates=16,
        max_rounds=6, minimum_improvement=1e-7, seed=71, active_probes=True)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=4608)
    prefix = session.run(max_tasks=4)
    window = prefix["results"][-1]["research"]
    assert all(row["research"]["confirmation_loss"] < 1e-7 for row in prefix["results"])
    assert window["selected_kind"] == "research_program"
    source = window["concepts_added"][0]
    old_prior = json.loads(json.dumps(next(prior for prior in prefix["knowledge"]["meta_priors"]
                                          if prior["source_concept_id"] == source)))
    assert old_prior["input_research_program"]["input_template"]

    restored = UnifiedResearchSession(state=json.loads(json.dumps(prefix)))
    result = restored.run()
    calibration, relation = [row["research"] for row in result["results"][-2:]]
    distinction = lambda record: [event for event in record["actions"]
        if event["action"] == "distinguish_input_and_mechanism_revision"]

    assert calibration["confirmation_loss"] < 1e-7
    assert calibration["selected_kind"] == "input_dependency_revision"
    assert calibration["method_source_ids"] == [source]
    assert calibration["selected_channels"] == []
    assert distinction(calibration)[-1]["intact_mechanism_found"]
    assert any(event["action"] == "instrument_mapping_counterexample" for event in calibration["actions"])
    saved = next(prior for prior in result["knowledge"]["meta_priors"]
                 if prior["source_concept_id"] == source)
    assert saved["research_program"] == old_prior["research_program"]
    assert saved["input_research_program"] == old_prior["input_research_program"]

    # The succeeding apparatus preserves the newly calibrated instruments, but
    # changes the response relation. It must reopen mechanism research instead
    # of adopting the old window as an instrument-only repair.
    assert distinction(relation)
    assert not any(event["intact_mechanism_found"] for event in distinction(relation))
    assert relation["selected_kind"] != "input_dependency_revision"
    assert not any(event["action"] == "instrument_mapping_counterexample" for event in relation["actions"])
    assert relation["costs"]["measurement_cost"] > 0
    # Fork the earlier saved judgment, before later mechanism research can
    # replace it with a correct explanation and make this noise path dormant.
    check_inherited_noise_method_uses_paired_commands(prefix["knowledge"], config, source)
