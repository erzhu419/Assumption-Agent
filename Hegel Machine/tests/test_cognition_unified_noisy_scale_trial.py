"""A coarse inherited noise model must still face newly paid fine instruments."""

from dataclasses import replace
import json
from math import isfinite
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class NoisyTotalLab(UnifiedObjectLab):
    def __init__(self, *, risk_quantile=.5, **kwargs):
        self.risk_quantile = risk_quantile
        super().__init__(**kwargs)

    def spec_for(self, channels=()):
        return replace(super().spec_for(channels), risk_quantile=self.risk_quantile)

    def acquire(self, probe, channels=()):
        batch = super().acquire(probe, channels)
        random = Random(1033 + 7919 * len(self.acquisitions))
        rows, history = [], []
        for row in batch.observations:
            target = row.target + random.uniform(-.08, .08)
            rows.append(Observation(row.features, target, tuple(history)))
            history.append({**row.features, "__target__": target})
        return replace(batch, observations=rows)


class ObservedNoiseResearcher(UnifiedResearchAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_evidence = []

    def _support(self, program, training, validation, spec, budget):
        self.noise_evidence.append((program, list(training), list(validation), spec))
        return super()._support(program, training, validation, spec, budget)


@pytest.mark.parametrize("quantile", [.5, .9])
def test_coarse_short_trial_projects_comparisons_but_tests_noise_on_fine_measurements(quantile):
    source = NoisyTotalLab(readout="total", apparatus_seed=9013, risk_quantile=quantile)
    spec = source.spec_for(("coarse",))
    object_schema = json.loads(spec.context["object_schema"])
    fields = object_schema["fields"][object_schema["objects"][0]]
    mean = Program("add", children=(Program("feature", name=fields["q"]),
                                     Program("feature", name=fields["u"])))
    if quantile != .5:
        mean = Program("add", children=(mean, Program("const", value=.08 * (2 * quantile - 1))))
    library = KnowledgeLibrary()
    rows = source.observe(source.random_probe(101, length=32), ("coarse",))
    source_id, _ = library.learn(Candidate("inherited_noisy_mean", mean, "refit"), spec, rows)
    library.meta_priors.append({"source_concept_id": source_id,
        "object_schema": object_schema, "source_features": list(spec.feature_names),
        "control_features": list(spec.feature_bounds), "counterexamples": [], "uses": 0,
        "research_method": {"fit_rows": 32, "validation_rows": 32,
                            "confirmation_rows": 32, "noise_supported": True}})
    lab = NoisyTotalLab(readout="total", apparatus_seed=9013, rename_prefix="later_", risk_quantile=quantile)
    config = ResearchConfig(observation_budget=384, compute_budget=360000, max_rounds=6,
                            minimum_improvement=1e-7, active_probes=True, seed=70120)
    researcher = ObservedNoiseResearcher(config, measurement_budget=4608, library=library)
    report = researcher.run(lab)
    trials = [action for action in report["actions"] if action["action"] == "test_learned_acquisition_method"]
    assert trials
    if quantile == .5:
        assert trials[0]["selected_channels"] == ["coarse"]
    assert trials[0]["validation_channels"] == ["fine"]
    assert researcher.noise_evidence
    for program, training, validation, evidence_spec in researcher.noise_evidence:
        assert evidence_spec.context["measurement_view"] == "fine"
        assert len(evidence_spec.feature_names) == 16
        assert all(set(row.features) == set(evidence_spec.feature_names) for row in [*training, *validation])
        projected = lab.project_rows(validation, ("fine",), ("coarse",))
        assert len(projected[0].features) == 4
        assert all(name.startswith("measured_later_") for name in program.required_features())
        assert all(isfinite(program.predict(row)) for row in validation)
    assert any(action["action"] == "acquire" and action["purpose"] == "confirm_short_frozen_program"
               for action in report["actions"])
    if quantile != .5:
        assert any(action["action"] == "propose_tail_calibration" for action in report["actions"])
