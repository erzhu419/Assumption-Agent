"""Restarted composition fitting must receive the counterexamples already bought."""

from dataclasses import replace

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for


class RecordingResearcher(UnifiedResearchAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.program_training = []

    def _program_claims(self, spec, training, *args, **kwargs):
        self.program_training.append(tuple(training))
        return super()._program_claims(spec, training, *args, **kwargs)


def test_return_research_remeasures_its_actual_counterexample_and_reuses_the_corrected_composition():
    # This is a regression from an already observed failure, not a new apparatus
    # used to estimate performance. The unchanged outer window is learned first.
    seed = 73070884189
    old_radius, new_radius = .7497007917150603, 1.0893236590790407
    tasks = [{"environment": "objects", "objects": 4, "apparatus_seed": seed,
        "seed": seed + 10007 * position, "profile": "confirmation", "variant": "exchange",
        "readout": readout, "relation_radius": new_radius if position == 4 else old_radius}
        for position, readout in enumerate(("total", "event", "total_event",
            "window_total_event", "window_total_event", "window_total_event"))]
    config = ResearchConfig(observation_budget=512, compute_budget=480000,
        minimum_improvement=1e-7, max_rounds=6, max_history=4, active_probes=True)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=6144)
    prefix = session.run(max_tasks=5)
    definitions = dict(session.library.definitions)
    researcher = RecordingResearcher(replace(config, seed=tasks[5]["seed"]),
        library=session.library, measurement_budget=6144)
    with laboratory_for(tasks[5]) as lab:
        report = researcher.run(lab)
        program = Program.from_dict(report["program"])
        fresh = lab.acquire(lab.random_probe(92341, length=64), report["selected_channels"]).observations
        assert sum((program.predict(row) - row.target) ** 2 for row in fresh) / len(fresh) < 1e-7
    ids = {action["probe_id"] for action in report["actions"]
           if action["action"] == "acquire" and action["purpose"] == "distinguish_transfer_conditions"}
    directed = [Observation(row["features"], row["target"], tuple(row["history"]))
        for batch in report["batches"] if batch["probe"]["probe_id"] in ids
        for row in batch["observations"]]
    assert len(directed) == 18 and any(row.history for row in directed)
    assert all(row in researcher.program_training[0] for row in directed)
    checked = [Observation(row["features"], row["target"], tuple(row["history"]))
        for row in report["batches"][1]["observations"]]
    assert not any(row in researcher.program_training[0] for row in checked)
    assert max(abs(program.predict(row) - row.target) for row in directed) < 1e-7
    acquisition = next(action for action in report["actions"]
        if action["action"] == "acquire" and action["purpose"] == "distinguish_instrument_boundary")
    measured = next(batch for batch in report["batches"] if batch["probe"]["probe_id"] == acquisition["probe_id"])
    assert any(batch["probe"]["sequence"] == measured["probe"]["sequence"]
               for batch in report["batches"] if batch["probe"]["probe_id"] in ids)
    assert len(measured["observations"]) == 9
    rejected = next(action for action in report["actions"] if action["action"] == "instrument_mapping_counterexample")
    repaired = next(action for action in report["actions"] if action["action"] == "research_instrument_mapping")
    assert set(rejected["field_losses"]) == {"measured_f_o1"}
    assert rejected["rows"] == repaired["fit_rows"] + repaired["validation_rows"] == 57
    assert repaired["fit_rows"] == 33 and repaired["validation_rows"] == 24
    assert 0 < report["costs"]["measurement_cost"] < 1440
    assert report["costs"]["observation_cost"] < 218
    assert report["confirmation_loss"] < 1e-7
    assert set(prefix["results"][3]["research"]["concepts_added"]) & set(report["method_source_ids"])
    assert {source for index in (0, 1) for source in prefix["results"][index]["research"]["concepts_added"]} <= set(report["used_sources"])
    assert all(session.library.definitions[name] == definition for name, definition in definitions.items())
