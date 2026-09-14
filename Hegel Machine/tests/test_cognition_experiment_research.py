"""Paid source learning leads to an executed, rebound experimental procedure."""

from hegel_machine.cognition.experiment_research import run_experiment_transfer
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation


def test_learned_competing_explanations_replay_an_experiment_and_follow_the_new_outcome():
    library = KnowledgeLibrary()
    result = run_experiment_transfer(library)
    assert all(episode["observation_cost"] > 0 and episode["compute_cost"] > 0 for episode in result["sources"])
    source_programs = [Program.from_dict(episode["program"]) for episode in result["sources"]]
    fresh = Observation({"x": .37}, 0.)
    assert abs(source_programs[0].predict(fresh) - .37) < 1e-8
    assert abs(source_programs[1].predict(fresh) - .37 ** 2) < 1e-8

    learning, transfer = result["learning"], result["transfer"]
    assert len(learning["initial_classes"]) == len(transfer["initial_classes"]) == 1
    assert len(learning["initial_classes"][0]) == 2
    assert len(learning["after_classes"]) == len(transfer["after_classes"]) == 2
    assert learning["events"][0]["action"] == "execute_new_experiment"
    assert transfer["events"][0]["action"] == "execute_recalled_experiment"
    old_probe = learning["events"][0]["probe"]
    new_probe = transfer["events"][0]["probe"]
    assert transfer["recall"][0]["role_mapping"] == {"x": "u"}
    assert [point["u"] for point in new_probe["sequence"]] == [2 * point["x"] for point in old_probe["sequence"]]
    assert learning["selected_explanation"] != transfer["selected_explanation"]
    acquired = transfer["events"][0]["observations"]
    assert all(row["target"] == row["features"]["u"] for row in acquired)
    queried = transfer["episode"]["research_state"]["queried"]
    assert any(probe["probe_id"] == new_probe["probe_id"] for probe in queried)
    assert any(action["action"] == "resume_research" for action in transfer["episode"]["actions"])
    assert result["summary"]["actual_recalled_queries"] == 1
    assert transfer["evaluation"]["mse"] < 1e-8 and transfer["evaluation"]["rows"] == 64
    assert result["summary"]["observation_cost"] > 0 and result["summary"]["compute_cost"] > 0
    assert any(record["uses"] > 0 for record in library.experiments.records)
