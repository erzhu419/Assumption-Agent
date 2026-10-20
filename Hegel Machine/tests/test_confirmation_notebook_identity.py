"""A fresh test belongs to its frozen predictor, including after source replacement."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_resume import SEED, agent_at, checkpoint, execute


@pytest.mark.parametrize("restore", [False, True])
def test_notebook_keeps_two_confirmed_predictors_distinct_from_current_fit(tmp_path, restore):
    agent, lab, _ = checkpoint(tmp_path)
    # A second exact paid-data fit still has a different untested prediction.
    # Its failed confirmation is executed, not inserted into the fixture.
    source = (
        "import json\n"
        "_known = {}\n"
        "def key(row):\n"
        "    return json.dumps([row['features'], row.get('history', [])], sort_keys=True)\n"
        "def fit(rows):\n"
        "    global _known\n"
        "    _known = {key(row): row['target'] for row in rows}\n"
        "def predict(row): return _known.get(key(row), 1000000.)\n"
    )
    execute(agent, lab, "evaluate", {"model": "second.py", "code": source})
    outcome = execute(agent, lab, "finish", {"model": "second.py"})
    assert outcome["fresh_confirmation"]["passed"] is False
    submitted = [deepcopy(item["submitted_candidate"]) for item in agent.confirmations]
    frozen_sources = [(agent.workspace / item["model"]).read_text() for item in submitted]

    # Source paths can be overwritten by ongoing research; the two old tests
    # must still identify the frozen sources from before their new labels.
    execute(agent, lab, "evaluate", {"model": "second.py", "code": source + "\n# current draft\n"})
    expected_current = deepcopy(agent._evaluated_summary())
    expected_costs = deepcopy(agent.costs)
    if restore:
        previous = agent.result("model_call_budget_exhausted")
        resumed, target = agent_at(tmp_path / "restored"), TemporalInquiryLab(task_index=1)
        restored = resumed.restore_research(previous, agent.workspace, target, confirmation_seed=SEED)
        assert restored["new_observations"] == 0
        agent, lab = resumed, target

    facts = agent._request(lab)["evidence_notebook"]["facts"]
    assert [facts[f"confirmation:{i}"]["submitted_candidate"] for i in range(2)] == submitted
    assert submitted[0]["source_model"] == "model.py"
    assert submitted[1]["source_model"] == "second.py"
    assert submitted[0]["model"] != submitted[1]["model"]
    assert submitted[0]["paid_rows"] < submitted[1]["paid_rows"] < len(agent.rows)
    assert [(agent.workspace / item["model"]).read_text() for item in submitted] == frozen_sources
    assert (agent.workspace / "second.py").read_text() != frozen_sources[1]
    assert facts["current_paid_evaluation"]["evaluated_rows"] == len(agent.rows)
    assert agent._evaluated_summary() == expected_current
    assert agent.costs == expected_costs
    assert "observations" not in facts["confirmation:0"]
    assert "observations" not in facts["confirmation:1"]
