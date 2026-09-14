"""A new executable proposal can reopen research after a measured zero return."""

import json

from hegel_machine.cognition.research_opportunities import opportunity_condition
from hegel_machine.cognition.session import ResearchSession
from test_cognition_session import MeasuredTask, config


def test_paid_new_knowledge_reopens_a_stopped_question_but_numeric_redelivery_does_not():
    target = MeasuredTask("unresolved", nonlinear=True)
    session = ResearchSession(config(learn_research_policy=False), [target],
        observation_budget=500, compute_budget=100000, max_visits=6)
    stopped = session.run()
    assert stopped["tasks"][0]["visits"] == 2
    assert stopped["pending"] == []
    old_returns = stopped["knowledge"]["research_opportunities"]["outcomes"]
    assert len(old_returns) == 1 and old_returns[0]["gain"] == 0.
    assert old_returns[0]["condition"]["knowledge_alternatives"] == []

    source = MeasuredTask("linear")
    session.add_environment(source)
    learned = session.run(max_steps=1)
    experiment = next(event for event in reversed(learned["scheduler_events"])
                      if event["reason"] == "try_new_knowledge")
    assert experiment["useful_candidates"] > 0 and experiment["compute_cost"] > 0
    assert learned["pending"] == [0]
    offer = learned["tasks"][0]["next_research"]
    assert offer["kind"] == "revision" and offer["expected_gain"] > 0.
    assert offer["knowledge_alternatives"]
    assert learned["tasks"][0]["revisit_basis"] == "new_knowledge"
    assert "linear" not in json.dumps(offer["knowledge_alternatives"])

    restored = ResearchSession(session.config, [target, source],
        observation_budget=500, compute_budget=100000, max_visits=6,
        state=json.loads(json.dumps(learned)))
    restored_offer = restored.to_dict()["tasks"][0]["next_research"]
    assert opportunity_condition(restored_offer) == opportunity_condition(offer)
    prior_queries = target.observed
    continued = restored.run(max_steps=1)
    episode = continued["episodes"][-1]
    assert episode["scheduler"]["reason"] == "new_knowledge"
    assert episode["task_index"] == 0 and episode["visit"] == 3
    assert target.observed - prior_queries == episode["observation_cost"] > 0
    assert episode["compute_cost"] > 0
    assert episode["scheduler"]["research_return"]["gain"] == 0.
    assert continued["pending"] == []
    assert "new_knowledge_alternatives" not in continued["tasks"][0]

    # Another actual linear source changes fitted numerical parameters. Its
    # target adaptation is still paid and finds useful fits, but it contributes
    # only structures already tried in the unproductive continuation.
    restored.add_environment(MeasuredTask("different_linear", slope=2.))
    repeated = restored.run(max_steps=1)
    trial = next(event for event in reversed(repeated["scheduler_events"])
                 if event["reason"] == "try_new_knowledge" and event["task_index"] == 0)
    assert trial["source_task_index"] == 2 and trial["useful_candidates"] > 0
    assert trial["compute_cost"] > 0
    assert repeated["tasks"][0]["visits"] == 3 and repeated["pending"] == []
    assert repeated["tasks"][0]["revisit_basis"] == "no_observed_continuation_gain"
    prior_shapes = episode["scheduler"]["research_return"]["condition"]["knowledge_alternatives"]
    assert all(shape in prior_shapes for shape in repeated["tasks"][0]["new_knowledge_alternatives"])
