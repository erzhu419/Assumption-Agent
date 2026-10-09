"""A paid child investigation must change an independently correct parent model."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.knowledge_evolution import bind_answers
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_research import run_temporal_research


def public_laboratory(lab):
    """The controller receives only the same interface available to an agent."""
    class PublicLab:
        public_spec = staticmethod(lab.public_spec)
        execute = staticmethod(lab.execute)
        confirmation_actions = staticmethod(lab.confirmation_actions)

        @property
        def costs(self):
            return lab.costs

    return PublicLab()


def mse(model, rows):
    # Discard targets before prediction as well as keeping meter readings out.
    return sum((model.predict(Observation(dict(row.features), 0., row.history))
                - row.target) ** 2 for row in rows) / len(rows)


@pytest.fixture(scope="module", params=["finite2", "recurrent"])
def completed_research(request):
    mode = request.param
    source = TemporalInquiryLab(task_index=0, apparatus_seed=910051,
                                instrument_mode=mode)
    target = TemporalInquiryLab(task_index=1, apparatus_seed=910051,
                                instrument_mode=mode)
    result = run_temporal_research(public_laboratory(source), public_laboratory(target),
                                   seed=71, tolerance=1e-10)
    return mode, source, target, result


def test_paid_research_confirms_and_predicts_unseen_controls(completed_research):
    mode, source, target, result = completed_research
    assert result.resultdict["source_confirmation_mse"] < 1e-10
    assert result.resultdict["final_confirmation_mse"] < 1e-10
    assert source.costs["observation_cost"] == 48
    assert target.costs["observation_cost"] <= 240
    assert result.resultdict["costs"] == {"source": source.costs, "target": target.costs}
    assert sum(record["observation_cost"] for record in result.records) == target.costs["observation_cost"]

    # The terminal sample is created only after research returns, outside the
    # public interface; no target or optional measurement is needed to predict.
    frozen = deepcopy(result.model.to_dict())
    source_rows = source.evaluation_rows(length=256, seed=910052)
    target_rows = target.evaluation_rows(length=256, seed=910053)
    assert mse(result.source_model, source_rows) < 1e-10
    assert mse(result.model, target_rows) < 1e-10
    assert result.model.to_dict() == frozen
    assert all(set(row.features) == {"u"}
               and all(set(point) == {"u"} for point in row.history)
               for row in target_rows)
    assert mse(result.source_model, target_rows) > 1e-6


def test_instrument_question_is_live_and_changes_parent(completed_research):
    mode, source, target, result = completed_research
    events = result.resultdict["events"]
    names = [event["event"] for event in events]
    if mode == "finite2":
        assert "history_revised" in names
        assert "child_opened" not in names
        assert target.costs["measurement_cost"] == 0
        assert not any(record["measurements"] for record in result.records)
        return

    def position(name, **attributes):
        return next(index for index, event in enumerate(events)
                    if event["event"] == name
                    and all(event.get(key) == value for key, value in attributes.items()))

    opened = position("child_opened", question_id="instrument")
    seed_query = position("query", phase="instrument_seed")
    distinguishing_query = position("query", phase="instrument_disagreement")
    answered = position("child_answered")
    revised = position("parent_revised")
    confirmed = position("confirmed")
    confirmation = next(index for index, event in enumerate(events)
                        if index > revised and event["event"] == "query"
                        and event.get("phase") == "confirmation")
    assert position("scope_counterexample") < opened
    assert position("history_revised") < opened
    assert opened < seed_query < distinguishing_query < answered < revised < confirmation < confirmed
    assert target.costs["measurement_cost"] > 0
    assert any(record["measurements"] for record in result.records)
    independent = target.evaluation_rows(length=256, seed=910054)
    assert mse(result.history_model, independent) > 1e-6
    assert mse(result.model, independent) < 1e-10
    assert result.model.to_dict() != result.history_model.to_dict()

    nodes = result.resultdict["research_graph"]["nodes"]
    child, parent = nodes["instrument"], nodes["response"]
    assert child["status"] == "answered" and child["observations"]
    assert parent["status"] == "confirmed" and parent["depends_on"] == ["instrument"]
    assert parent["answer"] == result.model.to_dict()
    expression = Program.from_dict(parent["expression"])
    wrong_answer = bind_answers(expression, {"instrument": Program("const", value=0.)})
    assert mse(wrong_answer, independent) > 1e-6

