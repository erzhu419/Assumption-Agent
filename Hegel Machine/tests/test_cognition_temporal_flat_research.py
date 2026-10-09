"""The flat comparator has the same experiments and models, without a child gate."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.knowledge_evolution import bind_answers
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_flat_research import run_flat_temporal_research
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_research import (
    fit_temporal_response,
    mse as response_mse,
    run_temporal_research,
)


def public_laboratory(lab, *, confirmation_shift=0.):
    """Expose only the paid research interface; optionally change fresh labels."""
    class PublicLab:
        confirming = False
        public_spec = staticmethod(lab.public_spec)

        def execute(self, action):
            record = lab.execute(action)
            if self.confirming and confirmation_shift:
                record = deepcopy(record)
                for row in record["rows"]:
                    row["target"] += confirmation_shift
            return record

        def confirmation_actions(self, count, rng):
            self.confirming = True
            return lab.confirmation_actions(count, rng)

        @property
        def costs(self):
            return lab.costs

    return PublicLab()


def mse(model, rows):
    return sum((model.predict(Observation(dict(row.features), 0., row.history))
                - row.target) ** 2 for row in rows) / len(rows)


@pytest.fixture(scope="module", params=["finite2", "recurrent"])
def paired_research(request):
    mode = request.param
    runs = {}
    for name, controller in [("flat", run_flat_temporal_research),
                             ("recursive", run_temporal_research)]:
        source = TemporalInquiryLab(task_index=0, apparatus_seed=910051,
                                    instrument_mode=mode)
        target = TemporalInquiryLab(task_index=1, apparatus_seed=910051,
                                    instrument_mode=mode)
        result = controller(public_laboratory(source), public_laboratory(target),
                            seed=71, tolerance=1e-10)
        runs[name] = source, target, result
    return mode, runs


def test_flat_confirms_with_only_public_data_and_predicts_new_trajectories(paired_research):
    _, runs = paired_research
    source, target, result = runs["flat"]
    assert result.resultdict["status"] == "confirmed"
    assert result.resultdict["source_confirmation_mse"] < 1e-10
    assert result.resultdict["final_confirmation_mse"] < 1e-10
    assert result.resultdict["costs"] == {"source": source.costs, "target": target.costs}
    assert source.costs["observation_cost"] == 48
    assert target.costs["observation_cost"] <= result.resultdict["limits"]["target_rows"]
    assert sum(record["observation_cost"] for record in result.records) == target.costs["observation_cost"]

    frozen = deepcopy(result.model.to_dict())
    assert mse(result.source_model, source.evaluation_rows(length=256, seed=910052)) < 1e-10
    assert mse(result.model, target.evaluation_rows(length=256, seed=910053)) < 1e-10
    assert result.model.to_dict() == frozen


def test_flat_and_recursive_receive_identical_initial_experiments_and_limits(paired_research):
    _, runs = paired_research
    flat, recursive = (runs[name][2] for name in ("flat", "recursive"))
    assert flat.resultdict["limits"] == recursive.resultdict["limits"]
    shared_phases = {"source_train", "source_confirmation", "scope_test", "instrument_seed"}

    def initial_queries(result):
        return [(event["phase"], event["action"], event["observation_cost"],
                 event["measurement_cost"])
                for event in result.resultdict["events"]
                if event["event"] == "query" and event["phase"] in shared_phases]

    assert initial_queries(flat) == initial_queries(recursive)
    assert flat.source_model.to_dict() == recursive.source_model.to_dict()
    assert flat.history_model.to_dict() == recursive.history_model.to_dict()


def test_flat_compares_parent_predictions_without_a_child_confirmation_gate(paired_research):
    mode, runs = paired_research
    _, target, result = runs["flat"]
    events = result.resultdict["events"]
    assert "research_graph" not in result.resultdict
    assert not any(event["event"] in {"child_opened", "child_answered"} for event in events)
    comparisons = [event for event in events if event["event"] == "flat_candidate_comparison"]
    if mode == "finite2":
        assert not comparisons
        assert target.costs["measurement_cost"] == 0
        assert not any(record["measurements"] for record in result.records)
        return

    comparison = comparisons[0]
    assert comparison["training_rows"] == 63
    assert comparison["validation_rows"] == 48
    assert len(comparison["candidates"]) >= 2
    selected = next(candidate for candidate in comparison["candidates"]
                    if candidate["hypothesis_id"] == comparison["selected_hypothesis"])
    assert selected["validation_mse"] < 1e-10
    assert any(candidate["validation_mse"] > 1e-6 for candidate in comparison["candidates"])
    # The response may be determined while the short measured traces retain
    # multiple meter explanations. Requiring a unique child would weaken flat.
    hypotheses = result.resultdict["instrument_model"]["fields"]["meter"]["hypotheses"]
    assert sum(item["status"] not in {"refuted", "counterexample"} for item in hypotheses) > 1
    assert target.costs["measurement_cost"] > 0
    assert not any(event["event"] == "query" and event["phase"] == "instrument_disagreement"
                   for event in events)

    # Reconstruct candidate fits from the declared paid split. Accidentally
    # fitting the 48 validation labels changes the wrong candidate's losses.
    train_records = result.records[:2] + result.records[4:9]
    validation_records = result.records[2:4]
    training_indices = {index for record in train_records for index in record["row_indices"]}
    validation_indices = {index for record in validation_records for index in record["row_indices"]}
    assert training_indices.isdisjoint(validation_indices)
    assert all(not record["rows"][0]["history"] for record in train_records + validation_records)
    train_rows = [row for record in train_records for row in record["rows"]]
    heldout_rows = [row for record in validation_records for row in record["rows"]]
    by_id = {hypothesis["hypothesis_id"]: hypothesis for hypothesis in hypotheses}
    for candidate in comparison["candidates"]:
        instrument = Program.from_dict(by_id[candidate["hypothesis_id"]]["program"])
        expression, rank, width = fit_temporal_response(train_rows, instrument)
        predictor = bind_answers(expression, {"instrument": instrument})
        assert (candidate["training_rank"], candidate["feature_count"]) == (rank, width)
        assert candidate["train_mse"] == pytest.approx(response_mse(predictor, train_rows), abs=1e-20)
        assert candidate["validation_mse"] == pytest.approx(response_mse(predictor, heldout_rows), abs=1e-20)


def test_fresh_confirmation_rejects_changed_response_without_refitting():
    source = TemporalInquiryLab(task_index=0, apparatus_seed=910051, instrument_mode="finite2")
    target = TemporalInquiryLab(task_index=1, apparatus_seed=910051, instrument_mode="finite2")
    result = run_flat_temporal_research(public_laboratory(source),
        public_laboratory(target, confirmation_shift=2.), seed=71, tolerance=1e-10)
    assert result.resultdict["status"] == "confirmation_rejected"
    assert result.resultdict["final_confirmation_mse"] == pytest.approx(4.)
    assert mse(result.model, target.evaluation_rows(length=256, seed=910055)) < 1e-10
    confirmation_rows = sum(event["observation_cost"] for event in result.resultdict["events"]
                            if event["event"] == "query" and event["phase"] == "confirmation")
    assert confirmation_rows == 48
    assert result.resultdict["fit_rows"] + confirmation_rows == target.costs["observation_cost"]
