"""Prospective, bounded research of a response whose experimental scope expands.

The procedure first revises control history. Persistent paid-data error opens
an instrument child; its separately measured answer becomes a coordinate of the
parent response. This is an implemented research algorithm, not an LLM-authored
procedure. The controller receives only the public laboratory interface.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from random import Random

import numpy as np

from .history_revision import revise_history
from .knowledge_evolution import bind_answers
from .language import Program
from .models import Observation
from .research_graph import ResearchGraph
from .temporal_instrument_research import (
    assess_temporal_instrument, fit_temporal_instrument,
    select_temporal_instrument_probe, temporal_instrument_program,
)


TEMPORAL_LIMITS = {"source_rows": 48, "target_rows": 240,
                   "fresh_confirmation_rows": 48, "measurement_units": 720,
                   "instrument_compute_units": 100000}


def observation(row):
    return (Observation(dict(row["features"]), row["target"], tuple(row["history"]))
            if isinstance(row, dict) else row)


def mse(program, rows):
    rows = [observation(row) for row in rows]
    return float(np.mean([(program.predict(row) - row.target) ** 2 for row in rows]))


def _linear_combination(terms, coefficients):
    parts = [Program("mul", children=(Program("const", value=float(value)), term))
             for term, value in zip(terms, coefficients)]
    while len(parts) > 1:
        parts = [Program("add", children=tuple(parts[index:index + 2]))
                 if index + 1 < len(parts) else parts[index]
                 for index in range(0, len(parts), 2)]
    return parts[0]


def _history_terms(lags):
    raw = [Program("feature", name="u")]
    raw.extend(Program("if", children=(Program("available", lag=lag),
        Program("lag", name="u", lag=lag), Program("const", value=0.)))
        for lag in range(1, lags + 1))
    terms = [Program("const", value=1.)]
    for index, term in enumerate(raw):
        terms.append(term)
        terms.extend(Program("mul", children=(previous, term)) for previous in raw[:index + 1])
    return terms


def fit_history_program(rows, lags):
    """Fixed-basis control for testing representation on the same paid rows."""
    terms = _history_terms(lags)
    observations = [observation(row) for row in rows]
    design = np.asarray([[term.predict(row) for term in terms] for row in observations])
    fitted = np.linalg.lstsq(design, [row.target for row in observations], rcond=None)[0]
    return _linear_combination(terms, fitted)


def _history_program(proposal):
    return _linear_combination(_history_terms(proposal.selected_lags), proposal.coefficients)


def fit_temporal_response(rows, instrument):
    """Fit all quadratic interactions of u, learned meter and previous meter.

    The coefficients choose the response equation; no simulator equation is
    supplied. The returned expression still names the child answer explicitly.
    """
    child = Program("arg", name="instrument")
    previous = Program("if", children=(Program("available", lag=1),
        Program("sub", children=(child, Program("event_delta", lag=1, children=(child,)))),
        Program("const", value=0.)))
    raw = [Program("feature", name="u"), child, previous]
    terms = [Program("const", value=1.), *raw,
             *(Program("mul", children=pair) for pair in combinations_with_replacement(raw, 2))]
    executable = [bind_answers(term, {"instrument": instrument}) for term in terms]
    observations = [observation(row) for row in rows]
    design = np.asarray([[term.predict(row) for term in executable] for row in observations])
    coefficients, _, rank, _ = np.linalg.lstsq(design, [row.target for row in observations], rcond=None)
    return _linear_combination(terms, coefficients), int(rank), len(terms)


@dataclass
class TemporalResearchResult:
    model: Program
    source_model: Program
    history_model: Program
    resultdict: dict
    records: list[dict]


def run_temporal_research(source_lab, target_lab, *, seed=71, tolerance=1e-10):
    """Learn on short probes, revise online, and confirm a frozen final model.

    Fixed development envelope: 48 source rows; at most 240 target rows, of
    which 48 are reserved for fresh response confirmation. No terminal labels
    enter this function. Candidate families and experiment templates are the
    existing bounded instrument/state research repertoire.
    """
    rng = Random(seed)
    events, records = [], []
    graph = ResearchGraph()
    result = {"status": "research_incomplete", "events": events,
              "source_confirmation_mse": None, "final_confirmation_mse": None,
              "instrument_model": None, "instrument_compute_cost": 0, "response_fit_calls": 0,
              "limits": dict(TEMPORAL_LIMITS)}

    def emit(event, **details):
        events.append({"event": event, **details})

    def purchase(lab, action, phase):
        limit = TEMPORAL_LIMITS["target_rows" if lab is target_lab else "source_rows"]
        reserve = TEMPORAL_LIMITS["fresh_confirmation_rows"] if lab is target_lab and phase != "confirmation" else 0
        count = len(action["sequence"])
        if lab.costs["observation_cost"] + count + reserve > limit:
            raise ValueError("research purchase would exceed the common observation allowance")
        meter_cost = sum(channel["cost_per_row"] for channel in lab.public_spec()["channels"]
                         if channel["name"] in action["channels"]) * count
        if lab.costs["measurement_cost"] + meter_cost > TEMPORAL_LIMITS["measurement_units"]:
            raise ValueError("research purchase would exceed the common measurement allowance")
        record = lab.execute(action)
        emit("query", phase=phase, action=record["action"],
             row_indices=record["row_indices"], observation_cost=record["observation_cost"],
             measurement_cost=record["measurement_cost"])
        if lab is target_lab:
            records.append(record)
        return record

    def random_action(length, meter=False):
        return {"kind": "trajectory", "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(length)],
                "channels": ["meter"] if meter else []}

    def finish(model, source_model, history_model):
        result["research_graph"] = graph.to_dict()
        result["costs"] = {"source": dict(source_lab.costs), "target": dict(target_lab.costs)}
        return TemporalResearchResult(model, source_model, history_model, result, records)

    source_rows = [row for _ in range(12)
                   for row in purchase(source_lab, random_action(3), "source_train")["rows"]]
    source_proposal = revise_history(source_rows)
    source_model = _history_program(source_proposal)
    fresh_source = [row for action in source_lab.confirmation_actions(12, Random(seed + 1))
                    for row in purchase(source_lab, action, "source_confirmation")["rows"]]
    result["source_confirmation_mse"] = mse(source_model, fresh_source)
    if result["source_confirmation_mse"] > tolerance:
        result["status"] = "source_rejected"
        return finish(source_model, source_model, source_model)
    emit("source_confirmed", mse=result["source_confirmation_mse"])
    parent = graph.add_question("response", "Predict the response over the current public horizon")
    parent.answer, parent.status = source_model, "answered"

    horizon = target_lab.public_spec()["max_probe_length"]
    for _ in range(4):
        purchase(target_lab, random_action(horizon), "scope_test")
    rows = [row for record in records for row in record["rows"]]
    parent.observations = [observation(row) for row in rows]
    result["frozen_source_target_mse"] = mse(source_model, rows)
    if result["frozen_source_target_mse"] > tolerance:
        emit("scope_counterexample", mse=result["frozen_source_target_mse"])
    proposal = revise_history(rows, holdout_fraction=.5)
    model = history_model = _history_program(proposal)
    result["history_revision"] = proposal.resultdict
    parent.answer = model
    emit("history_revised", status=proposal.resultdict["status"],
         selected_lags=proposal.selected_lags, train_mse=mse(model, rows))
    selected = "revision" if proposal.selected_lags == 3 else "base"
    supported = (proposal.resultdict["status"] != "unsupported"
                 and proposal.resultdict[selected]["heldout_mse"] <= tolerance
                 and mse(model, rows) <= tolerance)

    if not supported:
        # The dependency exists before the first instrument reading is bought.
        child = graph.add_question("instrument", "Which control-state law predicts the auxiliary meter?",
                                   required_measurements=("meter",))
        graph.set_dependencies("response", ("instrument",))
        parent.status = "blocked"
        emit("child_opened", question_id="instrument", parent_id="response",
             reason="history revision remains unsupported or inaccurate")
        measured_records = []
        impulse = {"kind": "trajectory", "sequence": [{"u": 1.}, {"u": 0.}, {"u": 0.}],
                   "channels": ["meter"]}
        for action in [impulse, *[random_action(3, True) for _ in range(4)]]:
            measured_records.append(purchase(target_lab, action, "instrument_seed"))
        instrument_model = fit_temporal_instrument(measured_records, target_lab.public_spec(),
            compute_budget=TEMPORAL_LIMITS["instrument_compute_units"])
        result["instrument_compute_cost"] += instrument_model["compute_cost"]
        result["instrument_model"] = instrument_model
        instrument = None
        for serial in range(4):
            plan = select_temporal_instrument_probe(instrument_model, target_lab.public_spec(),
                serial=serial, tolerance=tolerance,
                compute_budget=TEMPORAL_LIMITS["instrument_compute_units"] - result["instrument_compute_cost"])
            result["instrument_compute_cost"] += plan["compute_cost"]
            if plan["action"] is None:
                emit("child_unresolved", reason=plan["reason"])
                break
            if target_lab.costs["observation_cost"] + len(plan["action"]["sequence"]) + 48 > 240:
                emit("child_unresolved", reason="confirmation reserve")
                break
            emit("experiment_selected", question_id="instrument", **plan)
            measured = purchase(target_lab, plan["action"], "instrument_disagreement")
            measured_records.append(measured)
            assessment = assess_temporal_instrument(instrument_model, [measured],
                target_lab.public_spec(), tolerance=tolerance,
                compute_budget=TEMPORAL_LIMITS["instrument_compute_units"] - result["instrument_compute_cost"])
            result["instrument_compute_cost"] += assessment["compute_cost"]
            emit("instrument_assessed", **assessment)
            instrument = temporal_instrument_program(instrument_model)
            if instrument is not None:
                break
        if instrument is None:
            result["status"] = "child_unresolved"
            return finish(model, source_model, history_model)
        for record in measured_records:
            measured_by_index = {item["observation_index"]: item["values"]["meter"]
                                 for item in record["measurements"]}
            graph.supply_measurements([Observation({**row["features"], "meter": measured_by_index[index]},
                                       measured_by_index[index], tuple(row["history"]))
                                       for index, row in zip(record["row_indices"], record["rows"])],
                                      question_id="instrument")
        child.answer, child.status = instrument, "answered"
        emit("child_answered", question_id="instrument", program=instrument.to_dict())
        rows = [row for record in records for row in record["rows"]]
        expression, rank, width = fit_temporal_response(rows, instrument)
        result["response_fit_calls"] += 1
        graph.set_dependencies("response", ("instrument",), expression=expression)
        parent.observations = [observation(row) for row in rows]
        # Binding the live child answer changes the actual predictor.
        parent.answer = bind_answers(parent.expression, {"instrument": graph.nodes["instrument"].answer})
        parent.status = "answered"
        model = parent.answer
        result["response_readout"] = {"training_rank": rank, "feature_count": width,
                                     "train_mse": mse(model, rows)}
        emit("parent_revised", depends_on=["instrument"], **result["response_readout"])

    # Saved before new labels; confirmation never refits this model.
    result["fit_rows"] = sum(len(record["rows"]) for record in records)
    fresh = [row for action in target_lab.confirmation_actions(48, Random(seed + 2))
             for row in purchase(target_lab, action, "confirmation")["rows"]]
    result["final_confirmation_mse"] = mse(model, fresh)
    result["status"] = "confirmed" if result["final_confirmation_mse"] <= tolerance else "confirmation_rejected"
    parent.status = result["status"]
    emit(result["status"], mse=result["final_confirmation_mse"])
    return finish(model, source_model, history_model)
