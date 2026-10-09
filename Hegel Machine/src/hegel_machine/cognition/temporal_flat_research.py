"""Flat adaptive control with the recursive procedure's candidate repertoire.

Competing instrument programs are coordinates of competing response predictors.
Paid response validation can select one without first resolving an instrument
question. No question graph or parent/child acceptance dependency is constructed.
"""

from random import Random

from .history_revision import revise_history
from .knowledge_evolution import bind_answers
from .language import Program
from .temporal_instrument_research import (
    assess_temporal_instrument,
    fit_temporal_instrument,
    select_temporal_instrument_probe,
)
from .temporal_research import (
    TEMPORAL_LIMITS,
    TemporalResearchResult,
    _history_program,
    fit_temporal_response,
    mse,
)


def run_flat_temporal_research(source_lab, target_lab, *, seed=71, tolerance=1e-10):
    """Adapt the response directly, using only purchased training/validation data.

    The source and initial target purchases match recursive research exactly.
    Instrument-conditioned response candidates fit the first two long target
    trajectories and every purchased meter trajectory. The other two original
    target trajectories remain validation-only until a candidate is selected.
    A selected representation is then refit on all paid rows and frozen before
    the same fresh response confirmation used by recursive research.
    """
    rng = Random(seed)
    events, records = [], []
    result = {
        "status": "research_incomplete", "events": events,
        "source_confirmation_mse": None, "final_confirmation_mse": None,
        "instrument_model": None, "instrument_compute_cost": 0,
        "response_fit_calls": 0, "limits": dict(TEMPORAL_LIMITS),
    }

    def emit(event, **details):
        events.append({"event": event, **details})

    def purchase(lab, action, phase):
        limit = TEMPORAL_LIMITS["target_rows" if lab is target_lab else "source_rows"]
        reserve = (TEMPORAL_LIMITS["fresh_confirmation_rows"]
                   if lab is target_lab and phase != "confirmation" else 0)
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
        result["costs"] = {"source": dict(source_lab.costs), "target": dict(target_lab.costs)}
        return TemporalResearchResult(model, source_model, history_model, result, records)

    def remaining_compute():
        return TEMPORAL_LIMITS["instrument_compute_units"] - result["instrument_compute_cost"]

    source_rows = [row for _ in range(12)
                   for row in purchase(source_lab, random_action(3), "source_train")["rows"]]
    source_model = _history_program(revise_history(source_rows))
    fresh_source = [row for action in source_lab.confirmation_actions(12, Random(seed + 1))
                    for row in purchase(source_lab, action, "source_confirmation")["rows"]]
    result["source_confirmation_mse"] = mse(source_model, fresh_source)
    if result["source_confirmation_mse"] > tolerance:
        result["status"] = "source_rejected"
        return finish(source_model, source_model, source_model)
    emit("source_confirmed", mse=result["source_confirmation_mse"])

    horizon = target_lab.public_spec()["max_probe_length"]
    for _ in range(4):
        purchase(target_lab, random_action(horizon), "scope_test")
    rows = [row for record in records for row in record["rows"]]
    result["frozen_source_target_mse"] = mse(source_model, rows)
    if result["frozen_source_target_mse"] > tolerance:
        emit("scope_counterexample", mse=result["frozen_source_target_mse"])
    proposal = revise_history(rows, holdout_fraction=.5)
    model = history_model = _history_program(proposal)
    result["history_revision"] = proposal.resultdict
    emit("history_revised", status=proposal.resultdict["status"],
         selected_lags=proposal.selected_lags, train_mse=mse(model, rows))
    selected = "revision" if proposal.selected_lags == 3 else "base"
    supported = (proposal.resultdict["status"] != "unsupported"
                 and proposal.resultdict[selected]["heldout_mse"] <= tolerance
                 and mse(model, rows) <= tolerance)

    if not supported:
        fit_scope_rows = [row for record in records[:2] for row in record["rows"]]
        validation_rows = [row for record in records[2:4] for row in record["rows"]]
        measured_records = []
        impulse = {"kind": "trajectory", "sequence": [{"u": 1.}, {"u": 0.}, {"u": 0.}],
                   "channels": ["meter"]}
        for action in [impulse, *[random_action(3, True) for _ in range(4)]]:
            measured_records.append(purchase(target_lab, action, "instrument_seed"))
        instrument_model = fit_temporal_instrument(
            measured_records, target_lab.public_spec(), compute_budget=remaining_compute())
        result["instrument_compute_cost"] += instrument_model["compute_cost"]
        result["instrument_model"] = instrument_model

        def compare_candidates(round_index):
            fitting_rows = fit_scope_rows + [row for record in measured_records for row in record["rows"]]
            candidates, accepted = [], []
            for hypothesis in instrument_model.get("fields", {}).get("meter", {}).get("hypotheses", []):
                if hypothesis.get("status") in {"refuted", "counterexample"}:
                    continue
                instrument = Program.from_dict(hypothesis["program"])
                expression, rank, width = fit_temporal_response(fitting_rows, instrument)
                result["response_fit_calls"] += 1
                predictor = bind_answers(expression, {"instrument": instrument})
                training_loss = mse(predictor, fitting_rows)
                validation_loss = mse(predictor, validation_rows)
                # Rank is reported, as in the recursive response fit. Selection
                # uses paid predictive evidence rather than uniqueness of meter.
                candidate = {
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "instrument_status": hypothesis["status"],
                    "training_rank": rank, "feature_count": width,
                    "train_mse": training_loss, "validation_mse": validation_loss,
                    "supported": training_loss <= tolerance and validation_loss <= tolerance,
                }
                candidates.append(candidate)
                if candidate["supported"]:
                    accepted.append((validation_loss, training_loss, instrument.size,
                                     hypothesis["hypothesis_id"], instrument))
            chosen = min(accepted, key=lambda item: item[:4]) if accepted else None
            emit("flat_candidate_comparison", round=round_index,
                 training_rows=len(fitting_rows), validation_rows=len(validation_rows),
                 candidates=candidates, selected_hypothesis=None if chosen is None else chosen[3])
            if chosen is not None:
                result["selected_instrument_hypothesis"] = chosen[3]
                return chosen[4]
            return None

        instrument = compare_candidates(0)
        for serial in range(4):
            if instrument is not None:
                break
            plan = select_temporal_instrument_probe(
                instrument_model, target_lab.public_spec(), serial=serial,
                tolerance=tolerance, compute_budget=remaining_compute())
            result["instrument_compute_cost"] += plan["compute_cost"]
            if plan["action"] is None:
                emit("flat_unresolved", reason=plan["reason"])
                break
            count = len(plan["action"]["sequence"])
            if (target_lab.costs["observation_cost"] + count
                    + TEMPORAL_LIMITS["fresh_confirmation_rows"] > TEMPORAL_LIMITS["target_rows"]):
                emit("flat_unresolved", reason="confirmation reserve")
                break
            emit("experiment_selected", **plan)
            measured = purchase(target_lab, plan["action"], "instrument_disagreement")
            measured_records.append(measured)
            assessment = assess_temporal_instrument(
                instrument_model, [measured], target_lab.public_spec(),
                tolerance=tolerance, compute_budget=remaining_compute())
            result["instrument_compute_cost"] += assessment["compute_cost"]
            emit("instrument_assessed", **assessment)
            instrument = compare_candidates(serial + 1)
        if instrument is None:
            result["status"] = "flat_unresolved"
            result["fit_rows"] = sum(len(record["rows"]) for record in records)
            return finish(model, source_model, history_model)

        rows = [row for record in records for row in record["rows"]]
        expression, rank, width = fit_temporal_response(rows, instrument)
        result["response_fit_calls"] += 1
        model = bind_answers(expression, {"instrument": instrument})
        result["response_readout"] = {"training_rank": rank, "feature_count": width,
                                      "train_mse": mse(model, rows)}
        emit("flat_response_revised", instrument=instrument.to_dict(), **result["response_readout"])

    result["fit_rows"] = sum(len(record["rows"]) for record in records)
    fresh = [row for action in target_lab.confirmation_actions(48, Random(seed + 2))
             for row in purchase(target_lab, action, "confirmation")["rows"]]
    result["final_confirmation_mse"] = mse(model, fresh)
    result["status"] = "confirmed" if result["final_confirmation_mse"] <= tolerance else "confirmation_rejected"
    emit(result["status"], mse=result["final_confirmation_mse"])
    return finish(model, source_model, history_model)
