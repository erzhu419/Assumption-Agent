"""Settle native source returns against an Agent's actual fresh confirmation."""

from copy import deepcopy
from dataclasses import asdict

from .evaluation import loss
from .language import Program
from .models import Candidate


def _retained_receipts(library, selected_program, source_returns):
    canonical = selected_program.expanded().simplified()
    sources = {concept.concept_id: concept for concept in library.concepts}
    return [receipt for receipt in source_returns
        if receipt.get("retained_program") is not None
        and receipt.get("source_id") in sources
        and Program.from_dict(receipt["retained_program"]).expanded().simplified() == canonical]


def _deferred(reason):
    return {"status": "deferred", "reason": reason, "compute_cost": 0}


def settle_source_confirmation(library, spec, *, previous, selected, source_returns,
                               paid_rows, fresh_rows, frozen_code, submitted_code,
                               comparison_id, available_compute, passed, tolerance=1e-10):
    """Compare the unchanged source and incumbent on already purchased new rows.

    The caller supplies the actual ``finish`` observations and charges the
    returned computation. No measurement, fitting or inferred instrument value
    is introduced here. Existing memory rules decide confirmation and update
    production workflows; a successful Python call alone earns no source credit.
    """
    from .temporal_operator_agent import export_temporal_program

    selected_program = Program.from_dict(selected["program"])
    previous_program = Program.from_dict(previous["program"])
    try:
        expected_code = export_temporal_program(selected_program)
        export_temporal_program(previous_program)
    except ValueError:
        return _deferred("source_or_incumbent_requires_unavailable_prediction_fields")
    if frozen_code != expected_code or submitted_code != frozen_code:
        return _deferred("submitted_model_differs_from_frozen_source_program")
    return _settle_comparison(library, spec, previous=previous, selected=selected,
        source_returns=source_returns, paid_rows=paid_rows, fresh_rows=fresh_rows,
        comparison_id=comparison_id, available_compute=available_compute,
        passed=passed, tolerance=tolerance)


def confirm_source_return(library, laboratory, *, previous, selected, source_returns,
                          paid_rows, probe, available_compute, observation_allowance,
                          measurement_allowance, comparison_id, tolerance=1e-7):
    """Purchase one fresh paired check through the existing public laboratory.

    Native measured programs remain measured programs: the shared experiment
    buys the union of both predictors' channels, and does not export or certify
    a control-only predictor. The caller freezes ``probe`` before this call.
    """
    if not _retained_receipts(library, Program.from_dict(selected["program"]), source_returns):
        return _deferred("no_retained_source_for_selected_program")
    channels = tuple(dict.fromkeys((*previous.get("selected_channels", ()),
                                   *selected.get("selected_channels", ()))))
    spec = laboratory.spec_for(channels)
    for receipt in source_returns:
        source = next((item for item in library.concepts
                       if item.concept_id == receipt.get("source_id")), None)
        if source is not None and any(comparison.get("comparison_id") == comparison_id
                for prior in source.reuse_history if prior.get("task_id") == spec.task_id
                for comparison in prior.get("comparisons", ())):
            return _deferred("confirmation_already_settled")
    count = len(probe.sequence)
    measurement = count * sum(channel.cost_per_row for channel in laboratory.channels
                              if channel.name in channels)
    if not count:
        return _deferred("empty_confirmation_probe")
    if observation_allowance < count or measurement_allowance < measurement:
        return _deferred("insufficient_confirmation_observations_or_measurements")
    if available_compute < 2 * count:
        return _deferred("insufficient_paired_comparison_compute")
    # Both programs and their origins were frozen before any new labels arrive.
    selected, previous, source_returns = deepcopy((selected, previous, source_returns))
    batch = laboratory.acquire(probe, channels)
    result = _settle_comparison(library, spec, previous=previous, selected=selected,
        source_returns=source_returns, paid_rows=paid_rows, fresh_rows=batch.observations,
        comparison_id=comparison_id, available_compute=available_compute,
        passed=True, tolerance=tolerance)
    result.update(probe_id=probe.probe_id, channels=list(channels),
        observation_cost=batch.observation_cost, measurement_cost=batch.measurement_cost,
        new_costs={"observation_cost": batch.observation_cost,
                   "measurement_cost": batch.measurement_cost,
                   "compute_cost": result["compute_cost"], "queries": 1},
        batch={"probe": asdict(probe), "channels": list(channels),
               "observations": [asdict(row) for row in batch.observations], "transitions": []})
    return result


def _settle_comparison(library, spec, *, previous, selected, source_returns,
                       paid_rows, fresh_rows, comparison_id, available_compute,
                       passed, tolerance):
    selected_program = Program.from_dict(selected["program"])
    previous_program = Program.from_dict(previous["program"])
    sources = {concept.concept_id: concept for concept in library.concepts}
    receipts = _retained_receipts(library, selected_program, source_returns)
    if not receipts:
        return _deferred("no_retained_source_for_selected_program")
    if any(comparison.get("comparison_id") == comparison_id
           for receipt in receipts
           for prior in sources[receipt["source_id"]].reuse_history
           if prior.get("task_id") == spec.task_id
           for comparison in prior.get("comparisons", ())):
        return _deferred("confirmation_already_settled")
    rows = list(fresh_rows)
    compute = 2 * len(rows)
    if not rows:
        return _deferred("no_fresh_confirmation_rows")
    if available_compute < compute:
        return _deferred("insufficient_paired_comparison_compute")
    before = loss(previous_program, rows, previous.get("fallback", 0.), spec.risk_quantile)
    after = loss(selected_program, rows, selected.get("fallback", 0.), spec.risk_quantile)
    confirmed = bool(passed and after <= tolerance)
    retained = Candidate("agent_finish_source", selected_program, "source_confirmation")
    incumbent = Candidate("agent_finish_incumbent", previous_program, "retained_incumbent")
    library.begin_transfer_episode(spec, paid_rows)
    for receipt in receipts:
        origin = {key: deepcopy(receipt[key]) for key in (
            "source_id", "template_id", "condition", "mapping", "mapping_roles",
            "lineage_id", "joint_source_ids") if key in receipt}
        # Fitting was charged in the preceding native visit; this is a second
        # observation of the frozen program, not another adaptation.
        origin["adaptation_work"] = []
        candidate = Candidate("agent_finish_source", selected_program, "source_confirmation",
            parents=tuple(receipt.get("joint_source_ids", [receipt["source_id"]])),
            memory_id=receipt["source_id"])
        library.record_transfer_result(candidate, spec,
            accepted=confirmed and after < before, loss=after, baseline_loss=before,
            evidence_rows=len(rows), comparison_compute=compute,
            comparison_id=comparison_id, origin=origin)
    returned = library.finish_transfer_episode(spec,
        final_candidate=retained if confirmed else incumbent,
        final_loss=after if confirmed else before, baseline_loss=before,
        compute_cost=compute, observation_cost=len(rows), validation_rows=len(rows),
        final_status="completed" if confirmed else "unresolved")
    return {"status": "settled", "compute_cost": compute,
            "before_loss": before, "after_loss": after,
            "confirmed_program": confirmed, "knowledge_return": returned}
