"""Measure a failed parameter experiment before changing its relation claim."""

from copy import deepcopy
from dataclasses import replace

from .competing_object_experiments import _forecast, assess_competing_object_probe
from .language import Program
from .models import Probe
from .relocatable_programs import abstract_measurements, instantiate_measurements
from .research_program_inputs import abstract_research_inputs
from .scale_bindings import bind_measurement_arguments


def measured_relation_entry(entry, spec, library):
    """Recover the existing input contract and the unconditioned parameter template."""
    inputs = abstract_measurements(entry["candidate"].program, spec, library)
    if not inputs["complete"]:
        return None
    procedure = entry.get("research_program")
    if procedure:
        procedure = abstract_research_inputs(procedure, spec, library)
        if procedure is None:
            return None
    return entry, inputs, procedure


def bind_relation_entry(prepared, binding):
    entry, inputs, procedure = prepared
    if (set(inputs["required_fields"]) | set((procedure or {}).get("input_features", ()))) - set(binding):
        return None
    result = deepcopy(entry)
    result["candidate"] = replace(entry["candidate"], program=instantiate_measurements(inputs, binding))
    if procedure:
        result["research_program"] = procedure = deepcopy(procedure)

        def convert(value):
            return bind_measurement_arguments(Program.from_dict(value), binding).to_dict() if value else value

        for node in procedure["graph"]["nodes"].values():
            for name in ("expression", "answer", "residual_base"):
                node[name] = convert(node.get(name))
            node["alternatives"] = [convert(program) for program in node["alternatives"]]
        for step in procedure["selection_steps"]:
            step["program"] = convert(step["program"])
            if step.get("parameter_template"):
                step["parameter_template"] = convert(step["parameter_template"])
            if step.get("parameter_terms"):
                step["parameter_terms"] = [convert(term) for term in step["parameter_terms"]]
    return result


def diagnose_relation_inputs(agent, laboratory, selected, origin, plan, evidence,
                             acquire, paired_for, budget, actions, *, channels=()):
    """Acquire the exact failed trajectory's inputs, using the existing budget.

    A failed law warrants new input readings when all explanations fail, or
    when a competing radius could be masking an unmeasured instrument cut.
    ``origin`` is the entry saved before this parameter research narrowed it.
    Acquisitions use the caller's ordinary metered path; no labels enter fitting.
    """
    failed = any(item["candidate_id"] == selected["candidate"].candidate_id and
                 item["status"] == "counterexample" for item in evidence["hypotheses"])
    if (channels or not failed or evidence["calibration_status"] != "not_forecast" or
            not agent._learn_instruments or not agent._revise_inputs):
        return None
    views = getattr(laboratory, "measurement_views", ())
    measured_channels = tuple(views[-1]) if views else tuple(channel.name for channel in laboratory.channels)
    if not measured_channels:
        return None
    command_spec, spec = agent._spec(laboratory, ()), agent._spec(laboratory, measured_channels)
    if not spec.context.get("object_schema"):
        return None
    old_binding = agent._instrument_binding(spec, command_spec)
    if any(item["status"] == "supported_on_probe" for item in evidence["hypotheses"]):
        from .instrument_condition_probes import unmeasured_instrument_conditions
        before = budget.spent
        questions = unmeasured_instrument_conditions(old_binding, command_spec,
                                                     paired_for(measured_channels), budget)
        touched = [item for item in questions or []
                   if any(point.get(item["control"]) == item["value"] for point in plan["probe"].sequence)]
        actions.append({"action": "assess_relation_input_purchase", "probe_id": plan["probe"].probe_id,
            "unmeasured_conditions_touched": touched, "purchase": bool(touched),
            "compute_cost": budget.spent - before})
        if not touched:
            return None
    prepared = measured_relation_entry(origin, spec, agent.library)
    result = {"status": "unresolved_inputs", "entry": None, "command_rows": [],
              "plan": plan, "evidence": evidence, "binding": old_binding}
    previous_batches = len(paired_for(measured_channels))

    def finish():
        result["command_rows"] = [command for batch in paired_for(measured_channels)[previous_batches:]
                                  for command, _ in batch]
        actions.append({"action": "diagnose_relation_experiment_inputs", "probe_id": plan["probe"].probe_id,
            "measurement_probe_id": f"measure_relation:{plan['probe'].probe_id}",
            "status": result["status"], "field_losses": result["evidence"]["field_losses"],
            "measured_response_status": result["evidence"]["status"],
            "restored_relation_parameters": bool(result["entry"]),
            "observation_rows": len(result["command_rows"])})
        return result

    # Replay first: a pure relation counterexample needs no new instrument fit.
    # If inputs fail, varied background and independent validation follow; the
    # exact counterexample remains among the fitting rows.
    measured = acquire(measured_channels, "remeasure_relation_counterexample", probe=Probe(
        f"measure_relation:{plan['probe'].probe_id}", plan["probe"].sequence))
    if measured is None:
        return finish()
    measured_pairs = paired_for(measured_channels)[-1]
    rechecked = deepcopy(plan)
    rechecked["predicted_fields"] = {}
    for name, program in old_binding.items():
        if not budget.spend(len(measured_pairs)):
            return finish()
        rechecked["predicted_fields"][name] = _forecast(program, plan["probe"].sequence)
    hypotheses = []
    for hypothesis in plan["hypotheses"]:
        payload = abstract_measurements(Program.from_dict(hypothesis["program"]), spec, agent.library)
        if not payload["complete"]:
            break
        hypotheses.append({**hypothesis, "program": payload["program"]})
    if len(hypotheses) == len(plan["hypotheses"]):
        rechecked["hypotheses"] = hypotheses
    else:
        # Input observations still diagnose calibration; an unrecovered law
        # has no measured-coordinate interpretation to score or support.
        rechecked["hypotheses"] = []
    assessment = assess_competing_object_probe(rechecked, measured_pairs, budget,
        tolerance=agent.config.minimum_improvement)
    result.update(plan=rechecked, evidence=assessment)
    if assessment["calibration_status"] == "supported_on_probe":
        if set(assessment["field_losses"]) == set(old_binding):
            result["status"] = "inputs_supported"
        return finish()
    if assessment["calibration_status"] != "counterexample":
        return finish()
    from .instrument_condition_probes import instrument_condition_probe
    before = budget.spent
    endpoint_probe, questions = instrument_condition_probe(old_binding, command_spec,
        paired_for(measured_channels), budget, reference=plan["probe"].sequence[0],
        serial=plan["probe"].probe_id)
    if endpoint_probe is not None:
        endpoints = acquire(measured_channels, "resolve_instrument_conditions", probe=endpoint_probe)
        actions.append({"action": "measure_instrument_condition_boundaries",
            "probe_id": endpoint_probe.probe_id, "questions": questions,
            "observation_rows": len(endpoints) if endpoints is not None else 0,
            "compute_cost": budget.spent - before})
        if endpoints is None:
            return finish()
    if not previous_batches and acquire(measured_channels, "diagnose_relation_input_mapping") is None:
        return finish()
    checked = acquire(measured_channels, "compare_relation_input_mapping")
    if checked is not None:
        model = agent._fit_instruments(command_spec, spec, paired_for(measured_channels), budget, actions)
        if model is not None:
            binding = agent._instrument_binding(spec, command_spec)
            result["entry"] = bind_relation_entry(prepared, binding) if prepared else None
            if result["entry"] is not None:
                result["status"] = "inputs_revised"
    return finish()
