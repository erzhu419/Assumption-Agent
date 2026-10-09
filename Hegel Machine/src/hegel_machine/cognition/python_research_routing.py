"""Expose existing temporal revision steps to failed ordinary Python candidates.

This proposes one executable action. The caller retains control of acquisitions
and submission; the existing operators own fitting, assessment and accounting.
"""

from copy import deepcopy
import json

from .code_agent import python_model_context
from .temporal_instrument_research import temporal_instrument_program
from .temporal_research import TEMPORAL_LIMITS


TOLERANCE = 1e-10


def matching_refit(agent, model):
    """A rewritten predictor must not inherit the previous source's diagnosis."""
    source, inner = python_model_context(agent.workspace, model)

    def modules(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*.py")
                if "__pycache__" not in path.parts and "_evaluated_model" not in path.relative_to(directory).parts}

    for result in reversed(agent.operator_results):
        tested = result.get("tested_model")
        if result.get("operator") == "refit" and tested:
            context, inner_model = python_model_context(agent.workspace, tested)
            if inner == inner_model and modules(source) == modules(context):
                return result
    return None


def uses_refitted_training_view(agent, model, refit):
    """Separate a refitted predictor from old coefficients of the same program."""
    def rows(path):
        context, _ = python_model_context(agent.workspace, path)
        training = context / "model_observations.json"
        if not training.exists():
            training = context / "observations.json"
        return json.loads(training.read_text())
    return rows(model) == rows(refit["tested_model"])


def validated_candidates(agent):
    """Exported paths need trajectory validation before automatic assessment."""
    candidates = []
    for result in agent.operator_results:
        if result.get("operator") == "refit" and result.get("status") == "refit_supported_on_validation":
            candidates.append(result["path"])
        elif result.get("operator") == "response":
            candidates.extend(item["path"] for item in result.get("candidates", [])
                              if item.get("validation_mse") is not None
                              and item["validation_mse"] <= TOLERANCE)
    return list(dict.fromkeys(candidates))


def _offer(agent, diagnosis, name, action, *, observations=0, compute=0, measurement=0):
    rows, confirmation = len(agent.rows), agent.config.confirmation_rows
    return {"diagnosis": diagnosis, "name": name, "next_action": action,
            "observation_cost": observations + confirmation,
            "compute_cost": compute + rows + observations + confirmation,
            "measurement_cost": measurement}


def _run(operator, **arguments):
    return {"action": "run", "arguments": {"operator": operator, **arguments}}


def revision_action(agent, retained, spec):
    """Use paid outcomes to expose refit, calibration, and response research."""
    rows, index = len(agent.rows), len(agent.operator_results)
    refit = matching_refit(agent, retained["model"])
    if refit is None:
        if len(agent.observations) < 2:
            return {"diagnosis": "separate_validation_trajectory_needed"}
        return _offer(agent, "response_counterexample", "refit_response",
            _run("refit", model=retained["model"], path=f"shared/refits/revision_{index:04d}"),
            compute=2 * rows)

    diagnosis = refit["status"]
    # A new counterexample may warrant another fit after an earlier successful
    # fit. Failed validation alone cannot trigger repeated identical fitting.
    if diagnosis == "refit_supported_on_validation" and refit["paid_rows_at_refit"] < rows:
        arguments = {"model": retained["model"], "path": f"shared/refits/revision_{index:04d}"}
        selected = refit["split"]["records"]
        if selected != list(range(refit["paid_records_at_refit"])):
            arguments["records"] = selected + list(range(refit["paid_records_at_refit"], len(agent.observations)))
        return _offer(agent, "refitted_response_counterexample", "refit_response",
            _run("refit", **arguments),
            compute=2 * rows)
    return instrument_action(agent, spec, diagnosis)


def instrument_action(agent, spec, diagnosis="instrument_research"):
    """Continue an already opened calibration question without restarting refit."""
    rows, confirmation = len(agent.rows), agent.config.confirmation_rows
    index = len(agent.operator_results)
    fitted = next((result for result in reversed(agent.operator_results)
                   if result.get("operator") == "instrument" and result.get("path")), None)
    metered = [i for i, record in enumerate(agent.observations)
               if record.get("measurements")]
    instrument_path = fitted["path"] if fitted else "shared/research_instrument.json"
    # An attempted fit without paired readings has not calibrated anything.
    # Resume the existing acquisition path, including when readings arrived
    # after that empty fit, instead of treating it as a refuted instrument.
    if fitted and not any(agent.observations[i].get("measurements") for i in fitted["records"]):
        fitted = None
    cap = min(TEMPORAL_LIMITS["instrument_compute_units"],
              max(0, agent._remaining_compute() - 2 * rows - confirmation))

    # Replay measured controls in this resettable laboratory. Their readings are
    # newly purchased; the response targets were already observed. Prefer two
    # distinct, long trajectories to separate calibration fit and assessment.
    controls = []
    for record in sorted(agent.observations, key=lambda item: -len(item["rows"])):
        sequence = record["action"]["sequence"]
        if len(sequence) >= 6 and sequence not in controls:
            controls.append(deepcopy(sequence))
    meter_cost = next(item["cost_per_row"] for item in spec["channels"] if item["name"] == "meter")

    if fitted is None:
        if metered:
            return _offer(agent, diagnosis, "remeasure_inputs",
                _run("instrument", path=instrument_path, records=metered, compute_limit=cap), compute=cap)
        if len(controls) < 2:
            return {"diagnosis": "distinct_calibration_trajectories_needed"}
        sequences = controls[:2]
        acquired = len(agent.observations)
        count = sum(map(len, sequences))
        # Cap fitting while preserving scoring for assessment and confirmation.
        cap = min(cap, max(0, agent._remaining_compute() - 2 * (rows + count) - confirmation))
        return _offer(agent, diagnosis, "remeasure_inputs", {"action": "query", "arguments": {
            "actions": [{"kind": "trajectory", "sequence": sequence, "channels": ["meter"]}
                        for sequence in sequences],
            "after_observation": [
                {"operator": "instrument", "path": instrument_path, "records": [acquired],
                 "compute_limit": cap},
                {"operator": "assess_instrument", "instrument": instrument_path,
                 "records": [acquired + 1]}]}},
            observations=count, compute=cap + rows + count, measurement=count * meter_cost)

    model = json.loads((agent.workspace / instrument_path).read_text())
    survivors = [item for item in model.get("fields", {}).get("meter", {}).get("hypotheses", [])
                 if item.get("status") not in {"refuted", "counterexample"}]
    if not survivors:
        additional = [record for record in metered if record not in fitted["records"]]
        if additional:
            return _offer(agent, "instrument_counterexample", "remeasure_inputs",
                _run("instrument", path=instrument_path, records=metered, compute_limit=cap), compute=cap)
        return {"diagnosis": "instrument_revision_unresolved"}
    if temporal_instrument_program(model) is None:
        assessed = {i for result in agent.operator_results[fitted["index"] + 1:]
                    if result.get("operator") == "assess_instrument" and result.get("path") == instrument_path
                    for i in result.get("records", [])}
        assessment = [i for i in metered if i not in fitted["records"] and i not in assessed]
        if assessment:
            return _offer(agent, "instrument_needs_validation", "validate_inputs",
                _run("assess_instrument", instrument=instrument_path, records=assessment),
                compute=sum(len(agent.observations[i]["rows"]) for i in assessment) * len(survivors))
        if len(survivors) > 1:
            planned = next((result for result in reversed(agent.operator_results[fitted["index"] + 1:])
                            if result.get("operator") == "disagreement"
                            and result["arguments"].get("instrument") == instrument_path), None)
            consumed = {result.get("arguments", {}).get("experiment") for result in agent.operator_results
                        if result.get("operator") == "assess_instrument"}
            if planned and (planned.get("experiment") is None or planned["experiment"] not in consumed):
                action = planned.get("action")
                if action is None:
                    return {"diagnosis": "instrument_ambiguity_unresolved"}
                count = len(action["sequence"])
                return _offer(agent, "instrument_ambiguity", "discriminate_inputs",
                    {"action": "query", "arguments": {"actions": [action], "after_observation": [
                        {"operator": "assess_instrument", "instrument": instrument_path,
                         "experiment": planned["experiment"], "records": "acquired"}]}},
                    observations=count, compute=count * len(survivors), measurement=count * meter_cost)
            return _offer(agent, "instrument_ambiguity", "discriminate_inputs",
                _run("disagreement", instrument=instrument_path, compute_limit=cap), compute=cap)
        fit_controls = [agent.observations[i]["action"]["sequence"] for i in fitted["records"]]
        sequence = next((value for value in controls if value not in fit_controls), None)
        if sequence is None:
            return {"diagnosis": "distinct_calibration_trajectories_needed"}
        return _offer(agent, "instrument_needs_validation", "validate_inputs", {"action": "query", "arguments": {
            "actions": [{"kind": "trajectory", "sequence": sequence, "channels": ["meter"]}],
            "after_observation": [{"operator": "assess_instrument", "instrument": instrument_path,
                                   "records": "acquired"}]}}, observations=len(sequence),
            compute=len(sequence) * len(survivors), measurement=len(sequence) * meter_cost)

    previous = next((result for result in reversed(agent.operator_results)
                     if result.get("operator") == "response" and result.get("instrument") == instrument_path
                     and result["split"]["records"] == list(range(len(agent.observations)))), None)
    if previous is not None:
        return {"diagnosis": "response_revision_unresolved"}
    # Replayed controls have new meter readings but duplicate parent responses.
    # Keep every occurrence of a control trajectory on the same side of the
    # parent response split, so calibration purchases cannot leak its holdout.
    groups, sequences = [], []
    for i, record in enumerate(agent.observations):
        sequence = record["action"]["sequence"]
        if sequence not in sequences:
            sequences.append(sequence)
            groups.append([])
        groups[sequences.index(sequence)].append(i)
    if len(groups) < 2:
        return {"diagnosis": "separate_validation_trajectory_needed"}
    held = [i for group in groups[-min(2, len(groups) - 1):] for i in group]
    train, validation, _ = agent._split({"validation_records": held})
    if not validation:
        return {"diagnosis": "separate_validation_trajectory_needed"}
    compute = (12 * len(train) + len(validation) + 12 * rows) * len(survivors)
    compute += sum(len(agent.observations[i]["measurements"]) for i in metered) * len(survivors)
    return _offer(agent, "instrument_supported_response_unresolved", "revise_mechanism",
        _run("response", instrument=instrument_path, output_dir=f"shared/response_revision_{index:04d}",
            validation_records=held),
        compute=compute + rows)


def evaluate_action(agent, model, diagnosis, *, validated=False):
    """Use the last live call for scoring plus fresh confirmation when validated."""
    retained = agent._evaluated_summary()
    comparison = len(agent.rows) if retained and retained["support"] != "paid_fit" else 0
    if validated and agent.config.max_model_calls - agent.costs["model_calls"] == 1:
        # finish already scores the candidate before buying fresh confirmation.
        # A separate evaluate would consume the only remaining live decision.
        return _offer(agent, diagnosis, "confirm_candidate",
            {"action": "finish", "arguments": {"model": model}})
    return _offer(agent, diagnosis, "evaluate_candidate",
        {"action": "evaluate", "arguments": {"model": model}}, compute=len(agent.rows) + comparison)
