"""Compile a selected research step into one funded learning visit."""

from math import ceil

from .controller import _continuous_state_work, _representation_work
from .risk_search import trajectory_blocks
from .state_search import state_search_work
from .uncertainty import risk_calibration_quote


def continuation_plan(config, spec, workspace, *, action="residual_revision", estimate=None):
    """Fund preparation, one fit, new comparisons and the resulting report.

    Component returns describe a method step. The session buys the entire
    executable visit needed to test that step. Its allocation is a hard cap;
    unfinished evidence remains a question for a subsequent allocation.
    """
    estimate = estimate or {}
    rows = workspace.observations
    count = len(rows)
    depth = max((candidate.program.history_depth for candidate in
                 [workspace.incumbent, *workspace.bank] if candidate is not None), default=0)
    length = min(spec.max_probe_length, max(8, 4 * depth + 4))
    step_observations = (0 if estimate.get("basis") == "measured_continuation_return"
                         else estimate.get("observation_cost", 0))
    preparation = (max(length, ceil(step_observations))
                   if action in {"representative", "transition", "task_value"} else 0)
    # The controller can prepare an untested historical alternative before
    # fitting. Two restart trajectories still remain for its comparison.
    history_contrast = (spec.decision_costs is not None and config.active_probes
        and len({candidate.program.history_depth > 0 for candidate in
                 [workspace.incumbent, *workspace.bank] if candidate is not None}) > 1)
    preparation = max(preparation, 2 * length if history_contrast else 0)
    new_rows = count + preparation
    scale = new_rows / max(1, count)
    if action == "refit":
        fitting = 1600
    elif action == "continuous_state":
        fitting = ceil(_continuous_state_work(rows, spec) * scale)
    else:
        fitting = ceil(max(2600, _representation_work(rows, spec, config.max_history) * scale)
                       + (state_search_work(rows) * scale if config.state_search and spec.risk_quantile == .5 else 0))
    comparison_horizon = min(spec.max_probe_length, 4 * config.max_history + 4)
    comparisons = (min(config.max_candidates, 8) + 3) * (
        new_rows + len(workspace.reference) + 2 * max(comparison_horizon, 2 * length if history_contrast else length))
    preparation_work = 500 if preparation else 0
    adaptation = 2400 if config.use_memory and config.use_abstractions else 0
    # Screening, startup repair, probe planning and current-data diagnosis
    # use the same bounded work allowances as CognitiveAgent.
    initialization = adaptation + min(len(workspace.bank), config.max_candidates) * count + new_rows + 1700
    reporting = max(4 * (preparation + 2 * length) + 2 * len(workspace.reference),
                    (preparation + 2 * length) * (4 + len(spec.feature_names)))
    observation_budget = preparation + 2 * length
    calibration = {}
    if spec.decision_costs is not None and config.risk_learning:
        validation = max(1, new_rows // 4)
        calibration = risk_calibration_quote(training_rows=new_rows-validation,
            validation_rows=validation, training_trajectories=len(trajectory_blocks(rows)),
            feature_count=len(spec.feature_names), history_depth=depth,
            max_probe_length=spec.max_probe_length, tail_fit=config.tail_fit)
        calibration_rows = min(config.observation_budget // 2,
            calibration["comparison_observation_cost"] + calibration["followup_observation_cost"])
        # Risk learning reserves at most half of the visit's observations.
        observation_budget = max(observation_budget + calibration_rows, 2 * calibration_rows)
        reporting += (calibration["fit_compute_cost"] + calibration["comparison_compute_cost"]
                      + calibration["followup_compute_cost"] + 1200)
    requested_compute = fitting + comparisons + preparation_work + initialization + reporting
    # Candidate search is interruptible; it runs within the remaining one-round
    # allowance. A small configured visit need not fund every optional search
    # branch before trying a real fit. The comparison and report remain priced.
    minimum_compute = reporting + comparisons + 2 * new_rows
    compute_budget = max(minimum_compute, min(config.compute_budget, requested_compute))
    return {"action": action, "max_rounds": 1,
            "observation_budget": observation_budget, "compute_budget": compute_budget,
            "preparation_observations": preparation, "comparison_observations": 2 * length,
            "requested_fit_compute": fitting, "comparison_compute_budget": comparisons,
            "requested_initialization_compute": initialization + preparation_work,
            "learning_compute_budget": max(0, compute_budget-reporting),
            "report_compute_budget": reporting, "calibration": calibration}
