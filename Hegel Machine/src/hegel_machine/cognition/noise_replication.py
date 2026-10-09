"""Distinguish reproducible model error from variation in a paid repeat.

The caller repeats a complete control trajectory as a new experiment. These
comparisons use its observed inputs, not a laboratory label or hidden state.
"""

from statistics import mean, stdev

from .evaluation import point_loss, predict


def _inputs(row, reads_target):
    history = tuple({name: value for name, value in past.items()
                     if reads_target or name != "__target__"} for past in row.history)
    return row.features, history


def assess_noise_replication(program, original, repeated, budget, *,
                             minimum_improvement=1e-7, risk_quantile=.5,
                             fallback=0.):
    """Challenge a noise explanation using matched, independent acquisitions.

    Equal current inputs alone are insufficient for a temporal mechanism: the
    entire observed input history must match. Historical responses are outcomes,
    unless the predictor actually reads them, including through nested calls.
    An inconclusive repeat supplies no evidence that a model's error is noise.
"""
    original, repeated = list(original), list(repeated)
    start = budget.spent
    reads_target = any(name == "__target__" for name, _ in program.observed_variables())
    result = {"repeatable_error": False, "noise_supported": False,
        "reason": "insufficient_matched_rows",
        "matched_rows": 0, "original_rows": len(original), "repeated_rows": len(repeated),
        "target_history_required": reads_target, "fixed_quantile": risk_quantile,
        "repeat_loss": None, "repeat_mse": None, "original_model_loss": None,
        "repeated_model_loss": None, "repeatable_loss_limit": None,
        "repeat_noise_variance": None, "reproducible_mse": None,
        "reproducible_mse_margin": None, "reproducible_mse_lower_bound": None,
        "compute_cost": 0}

    def finish(reason, repeatable=False, noise_supported=False):
        result.update(reason=reason, repeatable_error=repeatable,
                      noise_supported=noise_supported,
                      compute_cost=budget.spent - start)
        return result

    if len(original) != len(repeated):
        return finish("different_trajectory_length")
    if not budget.spend(len(original)):
        return finish("insufficient_budget")

    pairs = [(left, right) for left, right in zip(original, repeated)
             if _inputs(left, reads_target) == _inputs(right, reads_target)]
    result["matched_rows"] = len(pairs)
    if len(pairs) < 2:
        return finish("insufficient_matched_rows")
    if len(pairs) != len(original):
        return finish("unmatched_trajectory_inputs")
    if not budget.spend(2 * len(pairs)):
        return finish("insufficient_budget")
    q = risk_quantile
    repeat_loss = mean((point_loss(left.target, right.target, q)
                       + point_loss(right.target, left.target, q)) / 2
                       for left, right in pairs)
    residuals = [(left.target - predict(program, left, fallback),
                  right.target - predict(program, right, fallback)) for left, right in pairs]
    original_loss = mean(point_loss(0., left, q) for left, _ in residuals)
    repeated_loss = mean(point_loss(0., right, q) for _, right in residuals)
    result.update(repeat_loss=repeat_loss,
        repeat_mse=mean((left.target - right.target) ** 2 for left, right in pairs),
        original_model_loss=original_loss, repeated_model_loss=repeated_loss,
        repeatable_loss_limit=minimum_improvement + repeat_loss)
    if repeat_loss > minimum_improvement:
        # Variation across inputs can make the residual-product margin wide
        # even when both responses reproduce a much larger prediction error.
        # First ask whether the measured within-input variation can account
        # for that error in the task's own loss units.
        if q == .5 and min(original_loss, repeated_loss) > result["repeatable_loss_limit"]:
            return finish("repeat_variation_does_not_explain_error", True)
        if q == .5:
            if not budget.spend(len(pairs)):
                return finish("insufficient_budget")
            # Independent responses at the same observed input have residuals
            # b + e1 and b + e2. Their product estimates b^2 without counting
            # independent observation variance as predictable model error.
            # A finite paired-row margin keeps weak random correlations from
            # automatically refuting an otherwise adequate noisy predictor.
            products = [left * right for left, right in residuals]
            reproducible = mean(products)
            margin = 3 * stdev(products) / len(products) ** .5
            result.update(repeat_noise_variance=.5 * result["repeat_mse"],
                reproducible_mse=reproducible, reproducible_mse_margin=margin,
                reproducible_mse_lower_bound=reproducible - margin)
            if reproducible - margin > minimum_improvement:
                return finish("reproducible_residual_despite_noise", True)
        return finish("response_varies_on_repeat", noise_supported=True)
    # An exactly repeated error is judged against the requested task loss.
    # Any additional allowance comes from observed repeat variation in those
    # same loss units, rather than a fixed multiplier of the task tolerance.
    if min(original_loss, repeated_loss) <= result["repeatable_loss_limit"]:
        return finish("no_material_repeatable_error")
    return finish("reproducible_unexplained_response", True)


def compare_response_refits(program, candidates, original, repeated, budget, *,
                            fallback=0., minimum_improvement=1e-7):
    """Select a promising squared-loss refit on a paid independent repeat pair.

    Each input position contributes one average paired gain, not two independent
    samples. Candidate fitting must precede this comparison; the selected answer
    still requires a new frozen-program confirmation from the caller.
    """
    original, repeated, candidates = list(original), list(repeated), list(candidates)
    start = budget.spent
    result = {"selected": None, "comparisons": [], "matched_rows": 0, "compute_cost": 0}

    def finish(reason):
        result.update(reason=reason, compute_cost=budget.spent - start)
        return result

    if not candidates:
        return finish("no_candidates")
    if len(original) != len(repeated):
        return finish("different_trajectory_length")
    count = len(original)
    if not budget.spend(count):
        return finish("insufficient_budget")
    reads_target = any(name == "__target__" for name, _ in program.observed_variables())
    pairs = [(left, right) for left, right in zip(original, repeated)
             if _inputs(left, reads_target) == _inputs(right, reads_target)]
    result["matched_rows"] = len(pairs)
    if len(pairs) < 2:
        return finish("insufficient_matched_rows")
    if len(pairs) != count:
        return finish("unmatched_trajectory_inputs")
    if not budget.spend(2 * count):
        return finish("insufficient_budget")
    reference_losses = [((left.target - predict(program, left, fallback)) ** 2
                         + (right.target - predict(program, right, fallback)) ** 2) / 2
                        for left, right in pairs]
    best_gain, incomplete = None, False
    for entry in candidates:
        before = budget.spent
        candidate = entry["candidate"]
        event = {"candidate_id": candidate.candidate_id, "kind": entry["kind"],
                 "paired_rows": count, "supported": False, "compute_cost": 0}
        requires_target = any(name == "__target__" for name, _ in candidate.program.observed_variables())
        if requires_target and not reads_target:
            if not budget.spend(count):
                incomplete = True
                break
            if any(_inputs(left, True) != _inputs(right, True) for left, right in pairs):
                event.update(reason="unmatched_target_history", compute_cost=budget.spent - before)
                result["comparisons"].append(event)
                continue
        if not budget.spend(3 * count):
            event.update(reason="insufficient_budget", compute_cost=budget.spent - before)
            result["comparisons"].append(event)
            incomplete = True
            break
        candidate_fallback = entry.get("fallback", fallback)
        candidate_losses = [((left.target - predict(candidate.program, left, candidate_fallback)) ** 2
                            + (right.target - predict(candidate.program, right, candidate_fallback)) ** 2) / 2
                            for left, right in pairs]
        gains = [reference - revised for reference, revised in zip(reference_losses, candidate_losses)]
        gain = mean(gains)
        margin = 3 * stdev(gains) / count ** .5
        supported = gain - margin > minimum_improvement
        event.update(gain=gain, margin=margin, lower_bound=gain - margin, supported=supported,
            reference_loss=mean(reference_losses), candidate_loss=mean(candidate_losses),
            reason="supported_paired_gain" if supported else "no_supported_paired_gain",
            compute_cost=budget.spent - before)
        result["comparisons"].append(event)
        if supported and (best_gain is None or gain > best_gain):
            result["selected"], best_gain = entry, gain
    return finish("supported_refit" if result["selected"] is not None else
                  "insufficient_budget" if incomplete else "no_supported_refit")
