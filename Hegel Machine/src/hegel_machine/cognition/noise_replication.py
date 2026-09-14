"""Distinguish reproducible model error from variation in a paid repeat.

The caller repeats a complete control trajectory as a new experiment. These
comparisons use its observed inputs, not a laboratory label or hidden state.
"""

from statistics import mean

from .evaluation import point_loss, predict


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
    result = {"repeatable_error": False, "reason": "insufficient_matched_rows",
        "matched_rows": 0, "original_rows": len(original), "repeated_rows": len(repeated),
        "target_history_required": reads_target, "fixed_quantile": risk_quantile,
        "repeat_loss": None, "repeat_mse": None, "original_model_loss": None,
        "repeated_model_loss": None, "compute_cost": 0}

    def finish(reason, repeatable=False):
        result.update(reason=reason, repeatable_error=repeatable,
                      compute_cost=budget.spent - start)
        return result

    if len(original) != len(repeated):
        return finish("different_trajectory_length")
    if not budget.spend(len(original)):
        return finish("insufficient_budget")

    def inputs(row):
        history = tuple({name: value for name, value in past.items()
                         if reads_target or name != "__target__"} for past in row.history)
        return row.features, history

    pairs = [(left, right) for left, right in zip(original, repeated)
             if inputs(left) == inputs(right)]
    result["matched_rows"] = len(pairs)
    if len(pairs) < 2:
        return finish("insufficient_matched_rows")
    if not budget.spend(2 * len(pairs)):
        return finish("insufficient_budget")
    q = risk_quantile
    repeat_loss = mean((point_loss(left.target, right.target, q)
                       + point_loss(right.target, left.target, q)) / 2
                       for left, right in pairs)
    original_loss = mean(point_loss(predict(program, left, fallback), left.target, q)
                         for left, _ in pairs)
    repeated_loss = mean(point_loss(predict(program, right, fallback), right.target, q)
                         for _, right in pairs)
    result.update(repeat_loss=repeat_loss,
        repeat_mse=mean((left.target - right.target) ** 2 for left, right in pairs),
        original_model_loss=original_loss, repeated_model_loss=repeated_loss)
    if repeat_loss > minimum_improvement:
        return finish("response_varies_on_repeat")
    if min(original_loss, repeated_loss) <= 10 * minimum_improvement:
        return finish("no_material_repeatable_error")
    return finish("reproducible_unexplained_response", True)
