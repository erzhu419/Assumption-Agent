"""Compare executable responses on one actually purchased observation view."""

from .evaluation import loss


def compare_observed_responses(entries, rows, budget, risk_quantile=.5):
    """Score frozen responses on one actual purchased view without fitting it."""
    eligible = [item for item in entries if all(
        item["candidate"].program.required_features() <= set(row.features)
        for row in rows)]
    if not rows or not eligible or not budget.spend(len(rows) * len(eligible)):
        return []
    return [{"entry": item, "loss": loss(item["candidate"].program, rows,
        item["fallback"], risk_quantile)} for item in eligible]
