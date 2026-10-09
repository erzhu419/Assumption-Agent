"""Abstract experiments and subquestions from evidence-bearing research steps.

These transformations apply only when a trace records the corresponding paid
comparison or repeated setups. They do not invent a task-specific procedure.
"""

from copy import deepcopy
from itertools import groupby


def abstract_recipe(recipe):
    result = deepcopy(recipe)
    comparison = result.get("instrument_comparison")
    if comparison and comparison.get("candidate_channels"):
        candidates = comparison["candidate_channels"]
        losses = comparison.get("losses", {})
        result["instrument_comparison"] = {
            "binding": "compare_available", "selection_rule": "minimum_validation_loss",
            "observed_candidate_count": len(candidates),
            "observed_losses": [losses[name] for name in candidates if name in losses],
        }
    rows = result.get("normalized_rows", ())
    groups = [len(list(group)) for _, group in groupby(rows)]
    if any(size > 1 for size in groups):
        result["repetition_groups"] = groups
    return result


def bind_subquestions(action, channel_names):
    """Turn a learned comparison into questions about currently available views."""
    recipe = action["recipe"]
    comparison = recipe.get("instrument_comparison", {})
    if comparison.get("binding") == "compare_available":
        action["selection_rule"] = comparison["selection_rule"]
        action["subquestions"] = [{
            "question_id": f"{action['node_id']}:instrument:{index}",
            "action": "fit", "channels": (name,),
            "consumes": list(action.get("consumes", ())),
            "depends_on": list(action["depends_on"]),
        } for index, name in enumerate(channel_names)]
    if action["action"] == "replicate" and recipe.get("repetition_groups"):
        action["subquestions"] = [{
            "question_id": f"{action['node_id']}:conditional_variance",
            "action": "estimate_repeat_variance", "groups": recipe["repetition_groups"],
            "consumes": list(action.get("produces", ())),
            "depends_on": [action["node_id"]],
        }]
    return action


def procedure_work(actions, channels=()):
    """Known acquisition demand and fitting questions, before execution.

    Candidate fitting compute is deliberately not guessed from row counts.
    Its actual cost must still be charged by the executing research budget.
    """
    prices = {channel["name"]: channel["cost_per_row"]
              for channel in channels if isinstance(channel, dict)}
    prices.update({channel.name: channel.cost_per_row for channel in channels
                   if hasattr(channel, "cost_per_row")})
    observation = measurement = fits = 0
    for action in actions:
        if "probe" in action:
            observation += action["probe"].cost
            unit_price = (max(prices.values(), default=0) if "channels_from" in action else
                          sum(prices.get(name, 0) for name in action.get("channels", ())))
            measurement += action["probe"].cost * unit_price
        if action["action"] == "fit":
            fits += max(1, len(action.get("subquestions", ())))
    return {"observation_cost": observation, "measurement_cost_upper_bound": measurement,
            "fit_questions": fits}
