"""Price unresolved response hypotheses against the current paid explanation."""

from copy import deepcopy
from statistics import mean, stdev

from .evaluation import predict
from .language import Program
from .models import Observation
from .object_continuation import decode_entry


def _current_quote(previous, question):
    quote = question.get("current_quote")
    if (quote is not None and quote["program"] == previous["program"]
            and quote["fallback"] == previous["fallback"]
            and quote["paid_batches"] == len(previous["batches"])):
        return quote
    return None


def _needs_repricing(previous, question):
    if _current_quote(previous, question) is not None:
        return False
    if question.get("current_quote") is not None:
        return True
    return (previous["program"] != question["parent"]["candidate"]["program"]
        or not question["probe_ids"]
        or question["probe_ids"][-1] != previous["batches"][-1]["probe"]["probe_id"])


def refresh_response_question_quotes(previous, questions, budget, *,
                                    minimum_improvement=1e-7, is_refuted=None):
    """Pay once to compare frozen alternatives on a changed task's shared data.

    This evidence prices research; it never becomes fresh confirmation. Repeat
    responses to the same commands form one paired loss per trajectory point.
    """
    if previous["selected_channels"]:
        return []
    incumbent = Program.from_dict(previous["program"])
    if any(name == "__target__" for name, _ in incumbent.observed_variables()):
        return []
    pending = [question for question in questions
               if question["status"] == "pending" and _needs_repricing(previous, question)]
    if not pending:
        return []
    groups, rows = [], []
    for batch in previous["batches"]:
        sequence = batch["probe"]["sequence"]
        group = next((item for item in groups if item["sequence"] == sequence), None)
        if group is None:
            group = {"sequence": sequence, "indices": [[] for _ in sequence]}
            groups.append(group)
        for index, item in enumerate(batch["observations"]):
            group["indices"][index].append(len(rows))
            rows.append(Observation(dict(item["features"]), item["target"],
                tuple(dict(point) for point in item["history"])))
    paired_indices = [indices for group in groups for indices in group["indices"]]
    if len(paired_indices) < 2:
        return []
    predictions, events = [], []

    def cached(program, fallback):
        return next((item[2] for item in predictions
                     if item[0] == program and item[1] == fallback), None)

    def measured(program, fallback):
        values = cached(program, fallback)
        if values is None:
            budget.spend(len(rows))
            losses = [(row.target - predict(program, row, fallback)) ** 2 for row in rows]
            values = [mean(losses[index] for index in indices) for indices in paired_indices]
            predictions.append((program, fallback, values))
        return values

    for question in pending:
        branches = [decode_entry(question[key]) for key in ("parent", "candidate")]
        alternatives = [(entry["candidate"].program, entry.get("fallback", previous["fallback"]))
            for entry in branches if entry["candidate"].program != incumbent
            and not (is_refuted is not None and is_refuted(entry["candidate"].program))]
        required = [(incumbent, previous["fallback"])]
        for program, fallback in alternatives:
            if (program, fallback) not in required:
                required.append((program, fallback))
        work = sum(len(rows) for program, fallback in required if cached(program, fallback) is None)
        if work > budget.remaining:
            continue
        before = budget.spent
        reference = measured(incumbent, previous["fallback"])
        comparisons = []
        for program, fallback in alternatives:
            candidate = measured(program, fallback)
            gains = [old - new for old, new in zip(reference, candidate)]
            gain, margin = mean(gains), 3 * stdev(gains) / len(gains) ** .5
            expected_gain = max(0., gain) + max(0., margin - abs(gain)) / 2
            comparisons.append({"program": program.to_dict(), "observed_gain": gain,
                "uncertainty_margin": margin, "candidate_loss": mean(candidate),
                "expected_gain": expected_gain})
        best = max(comparisons, key=lambda item: item["expected_gain"], default={
            "observed_gain": 0., "uncertainty_margin": 0., "candidate_loss": mean(reference),
            "expected_gain": 0.})
        quote = {**best, "program": deepcopy(previous["program"]),
            "fallback": previous["fallback"], "paid_batches": len(previous["batches"]),
            "reference_loss": mean(reference), "rows": len(reference), "paid_rows": len(rows),
            "compute_cost": budget.spent - before, "alternatives": comparisons,
            "basis": "current_incumbent_paid_evidence"}
        question["current_quote"] = quote
        events.append({"action": "reprice_response_question", "question_id": question["question_id"],
            **deepcopy(quote), "worth_comparing": quote["expected_gain"] > minimum_improvement})
    return events


def response_question_offer(previous, question, max_probe_length, remaining,
                            minimum_improvement=1e-7, *, resource_limits, is_refuted=None):
    """Quote a fresh trajectory and repeat using the current paid comparison."""
    if question["status"] != "pending" or previous["selected_channels"]:
        return None
    # If an attempted visit could not buy its comparison, restoring the same
    # session must not retry it until newly visited tasks contribute resources.
    deferred = question.get("deferred_resources")
    if deferred is not None and all(remaining[key] <= value for key, value in deferred.items()):
        return None
    quote = _current_quote(previous, question)
    if quote is None and _needs_repricing(previous, question):
        return None
    evidence = quote if quote is not None else question
    if quote is not None and is_refuted is not None:
        eligible = [item for item in quote["alternatives"]
                    if not is_refuted(Program.from_dict(item["program"]))]
        if not eligible:
            return None
        evidence = {**quote, **max(eligible, key=lambda item: item["expected_gain"])}
    gain = evidence["observed_gain"]
    margin = max(0., evidence["uncertainty_margin"])
    expected_gain = max(0., gain) + max(0., margin - abs(gain)) / 2
    if expected_gain <= minimum_improvement:
        return None
    parent, candidate = decode_entry(question["parent"]), decode_entry(question["candidate"])
    programs = [entry["candidate"].program for entry in (parent, candidate)]
    if is_refuted is not None and all(is_refuted(program) for program in programs):
        return None
    incumbent = Program.from_dict(previous["program"])
    if any(name == "__target__" for name, _ in incumbent.observed_variables()):
        return None
    additional_incumbent = incumbent not in programs
    if additional_incumbent:
        programs.append(incumbent)
    minimum_length = max(program.minimum_history_depth for program in programs) + 2
    if minimum_length > max_probe_length:
        return None
    old_length = max((len(batch["probe"]["sequence"]) for batch in previous["batches"]), default=3)
    length = min(max_probe_length, max(minimum_length,
        question.get("next_probe_length", min(24, max(3, old_length)))))
    resources = {"observation_cost": 2 * length, "measurement_cost": 0,
                 "compute_cost": (13 if additional_incumbent else 10) * length}
    if not all(remaining[key] >= amount for key, amount in resources.items()):
        return None
    parent_loss = next((value for value in (evidence.get("reference_loss"),
        parent.get("validation_loss"), parent.get("loss"), previous.get("selection_loss"))
        if value is not None), None)
    if parent_loss is None:
        return None
    cost_fraction = (resources["observation_cost"] / resource_limits["observation_budget"]
                     + resources["compute_cost"] / resource_limits["compute_budget"])
    relative_gain = expected_gain / (max(parent_loss, minimum_improvement) * cost_fraction)
    return {"candidate": {"response_question": deepcopy(question), "probe_length": length},
            "question_id": question["question_id"], "basis": "unresolved_response_comparison",
            "old_evidence_gain": gain, "old_evidence_rows": evidence["rows"],
            "expected_loss_reduction": expected_gain, "loss_kind": "squared_error",
            "relative_gain": relative_gain, "minimum_resources": resources}
