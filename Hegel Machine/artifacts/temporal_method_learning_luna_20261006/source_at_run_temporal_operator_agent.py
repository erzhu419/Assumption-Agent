"""Shared, paid-data research operators for the temporal LLM comparison.

The model chooses whether and when to call these bounded numerical methods.
They neither purchase observations nor choose a response candidate for it.
"""

import json
from copy import deepcopy
from math import nextafter
from pathlib import Path
from time import monotonic

from .history_revision import revise_history
from .knowledge_evolution import bind_answers
from .language import Program
from .online_inquiry_llm import InquiryLLMAgent
from .temporal_instrument_research import (
    assess_temporal_instrument, fit_temporal_instrument,
    select_temporal_instrument_probe,
)
from .temporal_research import fit_temporal_response, mse


OPERATOR_INTERFACE = """
All knowledge conditions additionally share these optional numerical operators.
Use action run with an operator key instead of Python code/path execution:
run {operator:'history',base_lags:2,path:'model.py'} compares complete quadratic
control-history bases with base_lags and base_lags+1 past controls. It reports
separate trajectory validation and exports the selected basis fitted on all
paid responses. Unsupported extension keeps the base; this is not confirmation.
run {operator:'instrument',path:'instrument.json'} fits competing bounded
finite-history polynomial and continuous-state explanations of purchased meter
readings only. It exports all fitted hypotheses, not a known apparatus formula.
run {operator:'response',instrument:'instrument.json',output_dir:'shared/candidates'}
fits quadratic response functions of current control, each surviving learned
meter coordinate, and its previous value. It returns EVERY candidate with its
validation error and Python path, without choosing a candidate. Each export is
then refitted on all paid responses and frozen; reported validation precedes
that refit. You decide whether any candidate warrants evaluate or finish.
history and response accept optional validation_records:[2,3], zero-based indices
in laboratory_observations.json, never row_indices. By default, the last two
among the longest paid trajectories are validation; the others are training.
When selecting both longest records would leave only shorter training records,
one long record stays in training and a shorter one takes its validation place.
Records are never split across training and validation. Results identify both
sets. These are paid-data comparisons, not fresh confirmation.
run {operator:'disagreement',instrument:'instrument.json'} proposes a trajectory
separating current instrument hypotheses. It does NOT acquire it; use query if
you choose to purchase it. Optional max_length and serial control its search.
run {operator:'assess_instrument',instrument:'instrument.json',records:[9]}
assesses the frozen instrument hypotheses using meter readings in the selected
already purchased records and updates their support/refutation status. It does
not refit them. Keep the distinction between fitting evidence and new evidence.
The methods use only paid records and public capabilities. Their fitting,
scoring and Python time consume the common limits. No operator buys data,
opens questions, writes your manifest, or finishes the task automatically.
Their output files are ordinary editable files, available equally in all arms.
You can explicitly compose acquisition with these same operators in one query:
query {actions:[trajectory],after_observation:[
  {operator:'assess_instrument',instrument:'shared/meter.json',records:'acquired'},
  {operator:'response',instrument:'shared/meter.json',output_dir:'shared/revised'}]}
A select_actions program may return the same after_observation list alongside
its actions. An explicit query argument overrides the program's list. Each
declared operator runs in order after the whole acquisition succeeds; records:
'acquired' means the exact newly bought laboratory record indices. Operators
see those new paid labels. Their returned candidates still require your choice.
Feedback contains authored steps, their origin (arguments or program path),
resolved arguments, and numerical results. A failed operator stops the list;
successful purchases and earlier results remain available. Failed acquisition
runs no continuation. A live question retains the actual continuation trace
without being automatically answered. This syntax runs only the declared
shared operators, not another acquisition or finish.
"""


_PROGRAM_RUNNER = '''
def _eval(node, features, history):
    op, value, name, lag, children = node
    if op == "const":
        return float(value)
    if op == "feature":
        return float(features[name])
    if op == "available":
        return float(len(history) >= lag)
    if op == "lag":
        if len(history) < lag:
            raise ValueError("insufficient control history")
        return float(history[-lag][name])
    if op == "if":
        return _eval(children[1 if _eval(children[0], features, history) else 2], features, history)
    if op == "object_sum":
        return sum(_eval(child, features, history) for child in children)
    if op == "accumulate":
        state = _eval(children[1], features, history)
        for index, point in enumerate(history):
            state = value * state + _eval(children[0], point, history[:index])
        return value * state + _eval(children[0], features, history)
    if op == "event_delta":
        if len(history) < lag:
            return 0.0
        index = len(history) - lag
        return (_eval(children[0], features, history)
                - _eval(children[0], history[index], history[:index]))
    if op == "window_sum":
        start = max(0, len(history) - lag + 1)
        return _eval(children[0], features, history) + sum(
            _eval(children[0], history[index], history[:index])
            for index in range(start, len(history)))
    left = _eval(children[0], features, history)
    if op == "abs":
        return abs(left)
    if op == "not":
        return float(not left)
    right = _eval(children[1], features, history)
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "min":
        return min(left, right)
    if op == "max":
        return max(left, right)
    if op == "ge":
        return float(left >= right)
    if op == "lt":
        return float(left < right)
    if op == "and":
        return float(bool(left) and bool(right))
    if op == "or":
        return float(bool(left) or bool(right))
    raise ValueError("unsupported operator: " + op)

def predict(row):
    features = row["features"] if isinstance(row, dict) else row.features
    history = row["history"] if isinstance(row, dict) else row.history
    return float(_eval(_PROGRAM, features, history))
'''


def export_temporal_program(program):
    """Export the learned control-history program without repository imports.

    Definition calls are expanded before export. Only the scalar and temporal
    operators used by the instrument/readout repertoire are implemented here.
    The exported predictor has no fitting hook and never reads response labels.
    """
    supported = {"const", "feature", "lag", "available", "if", "object_sum",
                 "accumulate", "event_delta", "window_sum", "abs", "not",
                 "add", "sub", "mul", "min", "max", "ge", "lt", "and", "or"}

    def encode(node):
        if node.op not in supported:
            raise ValueError(f"unsupported temporal export operator: {node.op}")
        if node.op in {"feature", "lag"} and node.name != "u":
            raise ValueError("temporal predictors can read only control u")
        return (node.op, node.value, node.name, node.lag,
                tuple(encode(child) for child in node.children))

    return "# Frozen paid-data predictor; controls and recorded history only.\n_PROGRAM = " + repr(
        encode(program.expanded())) + "\n" + _PROGRAM_RUNNER


def _hypotheses(model):
    return model.get("fields", {}).get("meter", {}).get("hypotheses", [])


def _summary(model):
    keys = ("hypothesis_id", "kind", "status", "fit_loss", "validation_loss")
    return [{key: hypothesis[key] for key in keys if key in hypothesis}
            for hypothesis in _hypotheses(model)]


class TemporalOperatorAgent(InquiryLLMAgent):
    """Ordinary inquiry actions plus the same optional operators in every arm."""

    operator_interface = OPERATOR_INTERFACE

    def _query(self, arguments, laboratory):
        first_question = len(self.questions)
        result = super()._query(arguments, laboratory)
        if result.get("error") or result.get("returncode") or not result.get("record_indices"):
            return result
        continuation = result.get("after_observation", [])
        if continuation == []:
            return result
        current_question = self.questions[-1] if len(self.questions) > first_question else None
        outcomes = result["after_observation_results"] = []
        if not isinstance(continuation, list):
            result["after_observation_error"] = "after_observation must be a list of shared operator arguments"
            if current_question is not None:
                current_question["after_observation_error"] = result["after_observation_error"]
                result["question"] = deepcopy(current_question)
            return result
        for authored in continuation:
            resolved = deepcopy(authored)
            before_compute = self.costs["compute_cost"]
            before_seconds = self.costs["python_execution_seconds"]
            try:
                if not isinstance(resolved, dict) or "operator" not in resolved:
                    raise ValueError("each continuation step needs a shared operator")
                if resolved.get("records") == "acquired":
                    resolved["records"] = list(result["record_indices"])
                feedback = self._execute("run", resolved, laboratory)
            except (ValueError, TypeError, KeyError, OSError) as error:
                feedback = {"operator": resolved.get("operator") if isinstance(resolved, dict) else None,
                            "error": str(error),
                            "compute_cost": self.costs["compute_cost"] - before_compute,
                            "python_execution_seconds": self.costs["python_execution_seconds"] - before_seconds}
            outcome = {"arguments": resolved, **feedback}
            outcomes.append(outcome)
            if current_question is not None:
                current_question.setdefault("after_observation_results", []).append(deepcopy(outcome))
            if "error" in feedback:
                result["after_observation_error"] = feedback["error"]
                if current_question is not None:
                    current_question["after_observation_error"] = feedback["error"]
                break
        if current_question is not None:
            result["question"] = deepcopy(current_question)
        return result

    def _remaining_compute(self):
        return max(0, self.config.compute_budget - self.costs["compute_cost"])

    def _charge(self, units):
        if units > self._remaining_compute():
            raise ValueError("shared research operator compute budget exhausted")
        self.costs["compute_cost"] += units

    def _save(self, path, source):
        destination = self.workspace / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
        self.files[path] = source

    def _save_instrument(self, path, model):
        self._save(path, json.dumps(model, indent=2))

    def _load_instrument(self, arguments):
        path = arguments.get("instrument", "instrument.json")
        return path, json.loads((self.workspace / path).read_text(encoding="utf-8"))

    def _record_indices(self, requested):
        indices = list(requested)
        if (len(indices) != len(set(indices)) or
                any(not isinstance(index, int) or not 0 <= index < len(self.observations)
                    for index in indices)):
            raise ValueError("records must be distinct zero-based laboratory record indices")
        return indices

    def _split(self, arguments):
        records = self.observations
        if not records:
            raise ValueError("buy complete trajectories before fitting a response")
        if "validation_records" in arguments:
            validation = self._record_indices(arguments["validation_records"])
            if not validation and len(records) > 1:
                raise ValueError("choose at least one whole validation trajectory")
        else:
            count = min(2, len(records) - 1)
            longest = sorted(range(len(records)), key=lambda index: (len(records[index]["rows"]), index))
            validation = sorted(longest[-count:]) if count else []
            if count == 2:
                train_long, held_short = longest[-3], longest[-2]
                if len(records[held_short]["rows"]) > len(records[train_long]["rows"]):
                    validation = sorted((set(validation) - {held_short}) | {train_long})
        training = [index for index in range(len(records)) if index not in validation]
        if not training:
            raise ValueError("keep at least one whole trajectory for training")
        train_rows = [row for index in training for row in records[index]["rows"]]
        held_rows = [row for index in validation for row in records[index]["rows"]]
        return train_rows, held_rows, {"training_records": training, "validation_records": validation,
                                      "train_rows": len(train_rows), "holdout_rows": len(held_rows)}

    def _history(self, arguments):
        train_rows, validation_rows, split = self._split(arguments)
        if len(self.observations) > 1 and not validation_rows:
            raise ValueError("history comparison needs at least one paid validation trajectory")
        base_lags = arguments.get("base_lags", 2)
        if not isinstance(base_lags, int) or base_lags < 0:
            raise ValueError("base_lags must be a nonnegative integer")
        widths = [(lags + 2) * (lags + 3) // 2 for lags in (base_lags, base_lags + 1)]
        # Two basis evaluations, two training fits and full-data score passes,
        # followed by the selected all-row refit and its score.
        self._charge((len(train_rows) + len(validation_rows)) * (sum(widths) + 4)
                     + 2 * len(train_rows))
        held = len(split["validation_records"])
        fraction = nextafter(held / len(self.observations), 0.) if held else 1 / 3
        proposal = revise_history(train_rows + validation_rows, base_lags=base_lags,
                                  holdout_fraction=fraction)
        path = arguments.get("path", "model.py")
        self._save(path, proposal.to_code())
        return {"path": path, **proposal.resultdict, "split": {**proposal.resultdict["split"], **split}}

    def _instrument(self, arguments, spec):
        model = fit_temporal_instrument(self.observations, spec,
                                        compute_budget=self._remaining_compute())
        self._charge(model["compute_cost"])
        path = arguments.get("path", "instrument.json")
        self._save_instrument(path, model)
        return {"path": path, "status": model["status"], "support_rows": model["support_rows"],
                "hypotheses": _summary(model), "unmodeled_fields": model.get("unmodeled_fields", {})}

    def _response(self, arguments):
        _, model = self._load_instrument(arguments)
        train_rows, validation_rows, split = self._split(arguments)
        if not validation_rows:
            raise ValueError("response comparison needs separate paid validation trajectories")
        surviving = [hypothesis for hypothesis in _hypotheses(model)
                     if hypothesis.get("status") not in {"refuted", "counterexample"}]
        all_rows = train_rows + validation_rows
        # Ten response columns, fit and score for training; validation scoring;
        # ten all-row columns plus refit and scoring. No uncharged candidate fits.
        per_candidate = 12 * len(train_rows) + len(validation_rows) + 12 * len(all_rows)
        self._charge(per_candidate * len(surviving))
        candidates = []
        for index, hypothesis in enumerate(surviving):
            instrument = Program.from_dict(hypothesis["program"])
            expression, rank, width = fit_temporal_response(train_rows, instrument)
            predictor = bind_answers(expression, {"instrument": instrument})
            train_loss, validation_loss = mse(predictor, train_rows), mse(predictor, validation_rows)
            expression, refit_rank, _ = fit_temporal_response(all_rows, instrument)
            refitted = bind_answers(expression, {"instrument": instrument})
            path = str(Path(arguments.get("output_dir", "shared/candidates")) / f"response_{index:02d}.py")
            self._save(path, export_temporal_program(refitted))
            candidates.append({"hypothesis_id": hypothesis["hypothesis_id"],
                "kind": hypothesis.get("kind"), "instrument_status": hypothesis.get("status"),
                "path": path, "training_rank": rank, "feature_count": width,
                "train_mse": train_loss, "validation_mse": validation_loss,
                "refit_train_mse": mse(refitted, all_rows), "refit_rank": refit_rank})
        return {"split": split, "candidates": candidates}

    def _disagreement(self, arguments, spec):
        _, model = self._load_instrument(arguments)
        result = select_temporal_instrument_probe(model, spec,
            max_length=arguments.get("max_length"), serial=arguments.get("serial", 0),
            compute_budget=self._remaining_compute())
        self._charge(result["compute_cost"])
        # The planner's full condition embeds programs; paths already preserve
        # them. Return only its actionable proposal and numerical forecasts.
        return {key: result[key] for key in ("action", "reason", "sequence_mode", "predictions",
                "predicted_separation", "field_disagreements") if key in result}

    def _assess(self, arguments, spec):
        path, model = self._load_instrument(arguments)
        indices = self._record_indices(arguments["records"])
        if not indices:
            raise ValueError("choose paid records containing meter readings")
        result = assess_temporal_instrument(model, [self.observations[index] for index in indices],
            spec, compute_budget=self._remaining_compute())
        self._charge(result["compute_cost"])
        self._save_instrument(path, model)
        return {**result, "path": path, "records": indices, "status": model["status"],
                "hypotheses": _summary(model)}

    def _execute(self, action, arguments, laboratory):
        if action != "run" or "operator" not in arguments:
            return super()._execute(action, arguments, laboratory)
        remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining <= 0:
            raise ValueError("Python execution time budget exhausted")
        before, started = self.costs["compute_cost"], monotonic()
        operator = arguments["operator"]
        try:
            spec = laboratory.public_spec()
            if operator == "history":
                result = self._history(arguments)
            elif operator == "instrument":
                result = self._instrument(arguments, spec)
            elif operator == "response":
                result = self._response(arguments)
            elif operator == "disagreement":
                result = self._disagreement(arguments, spec)
            elif operator == "assess_instrument":
                result = self._assess(arguments, spec)
            else:
                raise ValueError(f"unknown shared research operator: {operator}")
        finally:
            elapsed = monotonic() - started
            self.costs["python_execution_seconds"] += elapsed
        if elapsed > remaining:
            raise ValueError("shared research operator exceeded remaining Python execution time")
        return {"operator": operator, **result, "compute_cost": self.costs["compute_cost"] - before,
                "python_execution_seconds": elapsed}
