"""Refit observed control interactions and locally revise their time offsets."""

from dataclasses import replace
from itertools import product

from .language import Program
from .meta_prior_research import _sources
from .shared_transfer_fitting import SharedTransferFitter


def _key(factors, gate=0):
    factors = tuple(sorted(factors))
    return factors, gate if gate > max((lag for _, lag in factors), default=0) else 0


def _polynomial(program, budget):
    """Expand only arithmetic already present in a retained finite-history law."""
    if not budget.spend(1):
        return None
    if program.op == "call":
        return _polynomial(program.expanded(), budget)
    if program.op == "const":
        return {(_key(())): program.value} if program.value else {}
    if program.op in {"feature", "lag"}:
        return {_key(((program.name, program.lag if program.op == "lag" else 0),)): 1.}
    if program.op == "available":
        return {_key((), program.lag): 1.}
    if program.op == "if":
        condition, body, otherwise = program.children
        if condition.op != "available" or otherwise.op != "const" or otherwise.value:
            return None
        terms = _polynomial(body, budget)
        if terms is None:
            return None
        result = {}
        for (factors, gate), coefficient in terms.items():
            key = _key(factors, max(gate, condition.lag))
            result[key] = result.get(key, 0.) + coefficient
        return {key: coefficient for key, coefficient in result.items() if coefficient}
    if program.op not in {"add", "sub", "mul"}:
        return None
    left, right = [_polynomial(child, budget) for child in program.children]
    if left is None or right is None:
        return None
    if program.op in {"add", "sub"}:
        result = dict(left)
        for key, coefficient in right.items():
            result[key] = result.get(key, 0.) + coefficient * (-1. if program.op == "sub" else 1.)
        if len(result) > 64:
            return None
        return {key: coefficient for key, coefficient in result.items() if coefficient}
    result = {}
    for ((a, a_gate), a_value), ((b, b_gate), b_value) in product(left.items(), right.items()):
        if not budget.spend(1):
            return None
        key = _key((*a, *b), max(a_gate, b_gate))
        result[key] = result.get(key, 0.) + a_value * b_value
        if len(result) > 64:
            return None
    return {key: coefficient for key, coefficient in result.items() if coefficient}


def _term(key):
    factors, gate = key
    result = Program("const", value=1.)
    for name, lag in factors:
        factor = (Program("if", children=(Program("available", lag=lag),
                    Program("lag", name=name, lag=lag), Program("const", value=0.)))
                  if lag else Program("feature", name=name))
        result = Program("mul", children=(result, factor)).simplified()
    if gate:
        result = Program("if", children=(Program("available", lag=gate), result, Program("const", value=0.)))
    return result


class RetainedControlRevisions:
    """Finite candidates inherit interaction shapes, without trusting old weights.

    Shift a suffix of observed lags, retaining the original terms as competing
    coefficients. This can move an instrument delay while a response's shorter
    history remains intact. It does not enumerate a new polynomial grammar.
    """

    def __init__(self, calls, spec, budget, *, max_history):
        self.coordinates, self.fitter = [], SharedTransferFitter()
        seen = set()
        controls = set(spec.feature_names)
        for call in calls:
            if not {name for name, _ in call.observed_variables()} <= controls:
                continue
            polynomial = _polynomial(call, budget)
            if polynomial is None:
                continue
            keys = tuple(sorted(key for key in polynomial if key != ((), 0)))
            if not keys or keys in seen:
                continue
            seen.add(keys)
            lags = sorted({lag for factors, _ in keys for _, lag in factors if lag})
            if not lags:
                continue
            sources = _sources(call)
            self.coordinates.append({"keys": keys, "sources": sources, "lag_mapping": {}})
            for threshold in lags:
                for offset in range(1 - threshold, max_history - max(lags) + 1):
                    if offset == 0:
                        continue
                    if not budget.spend(len(keys)):
                        return
                    mapping = {lag: lag + offset for lag in lags if lag >= threshold}
                    shifted = {_key(tuple((name, mapping.get(lag, lag)) for name, lag in factors),
                                         mapping.get(gate, gate)) for factors, gate in keys}
                    combined = tuple(sorted(set(keys) | shifted))
                    if combined in seen:
                        continue
                    seen.add(combined)
                    self.coordinates.append({"keys": combined, "sources": sources, "lag_mapping": mapping})

    def propose(self, rows, budget, *, temporal=False):
        results = []
        for item in self.coordinates:
            if bool(item["lag_mapping"]) != temporal or len(item["keys"]) + 1 > len(rows):
                continue
            result = self.fitter.fit([_term(key) for key in item["keys"]], rows, budget,
                "control_history_revision" if temporal else "control_mechanism_refit", family="T08")
            if result is None:
                break
            result["method_source_ids"] = item["sources"]
            result["candidate"] = replace(result["candidate"], parents=tuple(item["sources"]))
            result["control_revision"] = {"lag_mapping": item["lag_mapping"],
                                           "coordinate_count": len(item["keys"])}
            results.append(result)
        return results
