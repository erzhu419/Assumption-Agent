"""Independent probes of verbatim small function bodies from eee787bf.

NOT a checkout or end-to-end execution of the repository. Program is replaced
by a minimal immutable dataclass implementing only const/arg/lt for these probes.
Source: Hegel Machine/src/hegel_machine/cognition/branch_reconciliation.py
Functions: _programs_algebraically_identical, anti_unify_programs,
propagate_premise_retraction_invalidation. Retrieved through GitHub connector.
"""
from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class Program:
    op: str
    value: float | None = None
    name: str | None = None
    lag: int = 0
    children: tuple = ()
    definition: Any = None
    def predict(self, x):
        if self.op == 'const': return float(self.value or 0.0)
        if self.op == 'arg': return float(x[self.name])
        if self.op == 'lt': return float(self.children[0].predict(x) < self.children[1].predict(x))
        raise ValueError(self.op)

# Source function body, unchanged except omission of docstring.
def _programs_algebraically_identical(pa: Program, pb: Program, *, tol: float = 1e-9) -> bool:
    if pa.op != pb.op or pa.name != pb.name or pa.lag != pb.lag:
        return False
    if pa.op == "const":
        return abs(float(pa.value or 0.0) - float(pb.value or 0.0)) <= tol
    if pa.value is not None or pb.value is not None:
        if pa.value is None or pb.value is None:
            return False
        if abs(float(pa.value) - float(pb.value)) > tol:
            return False
    if len(pa.children) != len(pb.children):
        return False
    if pa.op in {"add", "mul", "and", "or"} and len(pa.children) == 2:
        direct = all(
            _programs_algebraically_identical(ca, cb, tol=tol)
            for ca, cb in zip(pa.children, pb.children)
        )
        if direct:
            return True
        return _programs_algebraically_identical(
            pa.children[0], pb.children[1], tol=tol
        ) and _programs_algebraically_identical(
            pa.children[1], pb.children[0], tol=tol
        )
    return all(
        _programs_algebraically_identical(ca, cb, tol=tol)
        for ca, cb in zip(pa.children, pb.children)
    )

# Source function body, unchanged except omission of docstring.
def anti_unify_programs(prog_a: Program, prog_b: Program) -> tuple[Program, list[dict[str, Any]]] | None:
    slots: list[dict[str, Any]] = []
    def _walk(na: Program, nb: Program, path: tuple[int, ...]) -> Program | None:
        if na.op == "const" and nb.op == "const":
            va = float(na.value or 0.0)
            vb = float(nb.value or 0.0)
            if abs(va - vb) <= 1e-12:
                return Program("const", value=va)
            slot_name = f"theta_{len(slots)}"
            slots.append({"slot": slot_name, "path": path, "value_a": va, "value_b": vb})
            return Program("arg", name=slot_name)
        if na.op != nb.op or na.name != nb.name or na.lag != nb.lag:
            return None
        if len(na.children) != len(nb.children):
            return None
        if na.value != nb.value:
            return None
        new_children = []
        for idx, (ca, cb) in enumerate(zip(na.children, nb.children)):
            merged_c = _walk(ca, cb, (*path, idx))
            if merged_c is None:
                return None
            new_children.append(merged_c)
        return Program(na.op, value=na.value, name=na.name, lag=na.lag,
                       children=tuple(new_children), definition=na.definition)
    skeleton = _walk(prog_a, prog_b, ())
    if skeleton is None:
        return None
    return skeleton, slots

# Source function body, unchanged except omission of docstring/comments.
def propagate_premise_retraction_invalidation(claims: list[dict[str, Any]], *,
        retracted_premise_or_claim_ids: set[str] | list[str] = ()) -> list[dict[str, Any]]:
    retracted = set(retracted_premise_or_claim_ids or ())
    updated_claims = [deepcopy(c) for c in claims]
    by_id = {str(c["claim_id"]): c for c in updated_claims}
    grounded: set[str] = set()
    for cid, c in by_id.items():
        if cid in retracted or c.get("status") == "revoked":
            continue
        deps = list(c.get("depends_on_claim_ids") or ())
        req_prems = set(c.get("required_premise_ids") or ())
        if req_prems & retracted:
            continue
        if not deps and c.get("independent_empirical_support", True):
            grounded.add(cid)
        elif c.get("independent_empirical_support", False) and not (req_prems & retracted):
            grounded.add(cid)
    changed = True
    while changed:
        changed = False
        for cid, c in by_id.items():
            if cid in grounded or cid in retracted or c.get("status") == "revoked":
                continue
            req_prems = set(c.get("required_premise_ids") or ())
            if req_prems & retracted:
                continue
            deps = list(c.get("depends_on_claim_ids") or ())
            if deps and all(d in grounded for d in deps):
                grounded.add(cid)
                changed = True
    for cid, c in by_id.items():
        if cid in grounded:
            c["support_validity"] = "valid_grounded"
        elif cid in retracted:
            c["support_validity"] = "directly_retracted"
            c["status"] = "stale_retracted"
        else:
            deps = list(c.get("depends_on_claim_ids") or ())
            visited: set[str] = set()
            stack = list(deps)
            in_cycle = False
            while stack:
                cur = stack.pop()
                if cur == cid:
                    in_cycle = True
                    break
                if cur not in visited and cur in by_id:
                    visited.add(cur)
                    stack.extend(by_id[cur].get("depends_on_claim_ids") or ())
            if in_cycle:
                c["support_validity"] = "invalid_ungrounded_cycle"
                c["status"] = "unverified_cyclic_support"
            else:
                c["support_validity"] = "invalidated_by_upstream_retraction"
                c["status"] = "stale_dependency_invalidated"
    return updated_claims


def predicate(t):
    return Program('lt', children=(Program('const', value=t), Program('arg', name='x')))

p0, p1 = predicate(0.0), predicate(5e-10)
x = {'x': 2.5e-10}
au = anti_unify_programs(predicate(.22), predicate(.46))
assert au is not None
third = anti_unify_programs(au[0], predicate(.34))
unknown = propagate_premise_retraction_invalidation([{'claim_id':'new_root','status':'candidate'}])[0]
cycle = propagate_premise_retraction_invalidation([
    {'claim_id':'A', 'depends_on_claim_ids':['B'],'independent_empirical_support':False},
    {'claim_id':'B', 'depends_on_claim_ids':['A'],'independent_empirical_support':False}])
result = {
    'fixed_commit':'eee787bfdca8b87f38095adff0664955098e7d8a',
    'execution_scope':'verbatim_selected_function_bodies_with_minimal_Program_adapter',
    'full_repository_executed':False,
    'threshold_equivalence': {
        'function_returns_identical':_programs_algebraically_identical(p0,p1),
        'witness_x':x['x'], 'prediction_a':p0.predict(x),'prediction_b':p1.predict(x),
        'identical_control':_programs_algebraically_identical(p0,p0),
        'interpretation':'Tolerance on constants is not exact Boolean-predicate equivalence.'},
    'incremental_anti_unification': {
        'two_thresholds_succeed': au is not None,
        'slot_names':[s['slot'] for s in au[1]],
        'third_compatible_threshold_returns_none':third is None,
        'interpretation':'Existing theta arg cannot be unified with the next concrete constant.'},
    'unsupported_root':unknown,
    'unsupported_cycle_control':cycle,
}
out=Path(__file__).with_name('probe_results.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(out.read_text())
