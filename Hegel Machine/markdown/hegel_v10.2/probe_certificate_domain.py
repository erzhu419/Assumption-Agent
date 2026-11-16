#!/usr/bin/env python3
"""Read-only soundness probe for a66cd82b.

Default: independently checks the exact algebraic obligations read in
algebraic_proof_certificates.py:~398-524 (NOT execution of the repository).
--repo PATH: additionally calls the actual repository verifier and lab.
No network, paid model calls, or repository writes.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import importlib.util
import json
import hashlib
import sys
from pathlib import Path
from typing import Any

MODEL = dict(objects=2, positions=["0", "1/2"], coupling="1/2", radius="1", direction="right")
TASK = dict(initial_state=["0", "0"], control_bounds=["0", "1"],
            state_bounds=["-1", "0"], budget="1/2", step_cost="0",
            fuel_weights=["1/10", "1"], goal_matrix=[["0", "-1"]], goal_rhs=["-1"])
POTENTIAL = dict(kind="potential_lower_bound_certificate", certificate_id="audit_domain_counterexample",
                 alpha_0="1", alpha=["0", "-1"],
                 farkas_mu_step=["1/2", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0"],
                 farkas_mu_goal=["1"], used_premise_ids=["audit_unverified_state_box"])
WITNESS = dict(kind="reachability_witness_certificate", action_sequence=[["1", "0"],["1", "0"],["0", "0"]])

def reference_check() -> dict[str, Any]:
    n = 2
    A = [[F(1, 2), F(0)], [F(1, 2), F(1)]]
    B = [[F(1), F(0)], [F(0), F(1)]]
    alpha = list(map(F, POTENTIAL['alpha']))
    mu = list(map(F, POTENTIAL['farkas_mu_step']))
    fuel = list(map(F, TASK['fuel_weights']))
    lo, hi = map(F, TASK['state_bounds'])
    ul, uh = map(F, TASK['control_bounds'])
    gw = [alpha[i] - sum(A[r][i] * alpha[r] for r in range(n)) for i in range(n)]
    gu = [-sum(B[r][i] * alpha[r] for r in range(n)) for i in range(n)]
    checks = {
        'nonnegative_step_multipliers': all(x >= 0 for x in mu),
        'state_gradients_match': all(gw[i] == mu[i]-mu[n+i] for i in range(n)),
        'control_gradients_match': all(gu[i] == mu[2*n+i]-mu[3*n+i]+mu[4*n+i]-mu[5*n+i] for i in range(n)),
        'absolute_cost_dual_valid': all(mu[4*n+i]+mu[5*n+i] <= fuel[i] for i in range(n)),
        'step_upper_bound_nonpositive': sum(mu[i]*hi + mu[n+i]*(-lo) + mu[2*n+i]*uh + mu[3*n+i]*(-ul) for i in range(n)) <= 0,
        'goal_gradients_match': [F(x) for x in TASK['goal_matrix'][0]] == alpha,
        'goal_potential_nonpositive': F(POTENTIAL['alpha_0']) + F(TASK['goal_rhs'][0]) <= 0,
        'initial_potential_exceeds_budget': F(1) > F(TASK['budget']),
    }
    assert all(checks.values()), checks
    w = list(map(F, TASK['initial_state']))
    trace = [list(map(str, w))]
    cost = F(0)
    domain_ok = [all(lo <= x <= hi for x in w)]
    for row in WITNESS['action_sequence']:
        u = list(map(F, row))
        assert all(ul <= x <= uh for x in u)
        cost += sum(fuel[i]*abs(u[i]) for i in range(n))
        w = [sum(A[i][j]*w[j] for j in range(n)) + u[i] for i in range(n)]
        trace.append(list(map(str, w)))
        domain_ok.append(all(lo <= x <= hi for x in w))
    goal = -w[1] <= -1
    assert goal and cost <= F(TASK['budget'])
    return {
        'mode': 'independent_exact_arithmetic_check_of_source_obligations',
        'repository_module_executed': False,
        'potential_obligations': checks,
        'reference_potential_check_accepts': all(checks.values()),
        'rational_trace': trace,
        'state_box_respected': domain_ok,
        'witness_cost': str(cost),
        'budget': TASK['budget'],
        'witness_reaches_goal_within_budget_if_box_not_enforced': goal and cost <= F(TASK['budget']),
        'interpretation': 'The conditional Farkas inequality is valid only on the stated box. It cannot certify impossibility for a transition interface that permits this witness outside that box.',
        'existing_eight_benchmark_verdicts_refuted': False,
    }

def actual_check(repo: Path) -> dict[str, Any]:
    roots = [repo, repo / 'Hegel Machine']
    path = next((p/'src/hegel_machine/cognition/algebraic_proof_certificates.py' for p in roots
                 if (p/'src/hegel_machine/cognition/algebraic_proof_certificates.py').is_file()), None)
    if path is None:
        raise FileNotFoundError('Cannot locate algebraic_proof_certificates.py under supplied repository path')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('hegel_a66_proof_under_audit', path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    potential = mod.SymbolicProofKernel.verify_solvability_certificate(POTENTIAL, MODEL, TASK)
    witness = mod.SymbolicProofKernel.verify_solvability_certificate(WITNESS, MODEL, TASK)
    lab = mod.BudgetedControlLab(objects=2, positions=(0.0,0.5), coupling=0.5,
                                radius=1.0, direction='right', control_bounds=(0.0,1.0))
    task_float = dict(TASK, budget=0.5, fuel_weights=[0.1,1.0], goal_rhs=[-1.0])
    lab_result = lab.execute_trace([0.0,0.0], [[1.0,0.0],[1.0,0.0],[0.0,0.0]], task_float)
    lean = mod.emit_lean4_certificate(dict(POTENTIAL, certified_min_budget_rational='1', budget=0.5))
    return {
        'repository_module_executed': True,
        'source_path': str(path),
        'source_git_blob_sha1': hashlib.sha1(b'blob '+str(len(path.read_bytes())).encode()+b'\0'+path.read_bytes()).hexdigest(),
        'expected_a66_git_blob_sha1': '1f9f872a4a4e34e6ea3a93e45a3e06bf3a9369eb',
        'potential_verifier': potential,
        'witness_verifier': witness,
        'lab_trace': lab_result,
        'both_opposite_certificates_accepted': bool(potential.get('verified') and witness.get('verified')),
        'potential_lean_contains_theorem_declaration': 'theorem ' in lean,
        'potential_lean_compiled': False,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, help='Optional local checkout of a66cd82b')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = reference_check()
    if args.repo:
        result['actual_repository_probe'] = actual_check(args.repo)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)

if __name__ == '__main__':
    main()
