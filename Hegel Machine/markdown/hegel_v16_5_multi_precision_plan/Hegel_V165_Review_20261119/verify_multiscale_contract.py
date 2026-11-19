#!/usr/bin/env python3
"""Reproduce narrowly scoped mathematical counterexamples; not Hegel experiments.

Uses only standard-library arithmetic. Does not access network, invoke LLMs,
modify inputs, or infer empirical project performance.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('multiscale_contract_verified.json'))
    args = p.parse_args()
    epsilon = 1.0
    answers = [0.0, 0.75, 1.5]

    # Strictly convex F: Hessian = I + exp(u+z) v1 v1^T
    #                                  + .1 exp(2u-z) v2 v2^T.
    # Dynamics z_dot=-F_z, input u, conjugate readout p=F_u.
    def fz(u, z):
        return z + math.exp(u+z) - 0.1*math.exp(2*u-z) - 0.9

    def fu(u, z):
        return u + math.exp(u+z) + 0.2*math.exp(2*u-z)

    def hessian(u, z):
        a = math.exp(u+z)
        b = 0.1*math.exp(2*u-z)
        return 1+a+4*b, a-2*b, 1+a+b

    u = math.log(6)
    lo, hi = -10.0, 0.0
    for _ in range(120):
        mid = (lo+hi)/2
        if fz(u, mid) < 0:
            lo = mid
        else:
            hi = mid
    zstar = (lo+hi)/2
    p0, p_frozen, p_relaxed = fu(0, 0), fu(u, 0), fu(u, zstar)
    fuu, fuz, fzz = hessian(u, 0)
    eq_fuu, eq_fuz, eq_fzz = hessian(u, zstar)
    result = {
        'scope': __doc__,
        'epsilon_closeness_not_transitive': {
            'epsilon': epsilon, 'answers': answers,
            'd01': abs(answers[0]-answers[1]),
            'd12': abs(answers[1]-answers[2]),
            'd02': abs(answers[0]-answers[2]),
            '01_and_12_within_epsilon': True,
            '02_within_epsilon': False,
            'interpretation': 'An epsilon-diameter certificate is possible; pairwise epsilon edges do not define an equivalence relation.'
        },
        'aggregation_not_closed': {
            'fine_dynamics': 'xdot1=-x1, xdot2=-2*x2',
            'projection': 'z=x1+x2',
            'fine_states': [[1, 0], [0, 1]],
            'same_coarse_state': 1,
            'different_coarse_derivatives': [-1, -2],
            'coarse_trajectories_at_t1': [math.exp(-1), math.exp(-2)],
            'interpretation': 'No deterministic autonomous z_dot=f(z) can reproduce both preparations.'
        },
        'nonlinear_convex_gradient_counterexample': {
            'potential': 'F=.5*(u^2+z^2)+exp(u+z)+.1*exp(2u-z)-.9*z',
            'dynamics': 'z_dot=-F_z', 'readout': 'p=F_u',
            'initial_u': 0, 'initial_z': 0,
            'initial_stationarity_residual': fz(0, 0),
            'input_step': u, 'F_z_immediately_after_step': fz(u, 0),
            'F_uz_immediately_after_step': fuz,
            'p_dot_immediately_after_step': -fuz*fz(u, 0),
            'F_dot_during_fixed_u': -fz(u, 0)**2,
            'hessian_global_PD_reason': 'I + positive-weight v1*v1^T + positive-weight v2*v2^T',
            'relaxed_z': zstar, 'stationarity_residual': fz(u, zstar),
            'initial_p': p0, 'frozen_p': p_frozen, 'relaxed_p': p_relaxed,
            'finite_increment_relaxed_to_frozen_ratio': (p_relaxed-p0)/(p_frozen-p0),
            'local_frozen_stiffness_at_final_equilibrium': eq_fuu,
            'local_relaxed_stiffness_at_final_equilibrium': eq_fuu-eq_fuz**2/eq_fzz,
            'interpretation': 'Local Schur stiffness bound survives; finite-amplitude monotone readout and endpoint attenuation do not follow from it.'
        },
        'zero_RL_fixed_initial_linkage': {
            'L': 2, 'initial_total_linkage': 3,
            'endpoint_external_linkage': 5,
            'p_minus_I_for_any_ideal_external_flux_history': (5-3)/2,
            'history_gap_at_matched_endpoint_and_same_initial_linkage': 0,
            'interpretation': 'The conserved initial state may vary across preparations; it is not controllable through external-flux history while R stays exactly zero.'
        },
        'SLS_topology_difference': {
            'k1': 2, 'k2': 3, 'eta': 5,
            'spring_k1_series_KelvinVoigt_eta_parallel_k2': {'tau': 5/(2+3), 'rho': 2/(2+3), 'frozen_stiffness': 2},
            'Maxwell_k1_series_eta_parallel_spring_k2': {'tau': 5/2, 'rho': 2/(2+3), 'frozen_stiffness': 5},
            'interpretation': 'Both are SLS realizations but their named component parameters cannot be copied unchanged.'
        },
        'two_Maxwell_arms': {
            'relaxation_modulus': '2*exp(-t)+3*exp(-t/4)',
            'single_exponential': False, 'asymptotic_modulus': 0,
            'interpretation': 'Complete eventual relaxation is independent of single-mode membership.'
        }
    }
    assert abs(fz(0, 0)) < 1e-12
    assert result['nonlinear_convex_gradient_counterexample']['p_dot_immediately_after_step'] > 0
    assert result['nonlinear_convex_gradient_counterexample']['finite_increment_relaxed_to_frozen_ratio'] > 1
    assert 0 < eq_fuu-eq_fuz**2/eq_fzz < eq_fuu
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output':str(args.output), 'p_dot': -fuz*fz(u, 0),
                      'finite_ratio':(p_relaxed-p0)/(p_frozen-p0),
                      'local_stiffness_pair':[eq_fuu-eq_fuz**2/eq_fzz,eq_fuu]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
