#!/usr/bin/env python3
"""Deterministic arithmetic checks of two proposed V16 invariants.

These are analytic counterexamples, not a rerun of Hegel or a new model study.
"""
import json
import math
from pathlib import Path


def main():
    rows = []
    # After reversal g=dPhi_ext/dt=-1, L=R=1, I(0)=-1.
    # The exact RL solution is I(t)=1-2 exp(-t).
    for t in [0.0, 0.1, 0.3, math.log(2.0), 1.0]:
        current = 1.0 - 2.0 * math.exp(-t)
        derivative = 2.0 * math.exp(-t)
        g = -1.0
        induced_flux = current
        external_emf = -g
        energy_derivative = current * derivative
        input_power = external_emf * current
        dissipated_power = current * current
        rows.append({
            't': t, 'I': current, 'dI_dt': derivative,
            'dPhi_ext_dt': g, 'Phi_ind': induced_flux,
            'claimed_opposition_product': g * induced_flux,
            'external_emf_opposition_product': g * external_emf,
            'rl_equation_residual': derivative + current + g,
            'energy_balance_residual': energy_derivative - (input_power - dissipated_power),
            'dissipated_power': dissipated_power,
        })
    diffusion = []
    # Two-site closed diffusion: c1'=-(c1-c2), c2'=c1-c2.
    # c1(0)=2, c2(0)=0. Observations are nonlinear y_i=c_i**2.
    for t in [0.0, 0.1, 0.5, 1.0, 2.0]:
        z = math.exp(-2.0*t)
        c1, c2 = 1.0+z, 1.0-z
        diffusion.append({'t':t,'c1':c1,'c2':c2,
                          'latent_conserved_sum':c1+c2,
                          'y1':c1*c1,'y2':c2*c2,
                          'raw_sensor_sum':c1*c1+c2*c2})
    assert all(abs(x['rl_equation_residual']) < 1e-12 for x in rows)
    assert all(abs(x['energy_balance_residual']) < 1e-12 for x in rows)
    assert all(x['dissipated_power'] >= 0 for x in rows)
    assert rows[1]['claimed_opposition_product'] > 0
    assert all(abs(x['latent_conserved_sum']-2.0) < 1e-12 for x in diffusion)
    assert diffusion[0]['raw_sensor_sum'] != diffusion[-1]['raw_sensor_sum']
    result = {
        'scope':'Analytic counterexamples evaluated numerically; no Hegel or LLM execution.',
        'rl':{'assumptions':{'L':1.0,'R':1.0,'initial_current':-1.0,'external_flux_rate_after_reversal':-1.0},
              'exact_solution':'I(t) = 1 - 2 exp(-t)',
              'violates_claim_for':'0 <= t < ln(2)',
              'energy_balance':'d(0.5 I^2)/dt = (-g) I - I^2',
              'rows':rows},
        'diffusion':{'exact_solution':'c1=1+exp(-2t), c2=1-exp(-2t)',
                     'observation_map':'y_i=c_i^2',
                     'scope':'The sensor map is monotone on the nonnegative concentrations used here, yet raw sums do not preserve conservation.',
                     'rows':diffusion}
    }
    out = Path(__file__).with_name('counterexamples_verified.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out),'rl_counterexample_at_t_0_1':rows[1],
                      'diffusion_initial':diffusion[0],'diffusion_at_t_1':diffusion[3]},ensure_ascii=False))


if __name__ == '__main__':
    main()
