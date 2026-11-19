#!/usr/bin/env python3
"""Closed-form RL/control checks; no external dependencies, models, or network.

This verifies arithmetic and ODE residuals for explicitly specified toy systems.
It is not an experimental test of Le Chatelier's principle, superconductivity,
or the Hegel project. The accompanying Markdown contains the analytic proofs.
"""
import argparse
import cmath
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("electromagnetic_contract_verified.json"))
    args = parser.parse_args()
    # Positive units: L=1 H, R=1 ohm, A=1 Wb-turn. theta=t/tau=t.
    # The n=2 pulse begins at zero flux with zero initial slope and zero current.
    # For t<0 its causal extension is zero (C1 at zero); it is analytic for t>=0.
    rows = []
    for t in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 10.0]:
        e = math.exp(-t)
        external = t*t*e
        external_dot = (2*t-t*t)*e
        total = t**3*e/3
        total_dot = (t*t-t**3/3)*e
        induced = total-external
        current = induced
        current_dot = total_dot-external_dot
        energy_dot = current*current_dot
        supply_power = -external_dot*current
        dissipation = current*current
        rows.append({
            "theta": t, "external_linkage": external,
            "total_linkage": total, "induced_linkage": induced,
            "current": current,
            "total_to_external": total/external if external else None,
            "induced_abs_to_external_abs": abs(induced/external) if external else None,
            "total_exceeds_external": total > external + 1e-14,
            "induced_abs_exceeds_external_abs": abs(induced) > abs(external) + 1e-14,
            "RL_ODE_residual": current_dot+current+external_dot,
            "energy_derivative": energy_dot,
            "external_supply_power": supply_power,
            "nonnegative_dissipation": dissipation,
            "energy_balance_residual": energy_dot-supply_power+dissipation,
        })
    K = 27.0
    closed_loop_poles = [-1 + K**(1/3)*cmath.exp(1j*(math.pi+2*math.pi*j)/3)
                        for j in range(3)]
    zeta = 0.2
    report = {
        "scope": __doc__,
        "RL_model": {
            "L_H": 1, "R_ohm": 1, "A_Wb_turn": 1,
            "initial_state": {"external_linkage": 0, "current": 0, "total_linkage": 0},
            "external_formula": "A*theta^2*exp(-theta)",
            "total_formula": "A*theta^3*exp(-theta)/3",
            "ratio_formula_for_theta_gt_0": "total/external = theta/3",
            "sample_rows": rows,
            "max_abs_ODE_residual": max(abs(r["RL_ODE_residual"]) for r in rows),
            "max_abs_energy_balance_residual": max(abs(r["energy_balance_residual"]) for r in rows),
            "zero_state_energy_increases_at_theta_0_5": rows[2]["energy_derivative"] > 0,
        },
        "step_limit_noncommutation": {
            "formula": "total=A*(1-exp(-R*t/L)), external=A for t>0",
            "t_to_infinity_then_R_to_zero": "A",
            "R_to_zero_then_t_to_infinity": "0",
            "note": "These two limits are analytically derived, not inferred from a finite grid."
        },
        "positive_DC_gain_unstable": {
            "negative_feedback_loop_transfer": "K/(s+1)^3", "K": K,
            "closed_loop_characteristic": "(s+1)^3+K",
            "poles": [{"real": p.real, "imag": p.imag} for p in closed_loop_poles],
            "right_half_plane_poles": sum(p.real > 0 for p in closed_loop_poles),
            "max_characteristic_residual": max(abs((p+1)**3+K) for p in closed_loop_poles),
            "stable_K_interval": "-1 < K < 8",
        },
        "passive_RLC_step_overshoot": {
            "L_H": 1, "C_F": 1, "R_ohm": 2*zeta,
            "capacitor_voltage_transfer": "1/(s^2+0.4*s+1)",
            "damping_ratio": zeta,
            "unit_step_peak": 1+math.exp(-math.pi*zeta/math.sqrt(1-zeta*zeta)),
            "peak_time": math.pi/math.sqrt(1-zeta*zeta),
            "note": "An additional example, not the single-loop RL model."
        },
        "frequency_vs_pointwise": {
            "RL_total_transfer": "1/(1+tau*s)",
            "frequency_magnitude": "1/sqrt(1+(tau*omega)^2) <= 1",
            "pointwise_bound_for_arbitrary_inputs": False,
        }
    }
    assert report["RL_model"]["max_abs_ODE_residual"] < 1e-12
    assert report["RL_model"]["max_abs_energy_balance_residual"] < 1e-12
    assert rows[6]["total_exceeds_external"]
    assert rows[8]["induced_abs_exceeds_external_abs"]
    assert report["positive_DC_gain_unstable"]["right_half_plane_poles"] == 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "ODE_residual": report["RL_model"]["max_abs_ODE_residual"],
                      "total_ratio_theta_4": rows[6]["total_to_external"],
                      "induced_abs_ratio_theta_7": rows[8]["induced_abs_to_external_abs"],
                      "unstable_poles": 2}, ensure_ascii=False))


if __name__ == "__main__":
    main()
