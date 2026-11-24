import itertools
from plant_sim import NOMINAL_SCENARIOS, run_closed_loop
from bioreactor_ctrl import compute_control_step


def report(label, cases, seeds):
    rows = [(run_closed_loop(compute_control_step, s, k), s, k) for s, k in itertools.product(cases, seeds)]
    print(label, len(rows), 'unsafe', sum(not a[0]['safe'] for a in rows))
    for key, extreme in [('final_biomass_g', min), ('max_T_C', max), ('max_V_L', max)]:
        r, s, seed = extreme(rows, key=lambda a: a[0][key]); print(key, round(r[key], 6), s, 'seed', seed)


if __name__ == '__main__':
    report('provided', NOMINAL_SCENARIOS, [0, 10, 42])
    cases = [dict(zip(('mu0','Ks','Ki','ua_scale','heat_pulse'), v)) for v in itertools.product(
        [0.357, 0.483], [0.8, 0.9], [12.0, 18.0], [0.75, 1.0], [0, 2.2, 3.0])]
    report('combined corners', cases, [0, 10, 42])
