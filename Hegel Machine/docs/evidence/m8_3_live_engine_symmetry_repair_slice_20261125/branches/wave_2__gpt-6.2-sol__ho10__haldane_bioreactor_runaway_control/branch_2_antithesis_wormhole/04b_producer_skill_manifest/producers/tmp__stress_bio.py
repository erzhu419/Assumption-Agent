from itertools import product
from plant_sim import NOMINAL_SCENARIOS, run_closed_loop
from bioreactor_ctrl import compute_control_step

corners = [dict(name=str(p), mu0=p[0], Ks=p[1], Ki=p[2], ua_scale=p[3], heat_pulse=p[4])
           for p in product((0.357, 0.483), (0.8, 0.9), (12., 18.), (0.75, 1.), (0., 2.2, 5.))]
results = [(scen['name'], seed, run_closed_loop(compute_control_step, scen, seed))
           for scen in NOMINAL_SCENARIOS + corners for seed in (0, 10, 43)]
for name, seed, result in results[:9]:
    print(name, seed, result)
for metric, select in [('final_biomass_g', min), ('max_T_C', max), ('max_V_L', max)]:
    print('WORST', metric, select(results, key=lambda row: row[2][metric]))
print('violations', sum(not r['safe'] or r['final_biomass_g'] < 62.0 for _,_,r in results), '/', len(results))
