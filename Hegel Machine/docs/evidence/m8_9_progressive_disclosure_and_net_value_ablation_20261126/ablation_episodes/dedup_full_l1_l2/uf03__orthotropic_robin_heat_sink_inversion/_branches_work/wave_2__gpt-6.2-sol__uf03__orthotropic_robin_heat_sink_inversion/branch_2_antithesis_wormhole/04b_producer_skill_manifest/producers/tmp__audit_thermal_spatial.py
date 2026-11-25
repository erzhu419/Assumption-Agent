import runpy
import time
import numpy as np

ns = runpy.run_path('/tmp/check_thermal.py')
solve = ns['mod'].localize_thermal_source
limits = np.array([.035, .08, .008, 15.])
rows = []
for i, case in enumerate(ns['cases']):
    m, gt = case['measurement'], case['gt']
    xy = (m['sensor_ij'] + .5) / m['N']
    quadrant = 2 * (xy[:, 0] >= .5) + (xy[:, 1] >= .5)
    for q in range(4):
        keep = quadrant != q
        start = time.perf_counter()
        fit = solve({**m, 'sensor_ij': m['sensor_ij'][keep], 'T_obs': m['T_obs'][keep]})
        elapsed = time.perf_counter() - start
        err = np.array([np.linalg.norm(np.asarray(fit['source_xy']) - gt['source_xy']), abs(fit['amplitude']/gt['amplitude']-1), abs(fit['junction_offset']-gt['junction_offset']), elapsed])
        rows.append(err)
        print(f'public {i}, omitted quadrant {q}, retained {sum(keep)}: {err}')
    base = solve(m)
    for scale in (.1, 10.):
        changed = solve({**m, 'T_obs': scale * m['T_obs']})
        np.testing.assert_allclose(changed['source_xy'], base['source_xy'], rtol=0, atol=1e-6)
        np.testing.assert_allclose([changed['amplitude']/scale, changed['junction_offset']/scale], [base['amplitude'], base['junction_offset']], rtol=1e-6, atol=1e-7)
worst = np.max(rows, axis=0)
print('Worst public spatial holdout: location, relative amplitude, offset, seconds:', worst)
assert np.all(worst <= limits)
