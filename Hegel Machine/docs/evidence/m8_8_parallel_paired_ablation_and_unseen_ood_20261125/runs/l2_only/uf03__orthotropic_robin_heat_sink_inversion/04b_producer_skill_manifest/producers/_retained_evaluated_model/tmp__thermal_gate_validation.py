import pickle
import sys
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
sys.path[:0] = ['/holdout_task', '/tmp']
from thermal_sink_solver import localize_thermal_source as baseline, _plate_operator
from thermal_wider_gate import localize_thermal_source as wider

limits = np.array([.035, .08, .008, 15.])

def check(m, gt):
    rows = []
    for solve in (baseline, wider):
        t = time.perf_counter()
        p = solve(m)
        dt = time.perf_counter() - t
        assert set(p) == {'source_xy', 'amplitude', 'junction_offset'}
        assert type(p['source_xy']) is list and len(p['source_xy']) == 2
        assert all(type(v) is float and np.isfinite(v) for v in [*p['source_xy'], p['amplitude'], p['junction_offset']])
        assert p['amplitude'] > 0 and all(0 <= v <= 1 for v in p['source_xy'])
        rows.append([np.linalg.norm(np.subtract(p['source_xy'], gt['source_xy'])), abs(p['amplitude']/gt['amplitude']-1), abs(p['junction_offset']-gt['junction_offset']), dt])
    return np.array(rows)

with open('/tmp/check_thermal.py') as f:
    text = f.read()
gen = text[text.index('def synthetic('):text.index('\nfor label, args in')]
gen = gen.replace('corruption, seed):', 'corruption, seed, amplitude, offset):').replace('a, b = 23., -.055', 'a, b = amplitude, offset')
exec(gen)

def report(label, rows):
    a = np.array(rows)
    misses = np.any(a > limits, axis=2)
    new = np.flatnonzero(misses[:, 1] & ~misses[:, 0]).tolist()
    fixed = np.flatnonzero(misses[:, 0] & ~misses[:, 1]).tolist()
    print(label, 'worsts [xy, rel_amp, offset, sec] baseline / wider:', np.round(a.max(axis=0), 7).tolist(), 'misses', misses.sum(axis=0).tolist(), 'of', len(a), 'new', new, 'fixed', fixed, flush=True)

with open('/holdout_task/public_thermal_cases.pkl', 'rb') as f:
    entries = pickle.load(f)
for idx, entry in enumerate(entries):
    m, gt = entry['measurement'], entry['gt']
    rows = [check(m, gt)]
    for i in range(len(m['T_obs'])):
        rows.append(check(dict(m, sensor_ij=np.delete(m['sensor_ij'], i, axis=0), T_obs=np.delete(m['T_obs'], i)), gt))
    report(f'public {idx} full and sensor jackknife', rows)

for label, args in [
    ('interior', (24,.075,2.8,(.67,.40),53,1.5,.06)),
    ('weakly cooled edge', (31,.09,.6,(.08,.34),48,1.3,.06)),
    ('strongly cooled corner', (35,.09,7.,(.95,.06),64,1.1,.06)),
    ('narrow source near corner', (31,.055,2.2,(.91,.09),40,.8,.06)),
]:
    report(label, [synthetic(*args, 100+i, (16.,23.,30.)[i%3], (.04,-.05)[i%2]) for i in range(16)])
