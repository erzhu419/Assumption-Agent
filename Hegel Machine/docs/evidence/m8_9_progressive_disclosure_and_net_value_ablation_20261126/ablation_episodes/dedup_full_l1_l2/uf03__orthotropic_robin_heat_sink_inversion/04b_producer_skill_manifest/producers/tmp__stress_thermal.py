import importlib.util
import pickle
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

spec = importlib.util.spec_from_file_location('thermal', '/holdout_task/thermal_sink_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_thermal_cases.pkl','rb') as f:
    cases = pickle.load(f)

limits = np.array([.035, .08, .008, 15.])
results = defaultdict(list)

def check(group, label, m, gt):
    start = time.perf_counter()
    ans = mod.localize_thermal_source(m)
    elapsed = time.perf_counter() - start
    assert set(ans) == {'source_xy', 'amplitude', 'junction_offset'}
    assert type(ans['source_xy']) is list and len(ans['source_xy']) == 2
    assert all(type(x) is float for x in [*ans['source_xy'], ans['amplitude'], ans['junction_offset']])
    assert ans['amplitude'] > 0 and all(0 <= x <= 1 for x in ans['source_xy'])
    errors = np.array([np.linalg.norm(np.array(ans['source_xy'])-gt['source_xy']), abs(ans['amplitude']/gt['amplitude']-1), abs(ans['junction_offset']-gt['junction_offset']), elapsed])
    assert np.isfinite(errors).all()
    results[group].append((label, errors))
    if np.any(errors > limits):
        print('OUTSIDE', group, label, errors, ans, flush=True)

for i, c in enumerate(cases):
    m = c['measurement']
    check('public', i, m, c['gt'])
    for j in range(len(m['T_obs'])):
        keep = np.arange(len(m['T_obs'])) != j
        check('public leave-one-sensor-out', (i,j), {**m,'sensor_ij':m['sensor_ij'][keep], 'T_obs':m['T_obs'][keep]}, c['gt'])

# Generate independent data with a four-neighbor flux loop rather than solver internals.
def synthetic(n, gamma, sigma, location, seed, contrast, mcount):
    rng = np.random.default_rng(seed)
    c = (np.arange(n)+.5)/n
    x,y = c[:,None],c[None,:]
    k = np.exp(contrast*(.55*np.sin(4*x+6*y)+.35*np.cos(9*x-3*y)+.1*rng.normal(size=(n,n))))
    matrix = lil_matrix((n*n,n*n))
    for i,j in np.ndindex(n,n):
        row = i*n+j
        for a,b in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
            if not (0 <= a < n and 0 <= b < n):
                matrix[row,row] += gamma/n
            else:
                w = 2/(1/k[i,j]+1/k[a,b])
                matrix[row,row] += w
                matrix[row,a*n+b] = -w
    amp, base = rng.uniform(16,31), rng.uniform(-.08,.08)
    load = amp/n**2*np.exp(-.5*((x-location[0])**2+(y-location[1])**2)/sigma**2)
    u = spsolve(matrix.tocsc(),load.ravel())
    ix = rng.choice(n*n,mcount,replace=False)
    sensors = np.column_stack(np.divmod(ix,n))
    obs = u[ix] + base + rng.normal(0,.00125,mcount)
    corrupt = rng.choice(mcount,round(.06*mcount),replace=False)
    obs[corrupt] += rng.choice([-1.,1.],len(corrupt))*rng.uniform(.18,.27,len(corrupt))
    return dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=sensors,T_obs=obs),dict(source_xy=location,amplitude=amp,junction_offset=base)

for group,n,gamma,sigma,xy,contrast,mcount in [
    ('edge',22,2.2,.085,(.018,.60),.7,52),
    ('corner',22,2.2,.085,(.03,.96),.7,52),
    ('narrow',30,2.2,.052,(.19,.85),1.1,68),
    ('weak-Robin',22,.60,.085,(.42,.72),1.,52),
    ('strong-Robin',26,5.,.10,(.74,.32),1.6,52),
    ('heterogeneous',38,2.2,.085,(.65,.68),2.5,72),
]:
    for seed in range(8):
        check(group,seed,*synthetic(n,gamma,sigma,xy,601+seed,contrast,mcount))

failed = False
for group, items in results.items():
    values = np.array([err for _,err in items])
    indices = values.argmax(axis=0)
    print(group, 'n', len(items), 'max (location, rel amp, offset, seconds)', values.max(axis=0), 'worst labels', [items[j][0] for j in indices])
    failed |= bool(np.any(values > limits))
raise SystemExit(int(failed))
