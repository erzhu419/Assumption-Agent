import importlib.util
import pickle
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('thermal_sink_solver', path+'thermal_sink_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(path+'public_thermal_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
rng = np.random.default_rng(1703)
limits = np.array([.035, .08, .008, 15.])
results = {}

def check(group, label, m, g):
    t = time.monotonic()
    p = solver.localize_thermal_source(m)
    elapsed = time.monotonic()-t
    assert set(p) == {'source_xy', 'amplitude', 'junction_offset'}
    assert isinstance(p['source_xy'], list) and len(p['source_xy']) == 2
    assert all(type(v) is float and np.isfinite(v) for v in [*p['source_xy'], p['amplitude'], p['junction_offset']])
    assert all(0 <= v <= 1 for v in p['source_xy']) and p['amplitude'] > 0
    e = np.array([np.linalg.norm(np.subtract(p['source_xy'], g['source_xy'])), abs(p['amplitude']/g['amplitude']-1), abs(p['junction_offset']-g['junction_offset']), elapsed])
    results.setdefault(group, []).append(e)
    if not np.all(e <= limits):
        print('FAIL', group, label, e, p, g, flush=True)

for trial in range(120):
    n = (18, 22, 30)[trial % 3]
    gamma = (0.45, 2.2, 5.0)[(trial//3) % 3]
    sigma = (.06, .085, .12)[(trial//9) % 3]
    x, y = (np.indices((n, n))+.5)/n
    k = np.exp(.55*np.sin(6*x+trial)*np.cos(5*y)+.35*(x-y))
    xy = rng.uniform(.08, .92, 2)
    if trial % 2 == 0:
        xy[trial//4 % 2] = rng.uniform(0., .12) if trial % 8 < 4 else rng.uniform(.88, 1.)
    a = rng.uniform(14, 32)
    b = rng.uniform(-.075, .075)
    q = a*np.exp(-((x-xy[0])**2+(y-xy[1])**2)/(2*sigma**2))/n**2
    A = lil_matrix((n*n, n*n))
    for i,j in np.ndindex(n,n):
        row = i*n+j
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii,jj = i+di,j+dj
            if 0 <= ii < n and 0 <= jj < n:
                w = 2/(1/k[i,j]+1/k[ii,jj])
                A[row,row] += w
                A[row,ii*n+jj] = -w
            else:
                A[row,row] += gamma/n
    u = spsolve(A.tocsc(),q.ravel())
    ids = rng.choice(n*n,52,replace=False)
    yy = u[ids]+b+rng.normal(0,.0015,len(ids))
    outliers = rng.choice(len(ids),3,replace=False) if trial%2 else np.argsort(u[ids])[-3:]
    yy[outliers] += rng.choice([-1,1],3)*rng.uniform(.12,.45,3)
    check('synthetic edge' if trial%2 == 0 else 'synthetic interior', (trial,n,gamma,sigma), dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack(divmod(ids,n)),T_obs=yy), dict(source_xy=xy,amplitude=a,junction_offset=b))

failed = 0
for group, es in results.items():
    es = np.asarray(es)
    bad = np.count_nonzero(np.any(es > limits,axis=1))
    failed += bad
    print(group, 'n=',len(es),'failed=',bad,'worst (position, relative amplitude, offset, seconds):',np.max(es,axis=0))
raise SystemExit(bool(failed))
