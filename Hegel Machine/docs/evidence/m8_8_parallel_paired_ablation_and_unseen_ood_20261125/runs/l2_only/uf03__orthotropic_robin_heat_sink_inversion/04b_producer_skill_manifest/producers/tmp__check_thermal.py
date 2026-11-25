import pickle, sys, time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/holdout_task')
from thermal_sink_solver import localize_thermal_source, _plate_operator

LIMIT = np.array([.035, .08, .008, 15])
def check(m, gt):
    start = time.perf_counter()
    fit = localize_thermal_source(m)
    elapsed = time.perf_counter()-start
    assert set(fit) == {'source_xy', 'amplitude', 'junction_offset'}
    assert isinstance(fit['source_xy'], list) and len(fit['source_xy']) == 2
    assert all(type(x) is float and np.isfinite(x) for x in [*fit['source_xy'], fit['amplitude'], fit['junction_offset']])
    assert fit['amplitude'] > 0 and all(0 <= x <= 1 for x in fit['source_xy'])
    return np.array([np.linalg.norm(np.subtract(fit['source_xy'], gt['source_xy'])), abs(fit['amplitude']/gt['amplitude']-1), abs(fit['junction_offset']-gt['junction_offset']), elapsed])

def report(label, rows):
    a = np.array(rows)
    print(label, 'worst xyz/rel_amp/offset/sec', a.max(axis=0), 'failed', np.flatnonzero(np.any(a > LIMIT, axis=1)).tolist(), flush=True)

for idx, ent in enumerate(pickle.load(open('/holdout_task/public_thermal_cases.pkl', 'rb'))):
    m, gt = ent['measurement'], ent['gt']
    rows=[check(m, gt)]
    for i in range(len(m['T_obs'])):
        sub = dict(m, sensor_ij=np.delete(m['sensor_ij'], i, axis=0), T_obs=np.delete(m['T_obs'], i))
        rows.append(check(sub, gt))
    report(f'public {idx} full plus leave-one-thermistor-out', rows)

# Independently assemble each row and use a direct forward solve, not the adjoint bank.
def synthetic(n, sigma, gamma, xy, n_sensor, contrast, corruption, seed):
    rng = np.random.default_rng(seed)
    grid = (np.arange(n)+.5)/n
    a, b = 23., -.055
    k = np.exp(contrast*(.45*np.sin(6*grid[:,None]+1)*np.cos(5*grid[None,:])+.55*rng.uniform(-1,1,(n,n))))
    A=lil_matrix((n*n,n*n))
    for i in range(n):
        for j in range(n):
            z=i*n+j
            for u,v in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if 0 <= u < n and 0 <= v < n:
                    hface=2/(1/k[i,j]+1/k[u,v]); A[z,u*n+v]=-hface; A[z,z]+=hface
                else:
                    A[z,z]+=gamma/n
    A=A.tocsc()
    delta=A-_plate_operator(k,gamma)
    assert not delta.nnz or np.max(np.abs(delta.data)) < 1e-12*np.max(abs(A.data))
    q=a*np.exp(-.5*((grid[:,None]-xy[0])**2+(grid[None,:]-xy[1])**2)/sigma**2).ravel()/n**2
    field=spsolve(A,q)
    assert abs((A @ field).sum()-q.sum()) < 1e-11
    ids=rng.choice(n*n,n_sensor,replace=False)
    sensor=np.column_stack(np.divmod(ids,n))
    y=field[ids]+b+rng.normal(0,.001,n_sensor)
    stuck=rng.choice(n_sensor,round(corruption*n_sensor),replace=False)
    y[stuck]+=rng.choice([-1.,1.],len(stuck))*rng.uniform(.18,.26,len(stuck))
    return check(dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=sensor,T_obs=y),dict(source_xy=xy,amplitude=a,junction_offset=b))

for label, args in [('heterogeneous interior',(22,.085,2.2,(.48,.65),52,1.8,.06)),('weak Robin near edge',(29,.085,.45,(.043,.72),48,1.1,.06)),('strong Robin near corner',(35,.09,7.,(.96,.045),64,1.1,.08)),('narrow source sparse sensors',(31,.047,2.2,(.31,.35),36,.8,.06)),('large plate',(100,.07,2.2,(.57,.38),120,.8,.06))]:
    report(label, [synthetic(*args, seed) for seed in range(6)])
