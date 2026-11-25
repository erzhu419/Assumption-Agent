import pickle
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from thermal_sink_solver import localize_thermal_source

LIMIT = np.array([.035, .08, .008])

def errors(p, g):
    assert set(p) == {'source_xy', 'amplitude', 'junction_offset'}
    assert len(p['source_xy']) == 2 and all(type(v) is float for v in [*p['source_xy'], p['amplitude'], p['junction_offset']])
    assert p['amplitude'] > 0
    return np.array([np.linalg.norm(np.subtract(p['source_xy'], g['source_xy'])), abs(p['amplitude']/g['amplitude']-1), abs(p['junction_offset']-g['junction_offset'])])

def run(m, g):
    t = time.perf_counter()
    p = localize_thermal_source(m)
    return errors(p, g), time.perf_counter()-t

def report(name, results):
    err = np.array([v[0] for _, v in results]); dt = np.array([v[1] for _, v in results]); worst = np.argmax(err, axis=0)
    print(name, 'n', len(err), 'max error', err.max(axis=0), 'worst ids', [results[i][0] for i in worst], 'misses', (err>LIMIT).sum(axis=0), 'max seconds', round(dt.max(),3), flush=True)
    for i in np.flatnonzero(np.any(err>LIMIT, axis=1))[:12]:
        print(' miss', results[i][0], err[i], flush=True)

cases = pickle.load(open('/holdout_task/public_thermal_cases.pkl','rb'))
report('public', [(i, run(c['measurement'], c['gt'])) for i,c in enumerate(cases)])
for i, case in enumerate(cases):
    m, g = case['measurement'], case['gt']
    out = []
    for j in range(len(m['T_obs'])):
        d = dict(m, sensor_ij=np.delete(m['sensor_ij'],j,axis=0), T_obs=np.delete(m['T_obs'],j))
        out.append((j,run(d,g)))
    report('leave-one-sensor-out '+str(i),out)

def truth_grid(n,k,gamma,sigma,center,amp):
    A = lil_matrix((n*n,n*n)); q = np.empty(n*n)
    for i in range(n):
        for j in range(n):
            p = i*n+j
            q[p] = amp/n**2*np.exp(-(((i+.5)/n-center[0])**2+((j+.5)/n-center[1])**2)/(2*sigma**2))
            for a,b in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                if not (0<=a<n and 0<=b<n):
                    A[p,p] += gamma/n
                else:
                    w = 2/(1/k[i,j]+1/k[a,b]); A[p,p] += w; A[p,a*n+b] = -w
    u=spsolve(A.tocsc(),q).reshape(n,n)
    edge_heat=(u[0,:].sum()+u[-1,:].sum()+u[:,0].sum()+u[:,-1].sum())*gamma/n
    assert abs(edge_heat/q.sum()-1)<1e-9
    return u

for name,n,m,gamma,sigma,spread in [('public_geometry',22,52,2.2,.085,.6),('narrow_sparse',24,34,2.2,.058,.8),('strong_robin',32,48,5.,.070,1.2),('weak_robin',28,52,.65,.100,.8),('larger_grid',64,64,2.2,.042,1.0)]:
    results=[]
    for seed in range(24 if n<50 else 6):
        rng=np.random.default_rng(4200+seed)
        x=(np.arange(n)+.5)/n
        k=np.exp(spread*(np.sin(2*np.pi*x[:,None]+1.4*seed)*np.cos(2*np.pi*x[None,:])+0.4*np.cos(4*np.pi*x[:,None]+3*x[None,:])))
        xy = np.array([[.04,.05],[.03,.57],[.95,.88],[.88,.04],[.49,.49],[.31,.71]])[seed%6] + rng.uniform(-.013,.013,2)
        amp = rng.uniform(15.,29.); offset = rng.uniform(-.07,.07)
        u=truth_grid(n,k,gamma,sigma,xy,amp)
        i,j=np.unravel_index(rng.choice(n*n,m,replace=False),(n,n)); clean=u[i,j]+offset
        obs=clean+rng.normal(0,.0013,m)
        bad=rng.choice(m,round(.06*m),replace=False) if seed%2==0 else np.argsort(np.linalg.norm((np.column_stack((i,j))+.5)/n-xy,axis=1))[:round(.06*m)]
        obs[bad]+=rng.choice([-1.,1.],len(bad))*rng.uniform(.16,.26,len(bad))
        results.append((seed,run(dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack((i,j)),T_obs=obs),dict(source_xy=xy,amplitude=amp,junction_offset=offset))))
    report(name,results)
