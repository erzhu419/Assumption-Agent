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
rng = np.random.default_rng(84)
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

for trial in range(36):
    n = (18, 22, 30)[trial % 3]
    gamma = (0.45, 2.2, 5.0)[(trial//3) % 3]
    sigma = (.06, .085, .12)[(trial//9) % 3]
    x, y = (np.indices((n, n))+.5)/n
    k = np.exp(.55*np.sin(6*x+trial)*np.cos(5*y)+.35*(x-y))
    xy = rng.uniform(.08, .92, 2)
    if trial % 4 == 0:
        xy[trial//4 % 2] = .035 if trial % 8 == 0 else .965
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
    if trial % 4 != 0:
        continue
    from scipy.optimize import least_squares
    yy_clean = yy.copy()
    bad = np.isin(np.arange(len(ids)), outliers)
    yy_clean[bad] = u[ids[bad]] + b
    m = dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack(divmod(ids,n)),T_obs=yy)
    rhs = np.zeros((n*n,len(ids)))
    rhs[ids,np.arange(len(ids))] = 1/n**2
    green = solver._factor_operator(k,gamma).solve(rhs).T
    truth = np.r_[xy,a,b]
    def eval(p):
        dx,dy=x-p[0],y-p[1]
        g=np.exp(-.5*(dx**2+dy**2)/sigma**2)
        design=green@np.stack([p[2]*g*dx/sigma**2,p[2]*g*dy/sigma**2,g],axis=-1).reshape(-1,3)
        return p[2]*design[:,2]+p[3],np.column_stack((design,np.ones(len(ids))))
    pred,J=eval(truth)
    fit=least_squares(lambda p:(eval(p)[0]-yy)[~bad],truth,jac=lambda p:eval(p)[1][~bad],bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf]))
    sd=.0015*np.sqrt(np.diag(np.linalg.inv(J[~bad].T@J[~bad])))
    got=solver.localize_thermal_source(m)
    p=np.r_[got['source_xy'],got['amplitude'],got['junction_offset']]
    r=eval(p)[0]-yy
    print(trial, 'relative amplitude err robust/oracle/1sigma:', np.round([(p[2]-a)/a,(fit.x[2]-a)/a,sd[2]/a],5),'edge truth/oracle xy:',np.round(xy,5),np.round(fit.x[:2],5),'outlier residuals:',np.round(r[bad],5),'oracle inlier residual RMS:',np.sqrt(np.mean(fit.fun**2)))
