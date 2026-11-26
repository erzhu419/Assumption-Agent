import importlib.util
import time
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

path = '/holdout_task/robin_continuation_tomography.py'
spec = importlib.util.spec_from_file_location('robin', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def independent_matrix(s, theta):
    shape = s['kappa_base'].shape
    h = np.array([s['Lx'], s['Ly'], s['Lz']]) / shape
    volume = np.prod(h)
    k = (s['kappa_base'] + theta * (s['kappa_mod1'] + theta * s['kappa_mod2'])).ravel()
    d = volume * (s['sigma_base'] + theta * s['sigma_mod']).ravel()
    gamma = s['gamma_robin']
    n = k.size
    i = np.arange(n)
    xyz = np.array(np.unravel_index(i, shape))
    offsets = [0]
    offdiag = []
    for a, step in enumerate((shape[1]*shape[2],shape[2],1)):
        valid = i[xyz[a] < shape[a]-1]
        c = np.zeros(n-step)
        c[valid] = (volume / h[a]**2) / (.5/k[valid] + .5/k[valid+step])
        d[valid] += c[valid]
        d[valid+step] += c[valid]
        wall = (xyz[a] == 0) | (xyz[a] == shape[a]-1)
        d[wall] += (volume / h[a]) / (1/gamma + h[a]*.5/k[wall])
        offsets.extend((-step, step))
        offdiag.extend((-c,-c))
    return sparse.diags([d] + offdiag, offsets, format='csc'), volume

def make(i):
    rng = np.random.default_rng(8829+i)
    shape = (28,)*3
    xyz = np.indices(shape)
    if i == 0:
        base = np.where((xyz[0] < 10) | ((xyz[2] > 16) & (xyz[1] > 14)), 3500., 1.)
        q = np.where(xyz[0] % 4 < 2, 0.03, 8000.)
        gamma, sigma = 0.3, .08
    else:
        base = np.exp(rng.uniform(0,np.log(3500),shape))
        q = np.where((xyz[0] + xyz[1]) % 6 < 3, 1e4, .001)
        gamma, sigma = 15., .002
    theta = np.linspace(.15,1.85,9)[[8,0,7,1,6,2,5,3,4]]
    sources = rng.normal(size=(3,)+shape)*.1 + np.array([.8,-.2,0.])[:,None,None,None]
    sources[2] += (xyz[0]-xyz[2])/28
    sensors = np.concatenate((rng.integers(28,size=(96,3)), [[0,0,0],[27,0,27],[0,27,27],[27,27,27],[0,0,0]]))
    n = len(theta)*len(sensors)
    z = rng.normal(size=(n,7)); z[:,0] = 1; z[:,6] = z[:,1]+1e-8*z[:,2]
    return dict(Lx=.73,Ly=1.45,Lz=2.2,kappa_base=base,kappa_mod1=2+.2*base,kappa_mod2=q,sigma_base=np.full(shape,sigma),sigma_mod=np.full(shape,sigma*.1),gamma_robin=gamma,thetas=theta,source_modes=sources,sensor_ijk=sensors,Z_nuisance=z,Y_obs=rng.normal(size=(n,6))+z@rng.normal(size=(7,6)),nuisance_ridge=0.005,source_ridge=0.00015)

original_cg = mod.cg
def rel(x,y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)
total = 0
for i in range(2):
    s = make(i)
    iterates = []
    def record(*args, **kw):
        x, status = original_cg(*args, **kw)
        iterates.append(x.copy())
        return x, status
    mod.cg = record
    t = time.perf_counter()
    out = mod.solve_robin_continuation_tomography(s)
    elapsed = time.perf_counter()-t
    total += elapsed
    assert len(iterates) == 27, 'Inspect any correction solve before comparing per-mode iterates'
    assert set(out) == {'terminal_fields','continuation_sensor_predictions','profiled_amplitudes','joint_source_weights'}
    n = len(s['sensor_ijk'])
    assert [out[k].shape for k in ('terminal_fields','continuation_sensor_predictions','profiled_amplitudes','joint_source_weights')] == [(3,28,28,28),(9*n,3),(3,6),(3,6)]
    worst_resid = 0.
    expected = []
    with threadpool_limits(limits=1,user_api='blas'):
        for j, theta in enumerate(s['thetas']):
            a,v = independent_matrix(s,theta)
            b = v*s['source_modes'].reshape(3,-1).T
            x = np.array(iterates[3*j:3*(j+1)]).T
            residuals = np.linalg.norm(a@x-b,axis=0)/np.linalg.norm(b,axis=0)
            worst_resid = max(worst_resid,max(residuals))
            true_x = splu(a,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,options={'SymmetricMode': True}).solve(b)
            ids = np.ravel_multi_index(s['sensor_ijk'].T,(28,)*3)
            expected.append(true_x[ids,:])
        sf = np.concatenate(expected)
        z,y = s['Z_nuisance'],s['Y_obs']
        p = np.eye(len(z)) - z@linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T,assume_a='pos')
        expected_a = (sf.T@p@y)/np.diag(sf.T@p@sf)[:,None]
        design = np.column_stack((z,sf))
        penalty = np.sqrt(np.repeat([s['nuisance_ridge'],s['source_ridge']],[z.shape[1],sf.shape[1]]))
        expected_w = linalg.lstsq(np.vstack((design,np.diag(penalty))),np.vstack((y,np.zeros((len(penalty),y.shape[1])))))[0][-3:]
    errors = [rel(out['terminal_fields'].reshape(3,-1).T,true_x), rel(out['continuation_sensor_predictions'],sf), rel(out['profiled_amplitudes'],expected_a), rel(out['joint_source_weights'],expected_w)]
    print(f'study {i+1}: {elapsed:.3f}s, worst residual {worst_resid:.3g}, terminal/sensors/amplitudes/weights errors {errors}',flush=True)
    assert worst_resid <= 1e-7
    assert all(e < bound for e,bound in zip(errors,[1e-7,1e-7,1e-5,.03]))
print(f'two-study total {total:.3f}s')
assert total <= 25
