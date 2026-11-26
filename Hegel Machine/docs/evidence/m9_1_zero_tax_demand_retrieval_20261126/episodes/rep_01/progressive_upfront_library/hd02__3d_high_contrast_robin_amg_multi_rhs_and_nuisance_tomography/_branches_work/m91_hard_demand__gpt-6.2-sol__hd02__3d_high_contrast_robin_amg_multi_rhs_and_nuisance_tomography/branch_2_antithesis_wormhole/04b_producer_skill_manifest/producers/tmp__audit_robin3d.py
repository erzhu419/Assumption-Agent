import importlib.util
import pickle
import time
import numpy as np

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin3d_amg_tomography.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    p = pickle.load(f)[0]['problem']

# Constant fields isolate reaction and all six boundary faces: interior flux vanishes.
def leakage(p):
    k = p['kappa']
    h = np.array([p['Lx'], p['Ly'], p['Lz']]) / k.shape
    out = p['sigma'].copy()
    for axis, step in enumerate(h):
        a, b = np.moveaxis(out, axis, 0), np.moveaxis(k, axis, 0)
        for side in (0, -1):
            a[side] += 1 / (step * (1/p['gamma_robin'] + step/(2*b[side])))
    return out

A, vol = m._assemble(p)
assert (A - A.T).nnz == 0
leak_error = np.max(np.abs((A @ np.ones(A.shape[0])) / vol - leakage(p).ravel()))
assert leak_error < 1e-8, leak_error
print('Public stencil symmetry exact; max constant-field balance error', leak_error)

# Fit each model on seven sensor folds; measure prediction on the excluded fold.
r = m.solve_3d_robin_tomography(p)
S = r['basis_fields'][(slice(None),) + tuple(p['sensor_ijk'].T)].T
Z,Y = p['nuisance_design'],p['Y_obs']
lz,ls = p['nuisance_ridge'],p['source_ridge']
audit = []
for hold in np.array_split(np.random.default_rng(839).permutation(len(S)), 8):
    train = np.ones(len(S), dtype=bool)
    train[hold] = False
    D = np.column_stack((Z,S))
    R = np.diag(np.sqrt([lz]*Z.shape[1] + [ls]*S.shape[1]))
    coef = np.linalg.lstsq(np.vstack((D[train],R)), np.pad(Y[train],((0,len(R)),(0,0))), rcond=None)[0]
    _, w = m._profile(S[train],Z[train],Y[train],lz,ls)
    np.testing.assert_allclose(w, coef[Z.shape[1]:], rtol=1e-9, atol=1e-10)
    raw = np.linalg.lstsq(np.vstack((S[train],np.sqrt(ls)*np.eye(S.shape[1]))), np.pad(Y[train],((0,S.shape[1]),(0,0))), rcond=None)[0]
    audit.append((np.sqrt(np.mean((D[hold]@coef-Y[hold])**2)), np.sqrt(np.mean((S[hold]@raw-Y[hold])**2))))
print('Public worst held-out sensor-fold RMSE (profiled, source-only)', np.max(audit,axis=0))

# Distinct full-scale contrast endpoints: isolated high cells and connected high slabs.
# A manufactured field tests flux, corners and anisotropic grid-unit conventions.
x,y,z = np.indices((28,28,28))
target = np.array([np.sin((i+1)*.12 + .08*x + .07*y - .09*z) + .4*np.cos(.11*x*z/(i+1)) for i in range(8)])
total=0.
for label, k in [('checkerboard',np.where((x+y+z)%2,80.,.02)), ('slabs',np.where(x%3,80.,.02))]:
    q = dict(p,kappa=k,sigma=np.full(k.shape,.053),Lx=.85,Ly=1.31,Lz=1.08,gamma_robin=.28)
    h = np.array([q['Lx'],q['Ly'],q['Lz']])/k.shape
    src = target*leakage(q)
    for axis,step in enumerate(h):
        K = np.moveaxis(k,axis,0)
        u = np.moveaxis(target,axis+1,1)
        op = np.moveaxis(src,axis+1,1)
        flux = np.diff(u,axis=1) / (.5*step**2*(1/K[:-1]+1/K[1:]))
        op[:,1:] += flux
        op[:,:-1] -= flux
    q['candidate_sources'] = src
    t = time.perf_counter()
    got = m.solve_3d_robin_tomography(q)['basis_fields']
    elapsed = time.perf_counter()-t
    total += elapsed
    err = np.max(np.abs(got-target))
    A,vol=m._assemble(q)
    b = src.reshape(8,-1).T
    rel = np.max(np.linalg.norm(A@got.reshape(8,-1).T/vol-b,axis=0)/np.linalg.norm(b,axis=0))
    assert err < 1e-7 and rel <= 1e-7, (label,err,rel)
    print(label, 'seconds', elapsed, 'max field error', err, 'worst PDE residual',rel)
assert total < 14, total
print('Full-scale added stress pair seconds',total)
