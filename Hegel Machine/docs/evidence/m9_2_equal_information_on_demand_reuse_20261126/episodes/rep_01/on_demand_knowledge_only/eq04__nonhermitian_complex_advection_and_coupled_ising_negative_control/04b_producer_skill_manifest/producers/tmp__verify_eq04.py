import importlib.util
import itertools
import pathlib
import time

import numpy as np
from scipy.linalg import solve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
solver = module.solve_nonhermitian_coupled_inversion

def dense_reference(p):
    m, nx, ny = p['basis'].shape
    A = np.zeros((nx*ny, nx*ny), dtype=np.complex128)
    hx, hy = p['Lx']/(nx+1), p['Ly']/(ny+1)
    for i, j in np.ndindex(nx, ny):
        row = i*ny+j
        vx, vy = p['vx'][i,j], p['vy'][i,j]
        A[row,row] = (2*p['nu']/hx**2 + 2*p['nu']/hy**2 + abs(vx)/hx + abs(vy)/hy + p['sigma'][i,j] + 1j*p['omega'])
        for a, b, h in ((i-1,j,hx), (i+1,j,hx), (i,j-1,hy), (i,j+1,hy)):
            if 0 <= a < nx and 0 <= b < ny:
                A[row,a*ny+b] = -p['nu']/h**2
        a = i-1 if vx >= 0 else i+1
        b = j-1 if vy >= 0 else j+1
        if 0 <= a < nx:
            A[row,a*ny+j] -= abs(vx)/hx
        if 0 <= b < ny:
            A[row,i*ny+b] -= abs(vy)/hy
    rhs = p['basis'].reshape(m, -1).T
    return A, rhs, solve(A, rhs, assume_a='gen')

def objective(g, y, p, w):
    active = np.abs(w)>1e-8
    return (np.sum(abs(g@w-y)**2)/2 + p['lam2']*np.sum(abs(w)**2)/2 + p['lam0']*sum(active)
            + sum(gamma for u,v,gamma in p['pair_penalties'] if active[u] and active[v]))

def oracle(g, y, p):
    scores = []
    for k in range(g.shape[1]+1):
        for tup in itertools.combinations(range(g.shape[1]), k):
            w = np.zeros(g.shape[1], dtype=np.complex128)
            if k:
                u, s, vh = np.linalg.svd(g[:, tup], full_matrices=False)
                w[list(tup)] = vh.conj().T @ ((u.conj().T@y) * s/(s*s+p['lam2']))
            scores.append((objective(g,y,p,w), w))
    scores.sort(key=lambda x: x[0]); return scores[0], scores[1][0]-scores[0][0]

worst = np.zeros(4)
for idx, label in enumerate(('mixed-upwind/weak-pair', 'same-data/strong-pair', 'indefinite/negative-frequency', 'duplicate-basis/underdetermined', 'null-support')):
    rng = np.random.default_rng(84 + (0 if idx==1 else idx))
    shape = (32,32)
    basis = rng.normal(size=(11,*shape))
    if idx==3: basis[-1]=basis[-2]
    p = dict(basis=basis, Lx=0.67, Ly=2.3, nu=0.002 if idx==2 else 0.012,
             vx=rng.uniform(-3,3,shape), vy=rng.uniform(-1,1,shape),
             omega=-0.41 if idx==2 else 7.1,
             sigma=rng.uniform(-140,10,shape) if idx==2 else rng.uniform(0,5,shape),
             lam0=0.04, lam2=0.007 if idx!=3 else 1e-6)
    A, rhs, h = dense_reference(p)
    # Normalize the response scale so penalties, data fit and ridge all matter.
    scale = 0.4 / np.median(np.abs(h))
    rhs = rhs * scale
    p['basis'] = basis * scale
    h = h * scale
    ids = np.unique(np.r_[0,31,992,1023,rng.choice(1024,size=4 if idx==3 else 37,replace=False)])
    p['sensor_ij'] = np.column_stack(np.divmod(ids,32))
    g = h[ids]
    truth = np.zeros(11,dtype=complex); truth[[2,6,8]] = [1+0.7j, -0.5+0.6j, 0.2+0.1j]
    y = g@truth + 0.03*(rng.normal(size=len(ids))+1j*rng.normal(size=len(ids)))
    if idx==4: y *= 0
    p.update(y_obs_re=y.real,y_obs_im=y.imag,
             pair_penalties=[(2,6,5000 if idx==1 else 0.03), (8,2,0.012), (6,8,0.032), (0,10,0.12)])
    (ref, w_ref), gap = oracle(g,y,p)
    raw = module._basis_responses(p).reshape(1024,11)
    residual = np.linalg.norm(A@raw-rhs)/np.linalg.norm(rhs)
    t = time.perf_counter(); got = solver(p); elapsed = time.perf_counter()-t
    assert set(got)=={'weights_real','weights_imag','objective','predicted_field_abs'}
    for key, dims in [('weights_real',(11,)),('weights_imag',(11,)),('predicted_field_abs',shape)]:
        assert got[key].shape == dims and got[key].dtype == np.float64
        assert np.isfinite(got[key]).all()
    assert isinstance(got['objective'],float)
    w = got['weights_real'] + 1j*got['weights_imag']
    active = np.abs(w)>1e-8
    if idx==0: assert active[2] and active[6]
    if idx==1: assert not (active[2] and active[6])
    if idx==4: assert not active.any()
    field = abs(h@w_ref).reshape(shape)
    field_error = np.linalg.norm(field-got['predicted_field_abs'])/max(np.linalg.norm(field),1e-30)
    errors = [residual, abs(ref-got['objective']), field_error, abs(objective(g,y,p,w)-got['objective'])]
    print(f'{label}: residual={errors[0]:.3g} objective_error={errors[1]:.3g} field_rel={errors[2]:.3g} objective_consistency={errors[3]:.3g} runner_up_gap={gap:.3g} active={np.where(active)[0]} solver_s={elapsed:.3f}',flush=True)
    assert np.all(np.array(errors) <= [1e-10,1e-6,1e-5,1e-6])
    worst = np.maximum(worst,errors)
print('WORST residual, objective abs, field rel, objective consistency:',worst)
validator=pathlib.Path('/holdout_task/validate_public_schema.py')
cases=pathlib.Path('/holdout_task/public_eq04_cases.pkl')
print('Public validator present:',validator.is_file(),'public fixtures present:',cases.is_file())
