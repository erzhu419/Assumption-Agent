import importlib.util
import itertools
import pathlib
import time

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def forward(p):
    m, nx, ny = p['basis'].shape
    hx, hy = p['Lx']/(nx+1), p['Ly']/(ny+1)
    dx, dy = p['nu']/hx**2, p['nu']/hy**2
    a = lil_matrix((nx*ny, nx*ny), dtype=np.complex128)
    for i, j in itertools.product(range(nx), range(ny)):
        k = i*ny+j
        vx, vy = p['vx'][i,j], p['vy'][i,j]
        a[k,k] = 2*dx + 2*dy + abs(vx)/hx + abs(vy)/hy + p['sigma'][i,j] + 1j*p['omega']
        for ii, jj, h, diff, vel in ((i-1,j,hx,dx,vx), (i+1,j,hx,dx,-vx), (i,j-1,hy,dy,vy), (i,j+1,hy,dy,-vy)):
            if 0 <= ii < nx and 0 <= jj < ny:
                a[k,ii*ny+jj] = -diff-max(vel,0)/h
    a = a.tocsc()
    b = p['basis'].reshape(m,-1).T
    f = spsolve(a,b).reshape(nx,ny,m)
    f_impl = mod._responses(p)
    op_err = np.linalg.norm(a @ f_impl.reshape(-1,m) - b)/np.linalg.norm(b)
    return f, op_err

def oracle(g, y, p):
    m = g.shape[1]
    h = np.vstack((g, np.sqrt(p['lam2'])*np.eye(m)))
    target = np.r_[y, np.zeros(m)]
    best, win = np.inf, None
    for k in range(m+1):
        for support in itertools.combinations(range(m), k):
            w = np.zeros(m, dtype=complex)
            if k:
                w[list(support)] = np.linalg.lstsq(h[:,support], target, rcond=None)[0]
            score = objective(g,y,p,w)
            if score < best:
                best, win = score, w
    return best, win

def objective(g,y,p,w):
    active = np.abs(w)>1e-8
    err = g@w-y
    return (0.5*np.vdot(err,err).real + 0.5*p['lam2']*np.vdot(w,w).real
            + p['lam0']*np.count_nonzero(active)
            + sum(c for u,v,c in p['pair_penalties'] if active[u] and active[v]))

rng = np.random.default_rng(4102)
worst = np.zeros(3)
counts = {}
for label in ('mixed-flow','pair-switch','negative-frequency-correlated','empty'):
    x,y = np.indices((32,32))/31
    basis = rng.normal(size=(11,32,32))
    basis[:4] += np.stack([np.ones_like(x), x, y, x*y])
    if label == 'negative-frequency-correlated':
        basis[9] = basis[2]+1e-4*basis[9]
    interior = rng.choice(900, size=45, replace=False)
    sensors = np.vstack(([[0,0],[0,31],[31,0],[31,31],[0,15],[16,31]], np.c_[1+interior//30, 1+interior%30]))
    p = dict(Lx=1.7, Ly=0.63, nu=0.0004 if label=='negative-frequency-correlated' else 0.005,
             vx=1.1*np.sin(9*y)+0.4*(x-0.5), vy=0.7*np.cos(7*x)-0.3*y,
             sigma=rng.uniform(-0.6,1.2,(32,32)), omega=-8.0 if label=='negative-frequency-correlated' else 2.3,
             basis=basis, sensor_ij=sensors, lam2=1e-8 if label=='negative-frequency-correlated' else 0.004,
             lam0=0.0003, pair_penalties=[[0,3,0.025],[3,7,0.04],[5,8,0.03]])
    p['vx'][0,:] = 0.0
    p['vy'][:,31] = 0.0
    f, op_err = forward(p)
    g = f[sensors[:,0],sensors[:,1],:]
    truth = np.zeros(11,complex)
    if label != 'empty':
        truth[[0,3,7]] = [1+0.4j,-0.9+0.7j,0.8-0.4j]
    obs = g@truth
    if label == 'pair-switch':
        # Each co-activation alone costs more than the empty model.
        p['pair_penalties'] = [[i,j,1+np.vdot(obs,obs).real] for i,j in itertools.combinations(range(11),2)]
        uncoupled = dict(p, pair_penalties=[])
        _, w0 = oracle(g,obs,uncoupled)
        assert np.count_nonzero(np.abs(w0)>1e-8) > 1
    p.update(y_obs_re=obs.real, y_obs_im=obs.imag)
    ref_score, ref_w = oracle(g,obs,p)
    start = time.perf_counter()
    out = mod.solve_nonhermitian_coupled_inversion(p)
    elapsed = time.perf_counter()-start
    assert out.keys() == {'weights_real','weights_imag','objective','predicted_field_abs'}
    for key, shape in [('weights_real',(11,)),('weights_imag',(11,)),('predicted_field_abs',(32,32))]:
        assert out[key].shape == shape and out[key].dtype == np.float64 and np.all(np.isfinite(out[key]))
    assert isinstance(out['objective'],float)
    w = out['weights_real'] + 1j*out['weights_imag']
    a = np.abs(f@ref_w)
    field_err = np.linalg.norm(out['predicted_field_abs']-a)/(np.linalg.norm(a) or 1.0)
    obj_err = max(abs(out['objective']-ref_score),abs(out['objective']-objective(g,obs,p,w)))
    counts[label] = int(np.count_nonzero(np.abs(w)>1e-8))
    assert op_err < 1e-11 and field_err <= 1e-5 and obj_err <= 1e-6, (label,op_err,field_err,obj_err)
    if label == 'pair-switch':
        assert counts[label] == 1
    if label == 'empty':
        assert counts[label] == 0 and np.all(out['predicted_field_abs'] == 0)
    worst = np.maximum(worst,[op_err,obj_err,field_err])
    print(f'{label}: operator residual {op_err:.3e}, objective error {obj_err:.3e}, field error {field_err:.3e}, active {counts[label]}, solve {elapsed:.3f}s')
print('Worst [operator residual, objective absolute error, field relative L2]:', worst)
print('Public fixture available:', pathlib.Path('/holdout_task/public_eq04_cases.pkl').is_file())

# Regression: an allowed duplicate basis and tiny ridge made the former
# normal equations singular. A singleton attains the known optimum to tolerance.
rng = np.random.default_rng(148)
nx = ny = 32
basis = rng.normal(size=(11, nx, ny))
basis[1] = basis[0]
ij = np.column_stack(divmod(rng.choice(nx*ny, 40, replace=False), ny))
p = dict(basis=basis, Lx=1., Ly=.7, nu=.003,
         vx=rng.uniform(-.2,.2,(nx,ny)), vy=rng.uniform(-.4,.4,(nx,ny)),
         sigma=np.zeros((nx,ny)), omega=-1.5, sensor_ij=ij,
         lam2=1e-20, lam0=.001, pair_penalties=[[0,1,.01]])
f, op_err = forward(p)
g = f[ij[:,0],ij[:,1]]
obs = g[:,0]*(1+.5j)
p.update(y_obs_re=obs.real, y_obs_im=obs.imag)
out = mod.solve_nonhermitian_coupled_inversion(p)
w = out['weights_real'] + 1j*out['weights_imag']
field = np.abs(f[...,0]*(1+.5j))
obj_err = max(abs(out['objective'] - (p['lam0'] + .5*p['lam2']*1.25)),
              abs(out['objective'] - objective(g,obs,p,w)))
field_err = np.linalg.norm(out['predicted_field_abs']-field)/np.linalg.norm(field)
assert np.count_nonzero(np.abs(w)>1e-8) == 1
assert op_err < 1e-11 and obj_err <= 1e-6 and field_err <= 1e-5
print(f'duplicate basis, tiny ridge: operator residual {op_err:.3e}, objective error {obj_err:.3e}, field error {field_err:.3e}')
print('Combined worst [operator residual, objective absolute error, field relative L2]:', np.maximum(worst,[op_err,obj_err,field_err]))
