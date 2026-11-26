import importlib.util
import time
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('eq04_candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def reference_fields(p):
    a = lil_matrix((1024, 1024), dtype=np.complex128)
    hx, hy = p['Lx']/33, p['Ly']/33
    for i in range(32):
        for j in range(32):
            k = i*32 + j
            a[k,k] = 2*p['nu']/hx**2 + 2*p['nu']/hy**2 + p['sigma'][i,j] + 1j*p['omega']
            for h, v, loc, stride in [(hx,p['vx'][i,j],i,32),(hy,p['vy'][i,j],j,1)]:
                d = p['nu']/h**2
                if loc > 0: a[k,k-stride] = -d
                if loc < 31: a[k,k+stride] = -d
                a[k,k] += abs(v)/h
                if v >= 0 and loc > 0: a[k,k-stride] -= v/h
                if v < 0 and loc < 31: a[k,k+stride] += v/h
    rhs = p['basis'].reshape(11,1024).T
    return spsolve(a.tocsc(), rhs)

def score(g,y,w,p):
    active = np.abs(w)>1e-8
    return 0.5*np.linalg.norm(g@w-y)**2 + 0.5*p['lam2']*np.linalg.norm(w)**2 + p['lam0']*active.sum() + sum(c for i,j,c in p['pair_penalties'] if active[i] and active[j])

def brute(g,y,p):
    w = np.zeros(11, complex)
    best = score(g,y,w,p)
    winner = w.copy()
    for n in range(1,12):
        for s in combinations(range(11), n):
            columns = list(s)
            b = np.vstack((g[:,columns], np.sqrt(p['lam2'])*np.eye(n)))
            w[columns] = np.linalg.lstsq(b, np.r_[y, np.zeros(n)], rcond=None)[0]
            value = score(g,y,w,p)
            if value < best: best, winner = value, w.copy()
            w[columns] = 0
    return best,winner

rng = np.random.default_rng(940)
worst = np.zeros(4)
for name, nu, freq, scale, k in [('coupled',.02,1.7,1,79),('fast_boundary',.00003,-19,9,11),('reaction',.0003,.07,0,42),('collinear',.03,-.8,1,15),('null',.02,3.8,2,23)]:
    v = rng.normal(size=(2,32,32))*scale
    p = dict(Lx=.75,Ly=2.25,nu=nu,omega=freq,vx=v[0],vy=v[1],sigma=rng.uniform(-.3,.7,(32,32)),basis=rng.normal(size=(11,32,32)),lam0=1e-4,lam2=1e-5)
    if name == 'collinear':
        p['basis'][1] = p['basis'][0] + 1e-8*p['basis'][1]
        p['lam0'] = 1e-7
    f = reference_fields(p)
    ids = np.r_[0,31,992,1023, rng.choice(np.setdiff1d(np.arange(1024), [0,31,992,1023]), k-4, replace=False)]
    p['sensor_ij'] = np.column_stack((ids//32,ids%32))
    g = f[ids]
    truth = np.zeros(11, complex)
    truth[[0,1,6]] = [1.1+2j,-.4+1.1j,1.1-.7j]
    y = np.zeros(k,complex) if name == 'null' else g@truth
    p['y_obs_re'], p['y_obs_im'] = y.real,y.imag
    p['pair_penalties'] = [[0,1,10.0],[6,8,.2],[4,2,.1]] if name != 'collinear' else [[6,8,.01],[4,2,.01]]
    if name == 'coupled':
        unpenalized = brute(g,y,dict(p,pair_penalties=[]))[1]
        assert np.all(np.abs(unpenalized[:2]) > 1e-8)
    expected,wref = brute(g,y,p)
    t0 = time.perf_counter()
    got = mod.solve_nonhermitian_coupled_inversion(p)
    elapsed = time.perf_counter()-t0
    assert set(got) == {'weights_real','weights_imag','objective','predicted_field_abs'}
    assert isinstance(got['objective'], float)
    assert all(got[key].shape == (11,) and got[key].dtype == np.float64 for key in ['weights_real','weights_imag'])
    assert got['predicted_field_abs'].shape == (32,32) and got['predicted_field_abs'].dtype == np.float64
    w = got['weights_real']+1j*got['weights_imag']
    if name == 'null': assert not w.any()
    if name == 'coupled': assert not np.all(np.abs(w[:2]) > 1e-8)
    pred = np.abs(f@wref).reshape(32,32)
    ferr = np.linalg.norm(got['predicted_field_abs']-pred)/(np.linalg.norm(pred) or 1.)
    internal = mod._solve_fields(p,np.asarray(p['basis'],complex))
    pde_err = np.linalg.norm(internal-f)/np.linalg.norm(f)
    errors = np.array([abs(expected-got['objective']),ferr,abs(score(g,y,w,p)-got['objective']),pde_err])
    assert np.all(errors <= [1e-6,1e-5,1e-6,1e-11]), (name, errors)
    worst = np.maximum(worst, errors)
    print(name, 'objective/field/score/stencil errors',errors, 'seconds', round(elapsed,3))
print('Worst objective/field/score/stencil errors:', worst)
