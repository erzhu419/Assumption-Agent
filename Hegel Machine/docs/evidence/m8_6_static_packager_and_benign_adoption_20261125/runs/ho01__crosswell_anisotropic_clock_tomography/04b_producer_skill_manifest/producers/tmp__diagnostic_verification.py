import pickle
import sys
from time import perf_counter
from collections import defaultdict
import numpy as np
from scipy.special import roots_legendre
sys.path.insert(0, '/holdout_task')
from diagnostic_solver import invert_plume_and_clocks, _Rays, _bounded_clocks
from scipy.optimize import least_squares

stats = defaultdict(list)
def evaluate(label, name, survey, center, clock):
    start = perf_counter()
    result = invert_plume_and_clocks(survey)
    elapsed = perf_counter() - start
    xyz = np.asarray(result['plume_xz_m'])
    bias = np.asarray(result['rcv_clock_bias_s'])
    assert xyz.shape == (2,) and bias.shape == np.shape(clock)
    assert np.all(np.isfinite(xyz)) and np.all(np.isfinite(bias))
    assert 0 <= xyz[0] <= survey['W_m'] and 0 <= xyz[1] <= survey['H_m']
    assert abs(bias.sum()) < 1e-12 and np.max(np.abs(bias)) <= .004 + 1e-12
    err = np.linalg.norm(xyz - center)
    rmse = np.mean((bias - clock)**2)**.5
    stats[label].append((err, rmse, elapsed, name, tuple(xyz)))
    if err > 1.8 or rmse > .00025 or elapsed >= 15:
        print('MISS', label, name, 'center_m', round(err,4), 'clock_ms', round(1000*rmse,4), 'runtime_s',round(elapsed,4),'estimated',xyz,'true',center, flush=True)

nodes, weights = roots_legendre(100)
u, qw = (nodes+1)/2, weights/2
rng = np.random.default_rng(148522)
for i in range(120):
    w, h = rng.uniform(32,55), rng.uniform(70,95)
    ns,nr = rng.integers(12,20,size=2)
    sz = np.linspace(.07*h,.93*h,ns) + rng.uniform(-1,1,ns)
    rz = np.linspace(.06*h,.94*h,nr) + rng.uniform(-1,1,nr)
    position = i % 6
    if position < 3:
        center = np.array([rng.uniform(.15,.85)*w, rng.uniform(.16,.84)*h])
        region = 'interior'
    elif position < 5:
        center = np.array([rng.choice([1.,w-1.]), rng.uniform(.16,.84)*h])
        region = 'near well'
    else:
        center = np.array([rng.uniform(.13,.87)*w, rng.choice([2.,h-2.])])
        region = 'near depth edge'
    clock = rng.uniform(-1,1,nr)
    clock -= clock.mean()
    clock *= rng.choice([.0025,.004])/np.max(abs(clock))
    dz = rz[None,:] - sz[:,None]
    dl = np.hypot(w,dz)
    quad = dl * np.sum(qw*np.exp(-.5*((w*u-center[0])**2+(sz[:,None,None]+dz[:,:,None]*u-center[1])**2)/5.5**2),axis=-1)
    times = dl*rng.uniform(.00034,.00048)*(1-rng.uniform(.07,.24)*(dz/dl)**2) + rng.uniform(.000045,.000095)*quad + clock
    times += rng.normal(0,rng.choice([.00003,.000055]),times.shape)
    bad = rng.choice(times.size, int(.04*times.size+.5), replace=False)
    times.flat[bad] += rng.choice([-1,1],len(bad))*rng.uniform(.0054,.0062,len(bad))
    survey = dict(W_m=w,H_m=h,z_src_m=sz,z_rcv_m=rz,T_obs_s=times,R_plume_m=5.5)
    if i not in (25,33,59,71,83,87):
        continue
    result = invert_plume_and_clocks(survey)
    candidates, fit, fitted_cost, mask = result['diagnostics']
    rays = _Rays(w, 5.5, sz, rz)
    true_mask = np.ones_like(times,dtype=bool)
    true_mask.flat[bad] = False
    use = true_mask
    counts = use.sum(axis=0)
    def prof(p):
        e = times*1000 - (p[2]*rays.plume(*p[:2]) + rays.bg@p[3:]).reshape(times.shape)
        c = _bounded_clocks(np.where(use,e,0).sum(axis=0)/counts,counts)
        return (e-c)[use]
    def refit(xz):
        sol=least_squares(prof, np.r_[xz, .07, .4,.06], bounds=([0,0,0,0,-np.inf],[w,h,np.inf,np.inf,np.inf]), x_scale='jac', gtol=1e-11)
        return sol.x, sol.cost, np.sqrt(np.diag(np.linalg.pinv(sol.jac.T@sol.jac)))[:2] * np.std(sol.fun)
    k=rays.plume(*center)
    a=np.column_stack((k, rays.bg, np.tile(np.eye(nr)[:,:-1]-np.eye(nr)[:,-1:], (ns,1))))
    at_truth=np.linalg.lstsq(a[use.ravel()], times.ravel()[use.ravel()]*1000, rcond=None)[0]
    truth_cost=.5*np.linalg.norm(prof(np.r_[center,at_truth[:3]]))**2
    print('\ncase',i,'geometry',np.round([w,h],2),ns,nr,'center',np.round(center,4),'at-truth amplitude',round(at_truth[0],4),'truth profile loss',round(truth_cost,5),'mask errors',np.count_nonzero(mask!=true_mask))
    print('seed => cauchy loss, solution (x,z,ds,s0,s0eps), rejects')
    for x,z,cost,par,n in candidates:
        print(np.round([x,z],2),round(cost,4),np.round(par,4),n)
    print('final',np.round(fit,4),round(fitted_cost,5),'oracle mask from fitted',refit(fit[:2]),'oracle mask from truth',refit(center))
