import itertools
import pickle
import sys
import numpy as np
from scipy.linalg import solve
sys.path.insert(0, '/holdout_task')
from downstream_solver import _design
from _verified_upstream import calibrate_upstream

def axis(s, d, kind):
    k, v = s['kappa_'+d], s['vel_'+d]
    n = len(k)
    h = s['L'+d]/(n-1)
    a = np.zeros((n,n))
    for i in range(n):
        if kind == 'periodic' or 0 < i < n-1:
            j, l = (i-1)%n, (i+1)%n
            a[i,j] += -(k[i]+k[j])/(2*h*h)-max(v[i],0)/h
            a[i,l] += -(k[i]+k[l])/(2*h*h)-max(-v[i],0)/h
        else:
            a[i,1 if i == 0 else n-2] = -k[i]/h**2
        a[i,i] = -a[i].sum() + (s['robin_'+d]/h if kind == 'robin' and i in (0,n-1) else 0)
    return a

kinds = ['robin','reflecting','periodic']
models = list(itertools.product(kinds, repeat=2))
expected = models.index(('periodic','reflecting'))
worst_pulse = worst_shift = worst_design = worst_profiled = 0.
margin = np.inf
folds = 0
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    s = calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
    pulses = s['reference_tracer_pulses']
    w = np.column_stack([p['w_pulse'] for p in pulses]); y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    nx, ny = s['nx'],s['ny']; mu = s['mu_shifts']; ident = np.eye(nx*ny)
    v = np.diag(s['V_reaction_2d'].ravel(order='F'))
    modes = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
    ix, iy = np.asarray(s['sensor_ij']).T
    errors = []
    for j,(bx,by) in enumerate(models):
        a = np.kron(np.eye(ny),axis(s,'x',bx)) + np.kron(axis(s,'y',by),np.eye(nx)) + v
        d = np.concatenate([solve(a+t*ident,modes)[ix+nx*iy] for t in mu])
        errors.append(d@w-y)
        if j == expected:
            worst_design = max(worst_design,np.max(np.abs(d-_design(s)[0])))
    errors = np.array(errors)
    z = s['Z_nuisance']; p = solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T,assume_a='pos')
    profiled = errors - z@(p@errors)
    worst_profiled = max(worst_profiled,float(np.sqrt(np.mean(profiled[expected]**2))))
    masks = [(np.ones(len(y),dtype=bool),np.arange(y.shape[1]) != j) for j in range(y.shape[1])]
    masks += [(np.arange(len(y))//len(ix) != j,np.ones(y.shape[1],dtype=bool)) for j in range(len(mu))]
    for i,(rows,cols) in enumerate(masks):
        train = errors[:,rows][:,:,cols]
        scores = np.sqrt(np.mean(train**2,axis=(1,2)))
        best,second = np.argsort(scores)[:2]
        assert best == expected, (i,models[best],scores)
        margin = min(margin, scores[second]/scores[best])
        if i < y.shape[1]:
            e=errors[best,:,i].reshape(len(mu),-1,4)
            worst_pulse=max(worst_pulse,float(np.sqrt(np.mean(e**2,axis=1)).max()))
        else:
            e=errors[best,~rows,:].reshape(-1,4,y.shape[1])
            worst_shift=max(worst_shift,float(np.sqrt(np.mean(e**2,axis=0)).max()))
        folds += 1
assert folds and max(worst_pulse,worst_shift) < 5e-6 and worst_design < 1e-12
print(f'RESOLVES ng_t05: periodic x / reflecting y selected in all {folds} held-out folds among {len(models)} boundary combinations; minimum runner-up/train RMSE ratio={margin:.3g}')
print(f'Worst held-out pulse/shift/channel RMSE={worst_pulse:.9g}; worst held-out shift/channel/pulse RMSE={worst_shift:.9g}; profiled pulse RMSE={worst_profiled:.9g}; max production design disagreement={worst_design:.9g}')
