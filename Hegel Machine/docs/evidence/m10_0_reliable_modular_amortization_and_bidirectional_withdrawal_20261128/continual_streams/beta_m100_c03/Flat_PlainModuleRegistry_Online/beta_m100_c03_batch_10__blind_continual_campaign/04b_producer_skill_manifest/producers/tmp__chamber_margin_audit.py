import itertools
import pickle
import sys
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis, _box_simplex, _responses

with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    s = calibrate_upstream(pickle.load(stream)[0]['study'])
nx, ny = s['nx'], s['ny']
shifts = s['mu_shifts']
sensors = np.asarray(s['sensor_ij'])
rows = sensors[:, 0] + nx*sensors[:, 1]
source = np.column_stack([a.ravel(order='F') for a in s['source_modes']])
y = s['y_obs_raw'].ravel()
z = s['Z_nuisance']
cap, budget, sr, nr = [s[k] for k in ('u_box','total_budget','source_ridge','nuisance_ridge')]
axes = []
for d, n in zip('xy', (nx,ny)):
    h = s['L'+d]/(n-1)
    k, v = s['kappa_'+d], s['vel_'+d]
    reflect = _axis(k,v,h,False)
    periodic = _axis(k,v,h,True)
    leakage = np.zeros(n)
    leakage[[0,-1]] = s['robin_'+d]/h
    axes.append((reflect, periodic-reflect, diags(leakage,format='csc')))
ident = eye(nx*ny, format='csc')
reaction = diags(s['V_reaction_2d'].ravel(order='F'), format='csc')

def design(law):
    ax,ay = [R + wrap*C + loss*D for (R,C,D), (wrap,loss) in zip(axes, law)]
    L = kron(eye(ny),ax,format='csc') + kron(ay,eye(nx),format='csc') + reaction
    return np.vstack([splu(L+mu*ident).solve(source)[rows] for mu in shifts])

choices = [(v, 0.) for v in (0.,.25,.5,.75,1.)]+[(0.,v) for v in (.025,.1,1.)]
laws = list(itertools.product(choices,repeat=2))
base = ((1.,0.),(0.,0.))
designs = {law:design(law) for law in laws}
np.testing.assert_allclose(designs[base],_responses(s)[0],rtol=1e-12,atol=1e-13)

def fold(train, test):
    Z = z[train]
    P = np.linalg.solve(Z.T@Z+nr*np.eye(z.shape[1]),Z.T)
    t = y[train]-Z@(P@y[train])
    def res(S):
        A = S[train]
        X = A-Z@(P@A)
        w,_ = _box_simplex(X.T@X+sr*np.eye(X.shape[1]),-X.T@t,cap,budget)
        return S[test]@w + z[test]@(P@(y[train]-A@w))-y[test]
    return res

horizon = np.repeat(np.arange(len(shifts)),len(rows))
rosette = np.arange(len(y))//4
bank = s['row_bank_id']
rms = lambda e: np.sqrt(np.mean(e**2))
fixed, selected = [], []
for k in range(len(shifts)):
    test = horizon==k
    inner = [fold(~(test|m),m) for j in np.unique(rosette[~test]) if np.any(m:=(rosette==j)&~test)]
    rank = lambda law: np.mean([rms(f(designs[law]))**2 for f in inner])
    winner = min(laws,key=rank)
    outer = fold(~test,test)
    a,b = outer(designs[base]),outer(designs[winner])
    fixed.append(rms(a)); selected.append(rms(b))
    print('held shift',k,'inner selection',winner,'outer fixed/selected',fixed[-1],selected[-1], 'fixed bank mean residuals',[float(a[bank[test]==j].mean()) for j in np.unique(bank)])
print('worst shift RMSE, fixed/expanded nested:',max(fixed),max(selected))
for sym in (0.,1.):
    alt = ((sym,0.),(sym,0.))
    errors = [rms(fold(horizon!=k,horizon==k)(designs[alt])) for k in range(len(shifts))]
    print('symmetric control',alt,'per-shift RMSE',errors)
