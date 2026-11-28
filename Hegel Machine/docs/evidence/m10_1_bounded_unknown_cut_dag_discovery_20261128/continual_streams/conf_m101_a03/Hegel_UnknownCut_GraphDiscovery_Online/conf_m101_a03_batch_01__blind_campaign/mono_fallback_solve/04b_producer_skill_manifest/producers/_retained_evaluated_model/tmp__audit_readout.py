import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_batch.pkl','rb') as f:
    s = pickle.load(f)[0]['study']
d = s['train']
x, y, traj = (d[k] for k in ('x','y','traj'))
bias, gain = m._calibration(s['panel_a'])
panel = s['panel_b']
op = m._fit(m._affine(panel['u']),panel['v'])
z = m._affine((x-bias)/gain)@op
base = m._readout_design(z)
pairs = (z[:,0]*z[:,2],z[:,1]*z[:,3])
expanded = np.column_stack((base,pairs[0]-pairs[1]))
rows = np.arange(len(x))
folds = [('52/20', rows[52:]),('56/16', rows[56:])]
for seed in (18,41):
    for k,j in enumerate(np.array_split(np.random.default_rng(seed).permutation(rows),4)):
        folds.append((f'shuffle-{seed}-{k}',j))
# Select cuts by the input geometry, not by labels or fitted residuals.
for axis in (z[:,0]-z[:,2],z[:,1]-z[:,3],base[:,-1]):
    order = np.argsort(axis)
    folds.extend([(f'tail-{len(folds)}-low',order[:16]),(f'tail-{len(folds)}-high',order[-16:])])

def challenger(X,i,j,link):
    def predict(p,F):
        q = F@p[:-3]
        offset, amplitude, slope = p[-3:]
        h = slope*q+offset
        corr = h/(1+h*h) if link=='rational' else np.tanh(h)
        return q+amplitude*corr
    b = m._fit(X[i],y[i])
    opts = [least_squares(lambda p:predict(p,X[i])-y[i],np.r_[b,0.,0.45,w],bounds=(np.r_[np.full(len(b)+2,-np.inf),.12],np.r_[np.full(len(b)+2,np.inf),4.]),max_nfev=180) for w in (.4,.8,1.6,2.5)]
    return predict(min(opts,key=lambda p:p.cost).x,X[j])

records = []
for name,j in folds:
    i = np.setdiff1d(rows,j)
    result = m.solve_batch(dict(s,train=dict(d,x=x[i],y=y[i],traj=traj[i]),test={'x':x[j]}))
    pred, state = result['y_hat'], result['state_pred']
    np.testing.assert_array_equal(result['diagnostics_vec'],[pred.mean(),pred.std(),state[:,-1].mean(),state[:,-1].std()])
    quad = base[j]@m._fit(base[i],y[i])
    alternatives = [quad,challenger(base,i,j,'rational'),challenger(expanded,i,j,'tanh'),challenger(expanded,i,j,'rational')]
    state_err = np.abs(state-traj[j]).mean(0)
    report = [np.mean(np.abs(v-y[j])) for v in [pred,*alternatives]] + [state_err.mean(),state_err.max()]
    records.append(report)
    print(name, ' '.join(f'{r:.6f}' for r in report))
print('Columns: saved tanh, pooled quadratic, rational, tanh with interaction contrast, rational with interaction contrast, trajectory, worst timestamp')
print('Worst:', np.max(records,axis=0).round(6))
assert np.max(records,axis=0)[0] <= .038
