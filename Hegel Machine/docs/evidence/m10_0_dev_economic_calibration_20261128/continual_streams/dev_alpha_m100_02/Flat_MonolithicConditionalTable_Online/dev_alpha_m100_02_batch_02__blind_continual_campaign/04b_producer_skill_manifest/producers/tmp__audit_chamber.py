import importlib.util
import pickle
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']
L, R = c['true_channel_loads'], c['bridge_raw']
# The public sweeps show gross spikes well outside the normal raw range.
clean = np.max(abs(R), axis=1) < 6
L, R = L[clean], R[clean]
pairs = list(combinations(range(4), 2))
errors = np.empty((3, len(L), 4))
for j in range(4):
    q = (j + 1) % 4
    x, y = L[:, j], L[:, q]
    a = np.column_stack((np.ones(len(L)), L, x**3, x*y))
    extended = np.column_stack((a, L**2, *(L[:, u]*L[:, v] for u, v in pairs if {u,v} != {j,q})))
    # Separate self, sum and difference saturation; choose coefficients inside each fold.
    Q2 = np.column_stack((x*x, (x+y)**2, (x-y)**2))
    for i in range(len(L)):
        keep = np.arange(len(L)) != i
        beta = np.linalg.lstsq(a[keep], R[keep,j], rcond=None)[0]
        gamma = np.linalg.lstsq(extended[keep], R[keep,j], rcond=None)[0]
        fit = least_squares(lambda p: (a[keep]@p[:7])/(1+Q2[keep]@p[7:])-R[keep,j], np.r_[beta, 0.,0.,0.], bounds=(np.r_[np.full(7,-np.inf),0.,0.,0.],np.inf), gtol=1e-12).x
        errors[:,i,j] = [a[i]@beta-R[i,j], extended[i]@gamma-R[i,j], (a[i]@fit[:7])/(1+Q2[i]@fit[7:])-R[i,j]]
for name,e in zip(('cyclic cubic','expanded pair/quadratic','rational saturation'),errors):
    print(name, 'worst channel LOO RMSE', np.sqrt(np.mean(e**2,axis=0)).max(), 'max error', abs(e).max())
B = m._bridge(c)
j = np.arange(4)
q = (j+1)%4
for name in ('train','test'):
    p = d[name]
    for key in ('past_channel_raw','hold_channel_raw'):
        raw = p[key].reshape(-1,4)
        u = m._loads(raw,B)
        resid = B[0] + u@B[1:5] + B[5]*u**3 + B[6]*u*u[:,q] - raw
        jac = np.broadcast_to(B[1:5].T, (len(u),4,4)).copy()
        jac[:,j,j] += 3*B[5]*u**2+B[6]*u[:,q]
        jac[:,j,q] += B[6]*u
        print(name,key,'max inversion residual',abs(resid).max(),'worst condition',np.linalg.cond(jac).max())
        assert abs(resid).max() < 1e-9
tr, te = d['train'], d['test']
X, _ = m._features(tr,te)
s = m._prepare(tr,X,B)
t, dt = d['eval_times'], d['past_dt']
Y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    keep = tr['lab_id'] != lab
    tau, theta = m._fit(tuple(a[keep] for a in s),dt,t,Y[keep],r[keep])
    ac, ar = m._design(np.log(tau),tuple(a[~keep] for a in s),dt,t)
    ec, er = ac@theta-Y[~keep], ar@theta-r[~keep]
    print('held-out lab',lab,'median signed strain/rate residual',np.median(ec),np.median(er),'rate p90',np.quantile(abs(er),.9))
