import importlib.util
import pickle
from itertools import combinations
import numpy as np

s = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
z, target = c['bridge_raw'], c['true_channel_loads'].sum(1)
z = (z-z.mean(0))/z.std(0)
linear = np.c_[np.ones(len(z)), z]
a, b = np.triu_indices(4)
quad = np.c_[linear, z[:,a]*z[:,b]]
qs = [*z.T] + [z[:,i]+sign*z[:,j] for i,j in combinations(range(4),2) for sign in [-1,1]]
sat = np.c_[linear, *[q/(1+.2*q*q) for q in qs]]
order = np.argsort(target)
blocks = [np.arange(len(z))[i::5] for i in range(5)] + [order[:16],order[-16:]]
for name, mat in [('affine',linear),('quadratic',quad),('pair saturation',sat)]:
    err = []
    for idx in blocks:
        tr = np.ones(len(z),dtype=bool); tr[idx] = False
        p = np.linalg.lstsq(mat[tr],target[tr],rcond=None)[0]
        err.append(np.sqrt(np.mean((mat[idx]@p-target[idx])**2)))
    print(name, 'bridge held-out RMSE, interleaved then stress tails:', np.round(err,7))

tr = d['train']; dt,t = d['past_dt'],d['eval_times']; coupon,_ = m._prepare(d)
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]; r = tr['instant_rate_obs']
for key in ('lab_id','stratum'):
    for v in np.unique(tr[key]):
        keep = tr[key]!=v
        tau,theta = m._fit(dt,t,tuple(a[keep] for a in coupon),y[keep],r[keep])
        A,B = m._design(tau,dt,t,tuple(a[~keep] for a in coupon))
        ec,er = A@theta-y[~keep], B@theta-r[~keep]
        print(key,v,'n',len(er),'curve worst-horizon MdAE',round(np.median(abs(ec),axis=0).max(),7),'curve largest signed horizon median',round(abs(np.median(ec,axis=0)).max(),7),'rate MdAE',round(np.median(abs(er)),7),'rate signed median',round(np.median(er),7))
