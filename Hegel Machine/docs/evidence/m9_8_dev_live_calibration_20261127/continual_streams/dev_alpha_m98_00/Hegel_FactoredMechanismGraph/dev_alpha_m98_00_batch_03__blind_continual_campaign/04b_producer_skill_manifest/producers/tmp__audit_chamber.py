import importlib.util, pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d, q = item['dataset'], item['campaign_diagnostics']
tr = d['train']; dt, t = d['past_dt'], d['eval_times']
a, _ = s._prepare(d)
tau, th = s._fit(dt, t, a, tr['strain_curves_obs']-tr['strain_0'][:,None], tr['instant_rate_obs'])
cal = s._stress_calibration(d['calibration_load_cells']); stress = lambda x: x @ cal[:4] + cal[4]
x = s._aligned_features({k:v[:0] for k,v in q.items()}, q)
h = stress(q['hold_channel_raw']); past = stress(q['past_channel_raw']); coupon = (past,h,x)
drift = x @ th[2:]; C,R = s._design(tau,dt,t,coupon)
base = np.column_stack((C[:,:,:2]@th[:2], R[:,:2]@th[:2]))
gain = np.r_[-np.expm1(-t[:,None]/tau)@np.array([1.,5.])-.5*t, np.sum(np.array([1.,5.])/tau)-.5]
y = np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None], q['instant_rate_obs']))
e = y-base
pc,pr = s._chamber_response(tau,dt,t,coupon,th)
exact = np.column_stack((pc,pr)); np.testing.assert_allclose(exact,base+.43*drift[:,None]*gain,atol=1e-12)
missing = np.isnan(q['coupon_features_raw']).any(axis=1)

def metrics(pred, rows):
    r = (pred-y)[rows]; return np.round([np.sqrt(np.mean(r[:,:-1]**2)),np.sqrt(np.mean(r[:,-1]**2)),np.max(np.sqrt(np.mean(r**2,axis=1)))],6)
for field in ('lab_id','stratum'):
    for v in np.unique(q[field]):
        rows=q[field]==v; print('fixed',field,v,'n',rows.sum(),'curve/rate/worst',metrics(exact,rows),'signed rate bias',round(np.mean((exact-y)[rows,-1]),6))
for z in (False,True): print('fixed missing',z,metrics(exact,missing==z))
# Fit every coefficient and saturation scale inside each outer group split.
# Gap difference is measured in stress units and normalized on the fit fold.
gap = R[:,:2]*tau

def predict(kind, fit, target):
    if kind == 'rational':
        def design(x): return (drift/(1+x*drift**2))[:,None]*gain
        def loss(x):
            a=design(x)[fit]; b=np.sum(a*e[fit])/np.sum(a*a); return np.sum((a*b-e[fit])**2)
        v=minimize_scalar(loss,bounds=(0,10),method='bounded').x
        if loss(0)<=loss(v): v=0
        A=design(v)[:,:,None]
    else:
        cols=[drift]
        if kind == 'gap contrast':
            z=gap[:,0]-gap[:,1]; cols.append(drift*z/max(np.std(z[fit]),1e-9))
        if kind == 'cubic interaction':
            cols.append(drift*gap[:,0]*gap[:,1]/max(np.mean(np.abs(gap[fit,0]*gap[fit,1])),1e-9))
        A=np.stack(cols,axis=1)[:,None,:]*gain[None,:,None]
    beta=np.linalg.lstsq(A[fit].reshape(-1,A.shape[2]),e[fit].ravel(),rcond=None)[0]
    return base[target]+A[target]@beta
for field in ('lab_id','stratum'):
    labels=q[field]
    for kind in ('linear','rational','gap contrast','cubic interaction'):
        pred=np.empty_like(y)
        slices=[]
        for value in np.unique(labels):
            rows=labels==value; pred[rows]=predict(kind,~rows,rows); slices.append(metrics(pred,rows))
        print('held out',field,kind,'pooled',metrics(pred,np.ones(len(y),bool)),'worst-group',np.max(slices,axis=0))
