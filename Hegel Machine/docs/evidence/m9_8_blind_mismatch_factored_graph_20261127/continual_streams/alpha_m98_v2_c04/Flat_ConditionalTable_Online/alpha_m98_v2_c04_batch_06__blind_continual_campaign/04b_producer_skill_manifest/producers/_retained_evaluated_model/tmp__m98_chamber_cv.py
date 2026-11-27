import pickle
import numpy as np
from scipy.optimize import least_squares
from m98core import _bridge, _loads, _features, _fit

item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d, q = item['dataset'], item['campaign_diagnostics']
tr, te = d['train'], d['test']
dt, times = d['past_dt'], d['eval_times']
bridge = _bridge(d['calibration_load_cells'])
tau, theta = _fit(_features(tr,te)[:len(tr['lab_id'])], *_loads(tr,bridge), dt,times,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs'])
past, hold = _loads(q, bridge)
y = q['strain_curves_obs']-q['strain_0'][:,None]
r = q['instant_rate_obs']
labs = q['lab_id']
mem=np.zeros((len(y),2))
for h,p in zip(dt,past.T):
    mem = np.exp(-h/tau)*mem - np.expm1(-h/tau)*p[:,None]
archive = np.array([.4718099191421826,2.4756939307917163])

def solve_chamber(x, fit_rows, method):
    drift = theta[2] + x@theta[3:]
    inputs = np.column_stack([hold,drift])
    base = tau if method=='intrinsic' else archive if method=='archive-time' else tau*np.array([1.,.55])
    free = method in ('free','history')
    def pred(z):
        t = np.exp(z[:2]) if free else base
        c = z[2:] if free else z
        amplitudes = inputs @ c[:6].reshape(3,2).T
        if method=='history': amplitudes[:,:2] += mem*c[6:]
        return amplitudes @ np.vstack((-np.expm1(-times[None,:]/t[:,None]),times)), amplitudes @ np.r_[1/t,1.]
    def loss(z):
        a,b = pred(z)
        return np.r_[((a-y)[fit_rows]/.005).ravel(), (b-r)[fit_rows]/.004]
    guess=np.array([.619,.168,.281,1.392,.031,.081])
    if method=='history': guess=np.r_[guess,0.,0.]
    if free: guess=np.r_[np.log(base),guess]
    z=least_squares(loss,guess,loss='soft_l1',max_nfev=300).x
    return z,pred(z)

allblock={k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
preps={'diag':_features(q,q)[-len(y):], 'joint':_features(allblock,q)[-len(y):], 'test':_features(te,q)[-len(y):]}
for label,x in preps.items():
    for name in (('intrinsic','archive-time','ratio-time','free','history') if label=='diag' else ('free',)):
        z,_=solve_chamber(x,np.ones(len(y),dtype=bool),name)
        print(label,name,'fit',z.round(6))
        for lab in np.unique(labs):
            held=labs==lab
            _,(yp,rp)=solve_chamber(x,~held,name)
            print(' lab',lab,'strain, rate rmse',np.round([np.mean((yp[held]-y[held])**2)**.5,np.mean((rp[held]-r[held])**2)**.5],6))
