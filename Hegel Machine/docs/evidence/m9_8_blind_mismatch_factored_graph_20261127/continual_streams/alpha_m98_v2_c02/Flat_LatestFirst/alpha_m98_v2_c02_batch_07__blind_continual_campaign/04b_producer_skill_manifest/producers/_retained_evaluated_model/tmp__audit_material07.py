import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
i = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = i['dataset']; tr, te, c = d['train'], d['test'], i['campaign_diagnostics']
S = m._stress_calibration(d['calibration_load_cells']); dt,t=d['past_dt'],d['eval_times']
X,_ = m._standardize(tr,te)
_,C = m._standardize({k:v[:0] for k,v in c.items()},c)
D = m._design(tr,X,S,dt,t)
Y = tr['strain_curves_obs']-tr['strain_0'][:,None]; V=tr['instant_rate_obs']; labs=tr['lab_id']

def fit(j, mode):
    def sub(T):
        a,b=D(T); return a[j],b[j]
    T,H=m._fit(sub,Y[j],V[j])
    if mode=='current': return T,H
    def pair(p):
        ts=np.exp(p[:2]); a,b=sub(ts); h=p[2:]; return a@h-Y[j],b@h-V[j]
    p=np.r_[np.log(T),H]
    e,f=pair(p)
    def mad(z,axis=None):
        return 1.4826*np.median(np.abs(z-np.median(z,axis=axis,keepdims=True)),axis=axis)
    scale=mad(e) if mode!='per_time' else np.maximum(mad(e,axis=0),.5*mad(e))
    rate=mad(f)
    def fun(p):
        e,f=pair(p); return np.r_[(e/scale).ravel(),f/rate]
    ans=least_squares(fun,p,loss='cauchy' if mode=='tighter' else 'soft_l1',f_scale=1.1 if mode=='tighter' else 1.5,x_scale='jac',max_nfev=150)
    return np.exp(ans.x[:2]),ans.x[2:]

for mode in ('current','tighter','per_time'):
    errors=[]; params=[]
    for l in np.unique(labs):
        j=labs==l; T,H=fit(~j,mode); params.append(T)
        a,b=D(T); e=a[j]@H-Y[j]; f=b[j]@H-V[j]
        errors.append([np.quantile(abs(z),.9) for z in (e[:,:4],e[:,-4:],f)])
    T,H=fit(np.ones(len(labs),bool),mode)
    y,v=m._chamber(c,C,S,dt,t,T,H)
    e=y+c['strain_0'][:,None]-c['strain_curves_obs']; f=v-c['instant_rate_obs']
    ce=[]
    for l in np.unique(c['lab_id']):
        j=c['lab_id']==l
        ce.append([np.sqrt(np.mean(z*z)) for z in (e[j,:4],e[j,-4:],f[j])])
    print(mode,'taus',T,'held-out tau span',np.ptp(params,axis=0),'worst held-out benchtop p90',np.max(errors,axis=0),'worst verification lab RMSE',np.max(ce,axis=0))
