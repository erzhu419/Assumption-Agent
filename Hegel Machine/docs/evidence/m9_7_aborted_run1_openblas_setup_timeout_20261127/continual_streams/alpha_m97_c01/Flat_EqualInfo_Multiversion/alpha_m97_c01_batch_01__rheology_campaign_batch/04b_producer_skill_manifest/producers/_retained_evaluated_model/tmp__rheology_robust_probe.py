import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m = runpy.run_path('/holdout_task/alpha_m97_c01_batch_01_solver.py')
d = pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']
s = d['train']; dt,t = d['past_dt'],d['eval_times']; data,n = m['_prepare'](d)
state = tuple(v[:n] for v in m['_state'](data,dt,d['crit_stress'],True))
y = s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; design=m['_design']

def mad(e):
    return max(1.4826*np.median(np.abs(e-np.median(e))),1e-8)

def fit(mask, kind):
    ss=tuple(v[mask] for v in state); yy=y[mask]; rr=r[mask]
    p0,w0=m['_fit'](ss,dt,t,True,yy,rr)
    A,B=design(p0,ss,dt,t,True)
    sc,sr=mad(A@w0-yy),mad(B@w0-rr)
    print('scales',kind,round(sc,4),round(sr,4),'beyond 5 MAD',np.sum(abs(A@w0-yy)>5*sc),np.sum(abs(B@w0-rr)>5*sr),flush=True)
    if kind=='ols': return p0,w0
    target=np.r_[yy.ravel()/sc,rr/sr]
    def profile(p, ret=False):
        a,b=design(p,ss,dt,t,True)
        x=np.vstack((a.reshape(-1,9)/sc,b/sr))
        w=np.linalg.lstsq(x,target,rcond=None)[0]
        for _ in range(10):
            e=(x@w-target)/2.0
            q=np.power(1+e**2,-0.5 if kind=='soft_l1' else -1.)
            v=np.sqrt(q)
            w=np.linalg.lstsq(x*v[:,None],target*v,rcond=None)[0]
        if ret: return w
        e=x@w-target
        if kind=='soft_l1': return e*np.sqrt(2/(np.sqrt(1+(e/2)**2)+1))
        return np.sign(e)*2*np.sqrt(np.log1p((e/2)**2))
    fits=[least_squares(profile,p,bounds=(-5.,5.5),max_nfev=75) for p in (p0,p0[::-1],np.log([0.6,3.5]))]
    best=min(fits,key=lambda f:f.cost)
    return best.x,profile(best.x,True)

for kind in ('ols','soft_l1','cauchy'):
    pc=np.empty_like(y); pr=np.empty_like(r); pars=[]
    for lab in range(4):
        ix=s['lab_id']==lab; p,w=fit(~ix,kind); pars.append(np.exp(p))
        a,b=design(p,tuple(v[ix] for v in state),dt,t,True); pc[ix]=a@w; pr[ix]=b@w
    print(kind,'fold taus',np.round(pars,4))
    for name,ix in [(f'lab{i}',s['lab_id']==i) for i in range(4)]+[('active',state[3]>0),('inactive',state[3]==0),('missing',np.isnan(s['coupon_features_raw']).any(axis=1))]:
        a,b=abs(pc[ix]-y[ix]),abs(pr[ix]-r[ix]); print(kind,name,'MAE/median/90p curves',np.mean(a),np.median(a),np.quantile(a,.90),'rate RMSE/median',np.mean(b*b)**.5,np.median(b))
