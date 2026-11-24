import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

_NAMES = ('Vmax1','Km1','alpha_p','Vmax2','Km2','k_leak')

def _feed(t, amp, window):
    return amp if window[0] <= t < window[1] else 0.0

def _rhs(t, y, p, amp, window):
    V1,K1,a,V2,K2,k = p
    S,I,P = y
    v1 = V1*S/(K1+S+a*P)
    v2 = V2*I/(K2+I)
    return [-v1+_feed(t,amp,window), v1-v2-k*I, v2]

def _simulate(ts, y0, p, amp, window):
    ts=np.asarray(ts,float)
    out=np.empty((len(ts),3)); out[0]=y0
    cuts=sorted(set([float(ts[0]),float(ts[-1])] + [float(x) for x in window if ts[0]<x<ts[-1]]))
    y=np.asarray(y0,float).copy()
    for left,right in zip(cuts[:-1],cuts[1:]):
        ids=np.flatnonzero((ts>left+1e-12)&(ts<=right+1e-12))
        ev=np.r_[left,ts[ids]]
        sol=solve_ivp(_rhs,(left,right),y,args=(p,amp,window),t_eval=ev,rtol=2e-7,atol=2e-9,method='LSODA')
        if not sol.success: raise RuntimeError(sol.message)
        if len(ids): out[ids]=sol.y[:,1:].T
        y=sol.y[:,-1]
    return out

def identify_and_forecast(dataset):
    trs=dataset['train_trajectories']
    # Scale residuals by approximate 2.5% measurement noise, with a small floor near zero.
    scales=[]
    for tr in trs:
        Y=np.asarray(tr['y_obs'],float)
        scales.extend(np.maximum(np.median(np.abs(Y),axis=0)*0.025, [0.008,0.008,0.008]))
    scales=np.asarray(scales).reshape(len(trs),3)
    def residual(z, use=range(len(trs))):
        p=np.exp(z)
        rr=[]
        for j in use:
            tr=trs[j]; y=np.asarray(tr['y_obs'],float); t=np.asarray(tr['t_obs'],float)
            pred=_simulate(t,y[0],p,float(tr['feed_amp']),tr['feed_window'])
            rr.append(((pred-y)/scales[j]).ravel())
        return np.concatenate(rr)
    lower=np.log([0.15,0.04,0.08,0.15,0.04,0.005])
    upper=np.log([8.,4.,8.,8.,4.,1.])
    best=None
    # deterministic multistart across plausible kinetic scales
    starts=[[1.8,.65,1.25,1.35,.45,.12],[1.2,.3,.5,1.,.3,.08],[3.,1.,2.,2.,.8,.2],[.8,.15,3.,.8,.15,.04]]
    for s in starts:
        fit=least_squares(residual,np.log(s),bounds=(lower,upper),max_nfev=550,xtol=2e-9,ftol=2e-9,gtol=2e-8)
        cost=np.dot(fit.fun,fit.fun)
        if best is None or cost<best[0]: best=(cost,fit.x)
    p=np.exp(best[1])
    forecast=_simulate(dataset['t_test'],dataset['y0_test'],p,float(dataset['test_feed_amp']),dataset['test_feed_window'])
    return {'params':dict(zip(_NAMES,map(float,p))), 'forecast_SIP':forecast}
