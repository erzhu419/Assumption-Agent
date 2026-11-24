import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

_NAMES = ('Vmax1','Km1','alpha_p','Vmax2','Km2','k_leak')

def _feed(t, amp, window):
    return amp if window[0] <= t < window[1] else 0.0

def _simulate(times, y0, p, amp, window):
    v1m, km1, alpha, v2m, km2, leak = p
    def rhs(t, y):
        S, I, P = y
        v1 = v1m*S/(km1+S+alpha*P)
        v2 = v2m*I/(km2+I)
        return [-v1+_feed(t,amp,window), v1-v2-leak*I, v2]
    cuts = [float(times[0])] + [float(x) for x in window if times[0] < x < times[-1]] + [float(times[-1])]
    out = np.empty((len(times),3)); state=np.asarray(y0,float)
    for a,b in zip(cuts[:-1], cuts[1:]):
        ids=np.where((times>=a-1e-12)&(times<=b+1e-12))[0]
        if not len(ids): continue
        sol=solve_ivp(rhs,(a,b),state,t_eval=times[ids],rtol=2e-7,atol=2e-9,method='LSODA')
        if not sol.success: raise ValueError('ODE integration failed')
        out[ids]=sol.y.T; state=sol.y[:,-1]
    return out

def identify_and_forecast(dataset: dict) -> dict:
    trains=dataset['train_trajectories']
    def residual(q):
        p=np.exp(q); rr=[]
        for tr in trains:
            pred=_simulate(np.asarray(tr['t_obs'],float),tr['y0_approx'],p,float(tr['feed_amp']),tr['feed_window'])
            # Approximately constant relative observation noise; avoid letting the large-S
            # observations dominate the fit.
            scale=np.maximum(np.mean(np.abs(tr['y_obs']),axis=0)*0.025,0.025)
            rr.extend(((pred-tr['y_obs'])/scale).ravel())
        return rr
    starts=[np.log([1.8,.65,1.25,1.35,.45,.12]),np.log([1.5,.5,1.,1.2,.4,.1]),np.log([2.,.8,1.5,1.5,.6,.15])]
    best=None
    for q0 in starts:
        fit=least_squares(residual,q0,bounds=(np.log([.05,.03,.03,.05,.03,.002]),np.log([10,10,10,10,10,2])),max_nfev=1200,ftol=2e-10,xtol=2e-10,gtol=2e-10)
        if best is None or np.dot(fit.fun,fit.fun)<np.dot(best.fun,best.fun): best=fit
    p=np.exp(best.x)
    forecast=_simulate(np.asarray(dataset['t_test'],float),dataset['y0_test'],p,float(dataset['test_feed_amp']),dataset['test_feed_window'])
    return {'params':dict(zip(_NAMES,map(float,p))),'forecast_SIP':forecast}
