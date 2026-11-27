import importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('ref','/tmp/hegel_continual_archive/beta_m98_v2_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); Q=np.linalg.qr(s['Z_nuisance'],mode='reduced')[0]; axis=m._axis

def prediction(p,kind):
    ix=iter(p)
    def wrapped(k,v,h):
        A=axis(k,v,h).tolil()
        for i,j in ((0,1),(len(k)-1,len(k)-2)):
            d=next(ix); A[i,i]+=d/h
            if kind in ('edge','full'):
                t=next(ix)*k[i]/h**2; A[i,i]+=t; A[i,j]-=t
            if kind in ('wrap','full'):
                t=next(ix)*k[i]/h**2; A[i,i]+=t; A[i,len(k)-1-i]-=t
        return A.tocsc()
    m._axis=wrapped
    return m._forward(s)[0]@W

def resid(p,kind):
    r=prediction(p,kind)-Y
    return r-Q@(Q.T@r)
for kind,n in [('loss',4),('edge',8),('wrap',8),('full',12)]:
    for train in ([0],[1],[0,1]):
        fit=least_squares(lambda p:resid(p,kind)[:,train].ravel(),np.zeros(n),gtol=1e-13,xtol=1e-12,ftol=1e-12,max_nfev=150)
        r=resid(fit.x,kind); print(kind,train,'RMSE',np.sqrt(np.mean(r*r,axis=0)),'raw RMSE',np.sqrt(np.mean((prediction(fit.x,kind)-Y)**2,axis=0)),'params',fit.x.round(7),flush=True)
