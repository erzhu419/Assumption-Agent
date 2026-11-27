import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags
sp=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m98_v2_c04_s00_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; w=np.array([p['w_pulse'] for p in pulses]); y=np.array([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); ax0=m._axis

def design(p):
    def axis(k,v,L,r):
        j=0 if len(k)==s['nx'] else 1
        a=ax0(k,v,L,r).tolil(); h=L/(len(k)-1)
        for t,(i,kj) in enumerate(((0,1),(-1,-2))):
            rate=k[i]/h**2*p[j*4+2*t]; sink=r/h*p[j*4+2*t+1]
            a[i,kj]=-rate; a[i,i]=rate+sink
        return a.tocsc()
    m._axis=axis
    try: return m._forward(s)[0]
    finally: m._axis=ax0

def err(p): return w@design(p).T-y

def report(label,p):
    e=err(p).reshape(2,3,-1); print(label,'params',np.round(p,7),'pulse x shift RMSE',np.sqrt(np.mean(e**2,axis=-1)).round(8))

base=np.ones(8); report('baseline',base)
for train in [0,1,None]:
    def fun(p):
        e=err(p); return (e if train is None else e[train]).ravel()
    r=least_squares(fun,base,bounds=(0,12),gtol=1e-13,ftol=1e-13,xtol=1e-13)
    report(f'fit {train}',r.x); print('singular values',np.linalg.svd(r.jac,compute_uv=False).round(7))
    if train is None:
        for d,q in zip('xy',r.x.reshape(2,4)):
            h=s['L'+d]/(s['n'+d]-1); kap=s['kappa_'+d][[0,-1]]; vel=s['vel_'+d][[0,-1]]; rob=s['robin_'+d]; print(d,'h, kap, vel, rob',h,kap,vel,rob,'absolute coupling/sink',q[::2]*kap/h**2,q[1::2]*rob/h)
