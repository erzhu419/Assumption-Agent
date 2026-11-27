import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
ds=item['dataset']; diag=item['campaign_diagnostics']; out=m.solve_batch(ds)
tau,th=out['estimated_taus'],out['hereditary_weights']; assert 0.05<tau[0]<tau[1]<25
b=m._calibration(ds['calibration_load_cells']); t=ds['eval_times']; dt=ds['past_dt']; lab=diag['lab_id']; n=len(lab)
x=m._features(diag,diag)[:n]; q=m._coupon_inputs(diag,x,b); past,h,F=q
y=diag['strain_curves_obs']-diag['strain_0'][:,None]; r=diag['instant_rate_obs']; d=F@th[2:]
C,R=m._chamber(tau,th,h,x,t); G=2*t/(1+np.sqrt(1+t)); phi=-np.expm1(-t[:,None]/tau)
_,train_R=m._design(tau,q,dt,t); mem=h[:,None]-tau*train_R[:,:2]
# Perturb the reset law with separately retained branch states, then with a saturating drift-amplitude correction.
mem_c=-mem[:,None,:]*th[:2]*phi[None,:,:]; mem_r=-mem*th[:2]/tau
hd=(h-np.median(h)); u=np.c_[h-past[:,-1],mem[:,0]-mem[:,1],hd*d,hd*(x[:,0]-x[:,1])*d]
blocks={'partial memory':(mem_c,mem_r),'pair differences and cross-products':(u[:,None,:]*G[None,:,None],u),'rational amplitude':((d/(1+hd**2)-d)[:,None,None]*G[None,:,None],(d/(1+hd**2)-d)[:,None])}

def summarize(C,R):
    rm=lambda z: float(np.sqrt(np.mean(z*z)))
    labs=[(rm((C-y)[lab==k]),rm((R-r)[lab==k])) for k in np.unique(lab)]
    complete=~np.isnan(diag['coupon_features_raw']).any(axis=1)
    return np.round([*np.max(labs,axis=0),rm((C-y)[complete]),rm((R-r)[complete]),rm((C-y)[~complete]),rm((R-r)[~complete]),max(rm(z) for z in (C-y).T)],6)

print('columns: worst-lab strain/rate; complete strain/rate; missing strain/rate; worst horizon strain')
print('candidate',summarize(C,R))
for name,(bc,br) in blocks.items():
    cc=C.copy(); rr=R.copy()
    for l in np.unique(lab):
        test=lab==l; fit=~test
        a=least_squares(lambda a: np.r_[((C+bc@a-y)[fit]).ravel(),np.sqrt(len(t))*(R+br@a-r)[fit]],np.zeros(br.shape[1]),loss='soft_l1',f_scale=.005).x
        cc[test]+=bc[test]@a; rr[test]+=br[test]@a
    print(name,summarize(cc,rr))
# A rational time-rate law preserves the initial rate; estimate its scale only outside the scored lab.
branch=C-d[:,None]*G; cr=C.copy()
for l in np.unique(lab):
    fit=lab!=l; test=~fit
    def g(p):
        a=np.exp(p[0]); return np.arctan(a*t)/a
    p=least_squares(lambda p: (branch[fit]+d[fit,None]*g(p)-y[fit]).ravel(),[-1.],loss='soft_l1',f_scale=.005,bounds=(-8,5)).x
    cr[test]=branch[test]+d[test,None]*g(p)
print('rational time rate',summarize(cr,R))
print('lab strain/rate mean residuals',[(int(l),np.round([np.mean((C-y)[lab==l]),np.mean((R-r)[lab==l])],6).tolist()) for l in np.unique(lab)])
