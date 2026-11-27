import sys,pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as bs
import downstream_solver as ds
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    item=pickle.load(f)[0]
data=item['dataset']; diag=item['campaign_diagnostics']
cal=calibrate_upstream(data,bs._PAY); dcal=calibrate_upstream(dict(data,test=diag),bs._PAY)['test']
out=bs.solve_batch(data)
for key,value in ds.solve_downstream(cal).items(): np.testing.assert_array_equal(value,out[key])
a=out['estimated_taus']; w=out['hereditary_weights']; assert .05<a[0]<a[1]<25
print('material tau/weights',a,w)
tr,te=cal['train'],cal['test']; n=len(tr['lab_id']); x=ds._features(tr,te)
dt=np.asarray(cal['past_dt']); t=np.asarray(cal['eval_times'])
b=np.r_[cal['total_load_projection'][0],cal['total_load_projection'][1]]
rm=lambda e: np.sqrt(np.mean(e**2))
lab=dcal['lab_id']; complete=np.isfinite(dcal['coupon_features_raw']).all(1)
xd=ds._features(dcal)
cp=ds._coupon(dcal,xd,b); curve,rate=ds._design(a,cp,dt,t)
B=curve[:,:,:2]@w[:2]; I=rate[:,:2]@w[:2]; D=cp[2]@w[2:]; Y=dcal['strain_curves_obs']-dcal['strain_0'][:,None]; R=dcal['instant_rate_obs']
f=-np.expm1(-t); hold=cp[1]; resetB=hold[:,None]*(-np.expm1(-t[:,None]/a)@w[:2]); resetI=hold*(w[:2]@(1/a))
# Project the residual onto source-separated, rate-consistent shape perturbations.
def pred(p,kind):
    if kind=='exp': return B+D[:,None]*f,I+D
    if kind=='linear': return B+D[:,None]*t,I+D
    if kind=='reset': return resetB+D[:,None]*np.log1p(t),resetI+D
    if kind=='rational': return B+D[:,None]*t/(1+p[0]*t),I+D
    if kind=='rational-square': return B+D[:,None]*t/(1+p[0]*t*t),I+D
    if kind=='memory-difference':
        return B+p[0]*(resetB-B)+D[:,None]*f,I+p[0]*(resetI-I)+D
    if kind=='drift-interaction':
        drift=D*(1+p[0]*(hold-cp[0][:,-1])); return B+drift[:,None]*f,I+drift
models=('exp','linear','reset','rational','rational-square','memory-difference','drift-interaction')
for kind in models:
    scores=[]; params=[]
    for l in np.unique(lab):
        m=lab==l; train=~m & complete
        p=[]
        if kind not in models[:3]:
            def res(p):
                y,r=pred(p,kind); return np.r_[(y[train]-Y[train]).ravel(),(r[train]-R[train])]
            rational=kind.startswith('rational')
            p=least_squares(res,[.5 if rational else 0.],bounds=([0. if rational else -1.],[8. if rational else 1.]),loss='soft_l1',f_scale=.005).x
            params.append(float(p[0]))
        y,r=pred(p,kind); err=y[m]-Y[m]; scores.append([rm(err),max(rm(v) for v in err.T),rm((y-Y)[m&complete]),rm(r[m]-R[m])])
    print(kind,'worst lab curve/horizon/complete/rate',np.max(scores,0).round(6),'fold params',np.round(params,5))
y,r=pred([], 'exp'); errors=y-Y
print('diagnostic complete/missing curve RMSE',rm(errors[complete]),rm(errors[~complete]))
for l in np.unique(lab):
    m=(lab==l)&complete
    print('lab',l,'complete coupons',m.sum(),'signed curve/rate bias',errors[m].mean(),(r-R)[m].mean())
# Check whether the shared material fit is carried by one benchtop lab.
traincp=ds._coupon(tr,x[:n],b); Ty=tr['strain_curves_obs']-tr['strain_0'][:,None]; Tr=tr['instant_rate_obs']; tl=tr['lab_id']
for l in np.unique(tl):
    m=tl==l
    aa,ww=ds._fit(tuple(v[~m] for v in traincp),dt,t,Ty[~m],Tr[~m]); C,Q=ds._design(aa,tuple(v[m] for v in traincp),dt,t)
    e=C@ww-Ty[m]; er=Q@ww-Tr[m]
    print('benchtop held-out lab',l,'taus',aa.round(5),'curve median/p95 abs',np.quantile(abs(e),[.5,.95]).round(6),'rate RMSE',round(rm(er),6))
