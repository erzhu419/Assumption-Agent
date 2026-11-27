import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
p,h,x,y0=s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
l=d['train']['lab_id']; n=len(l)
y=d['train']['strain_curves_obs']-y0[:n,None]; r=d['train']['instant_rate_obs']
e=np.empty_like(y); er=np.empty_like(r)
for j in np.unique(l):
    tr=l!=j; va=~tr
    z,w=s._fit(p[:n][tr],h[:n][tr],x[:n][tr],dt,t,a[:n][tr],y[tr],r[tr],True)
    cb,rb,*_=s._basis(z,p[:n][va],h[:n][va],x[:n][va],dt,t,a[:n][va],True)
    e[va]=cb@w-y[va]; er[va]=rb@w-r[va]
print('Curve residual absolute quantiles by time:',np.quantile(abs(e),[.5,.8,.9,.95,.99],axis=0).round(4))
print('Rate residual absolute quantiles:',np.quantile(abs(er),[.5,.8,.9,.95,.99]).round(4))
print('Curve outliers > 0.15 per coupon:',np.unique((abs(e)>.15).sum(axis=1),return_counts=True))
print('Curve outliers > 0.15 by time:',(abs(e)>.15).sum(axis=0))
print('Lab/damage slices: lab, active, count, worst time median absolute curve error, median absolute rate error')
for j in np.unique(l):
    for active in [False,True]:
        m=(l==j)&((a[:n]>0)==active)
        print(j,active,m.sum(),np.median(abs(e[m]),axis=0).max(),np.median(abs(er[m])))
for plateau in (False,True):
    logtau=np.log([.5,3.8]); taus=np.exp(logtau)
    act=a if plateau else np.zeros_like(a)
    z=np.zeros((len(p),2))
    for stress,ds in zip(p.T,dt):
        z=np.exp(-ds/taus)*z+(-np.expm1(-ds/taus))*stress[:,None]
    cb,rb,*_=s._basis(logtau,p,h,x,dt,t,act,plateau)
    phi=np.log1p(t) if plateau else t
    step=-np.expm1(-t[:,None]/taus)
    for m in range(2):
        np.testing.assert_allclose(cb[:,:,m],(h-z[:,m])[:,None]*step[:,m],atol=1e-13)
    np.testing.assert_allclose(cb[:,:,2:7],x[:,None,:]*phi[None,:,None],atol=1e-13)
    np.testing.assert_allclose(cb[:,:,7],(act*h)[:,None]*step[:,0],atol=1e-13)
    np.testing.assert_allclose(cb[:,:,8],(act*z[:,0])[:,None]*phi,atol=1e-13)
    cb_small,*_=s._basis(logtau,p,h,x,dt,np.array([1e-7]),act,plateau)
    np.testing.assert_allclose(cb_small[:,0,:]/1e-7,rb,rtol=2e-7,atol=1e-12)
print('Recurrence and initial-rate checks passed for both terminal bases.')
