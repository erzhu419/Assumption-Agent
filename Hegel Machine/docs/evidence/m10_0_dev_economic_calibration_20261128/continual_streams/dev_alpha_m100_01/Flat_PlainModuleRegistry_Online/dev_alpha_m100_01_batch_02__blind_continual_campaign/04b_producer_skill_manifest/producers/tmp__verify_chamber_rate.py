import sys, pickle
import numpy as np
sys.path.insert(0, '/holdout_task')
import downstream_solver as s
from batch_solver import solve_batch
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    item=pickle.load(f)[0]
raw=item['dataset']; cal=calibrate_upstream(raw); tr,te=cal['train'],cal['test']; d=calibrate_upstream(dict(raw,test=item['campaign_diagnostics']))['test']
fit=solve_batch(raw); ta,th=fit['estimated_taus'],fit['hereditary_weights']; dt,t=cal['past_dt'],cal['eval_times']; B=s._basis(ta,t)
lab=d['lab_id']; x=d['coupon_features_raw'].copy(); missing=np.isnan(x).any(1)
for j in np.unique(lab):
    m=lab==j; r=d['ref_coupons'][m]; x[m]=(x[m]-np.median(r.mean(0)))/np.median(r.std(0))
x=np.where(np.isfinite(x),x,np.nanmean(x,axis=0)); score=th[2]+x@th[3:]
base=s._predict(ta,th,*s._loads(d),x,dt,t,np.zeros((2,3))); e=s._observations(d)-base
S=np.c_[np.ones(len(x)),score]; C=np.vstack((np.eye(2),-1/ta)); print('shipped coefficient rate perturbations',s._CHAMBER@B[-1]); print('observed baseline rate residual by lab',[(int(k),round(e[lab==k,-1].mean(),5),round(np.sqrt(np.mean(e[lab==k,-1]**2)),5)) for k in np.unique(lab)])
for name, basis, sc in [('affine',B,S),('affine rate-null',B@C,S),('proportional rate-null',B@C,S[:,1:])]:
    A=(sc[:,None,:,None]*basis[None,:,None,:]).reshape(len(x),len(B),-1); p=A.shape[-1]
    def learn(m): return np.linalg.lstsq(A[m].reshape(-1,p),e[m].ravel(),rcond=None)[0]
    def rmse(res):
        q=np.sqrt(np.mean(res**2,axis=0)); return np.r_[q[:-1].max(),q[-1]]
    coef=learn(np.ones(len(x),bool)); print(name,'coefficients',coef.round(6),'worst horizon/rate',rmse(e-A@coef).round(6))
    for label,g in [('lab',lab),('stratum',d['stratum']),('missing',missing),('coupon',np.arange(len(x)))]:
        re=[]
        for k in np.unique(g):
            m=g==k; v=learn(~m); re.append(rmse(e[m]-A[m]@v))
        print(' ',label,np.max(re,axis=0).round(6))
xtr=s._features(tr,te)[:len(tr['lab_id'])]; past,h=s._loads(tr); obs=s._observations(tr)
for j in np.unique(tr['lab_id']):
    m=tr['lab_id']==j; a,b=s._fit(past[~m],h[~m],xtr[~m],dt,t,obs[~m]); err=obs[m]-s._predict(a,b,past[m],h[m],xtr[m],dt,t,np.zeros((2,3))); print('material held-out lab',j,'max relative tau delta',np.max(abs(a/ta-1)),'max theta delta',np.max(abs(b-th)), 'median absolute strain/rate',np.round([np.median(abs(err[:,:-1])),np.median(abs(err[:,-1]))],5)); assert np.max(abs(a/ta-1))<.1 and np.max(abs(b-th))<.05
