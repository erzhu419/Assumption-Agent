import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import downstream_solver as s
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
a,b,c=(d[k] for k in ('train','test','chamber_verification_trials'))
t,dt=d['eval_times'],d['past_dt'];n=len(a['lab_id']);x=s._features(a,b)[:n];xc=s._features(c,c)[:len(c['lab_id'])]
y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; pp,hh=s._loads(a)
u,v=s._fit(x,pp,hh,dt,t,y,r)
h=s._loads(c)[1]; cy=c['strain_curves_obs']; cr=c['instant_rate_obs']; ok=np.isfinite(c['coupon_features_raw']).all(axis=1)
def metrics(ry,rr): return np.max(np.sqrt(np.mean(ry**2,axis=0))),np.sqrt(np.mean(rr**2))
for j in np.unique(a['lab_id']):
 test=a['lab_id']==j; train=~test
 q,p=s._fit(x[train],pp[train],hh[train],dt,t,y[train],r[train]);A,B=s._design(q,x[test],pp[test],hh[test],dt,t)
 print('benchtop held-out lab',j,'taus',q,'worst-horizon and rate RMSE',metrics(y[test]-A@p,r[test]-B@p))
py,pr=s._predict_chamber(u,v,xc,h,t,c['strain_0']);
for k,m in [('complete',ok),('missing',~ok)]: print('frozen candidate diagnostic',k,'count',m.sum(),'worst-horizon and rate RMSE',metrics((cy-py)[m],(cr-pr)[m]))
R=-np.expm1(-t[:,None]/u);D=v[2]+xc@v[3:];E=cy-c['strain_0'][:,None]-(h[:,None]*v[:2])@R.T-D[:,None]*t
M=D[:,None,None]*(R*u-t[:,None])[None,:,:]
for mask in (ok,~ok,np.ones(len(ok),bool)):
 f=np.linalg.lstsq(M[mask].reshape(-1,2),E[mask].ravel(),rcond=None)[0]; print('drift fractions',mask.sum(),f)
yp,rp=s._predict_chamber(u,v,xc,h,np.array([0,1e-6]),c['strain_0']);print('derivative max mismatch',np.max(abs((yp[:,1]-yp[:,0])/1e-6-rp)));assert np.max(abs((yp[:,1]-yp[:,0])/1e-6-rp))<1e-5
