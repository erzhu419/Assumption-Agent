import sys,pickle,json,numpy as np
from scipy.optimize import least_squares
sys.path[:0]=['/holdout_task','/tmp']
from _verified_upstream import calibrate_upstream
from chamber_prior import _features,_design,_response,_fit
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=it['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); d=calibrate_upstream(raw,pay); q=calibrate_upstream(dict(raw,test=it['campaign_diagnostics']),pay)['test']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; x,xt,xq=_features(tr,te,q); tau,b=_fit(tr,x,dt,t)
print('tau, theta',tau,b)
for p,xx in [(tr,x),(q,xq)]:
 r=_response(p)-_design(p,xx,dt,t,tau)@b; print('benchtop residual abs quantiles by time',np.round(np.quantile(abs(r),[.5,.8,.95],axis=0),4))
print('diagnostic count distinct past histories versus test',np.sum(np.min(np.linalg.norm((q['past_channel_raw'][:,None]-te['past_channel_raw'][None]).reshape(len(xq),len(xt),-1),axis=2),axis=1)>1e-8))
h=q['hold_channel_raw'].sum(-1); past=q['past_channel_raw'].sum(-1); drift=b[2]+xq@b[3:]; u=dt[:,None]/tau; z=past@(-np.expm1(-u)*np.exp(u-np.cumsum(u[::-1],axis=0)[::-1])); g=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)).T; time=np.r_[t,1.]; y=_response(q); hs=h[:,None,None]*b[:2][None,:,None]*g[None]; mem=z[:,:,None]*b[:2][None,:,None]*g[None]; ds=drift[:,None,None]*tau[None,:,None]*g[None]; steady=drift[:,None]*time
base=hs.sum(1)+steady
names=['hold0','hold1','memory0','memory1','drift0 conserve','drift1 conserve','steady','offset0','offset1']; F=np.column_stack([]) if False else np.concatenate([hs,mem,ds-steady[:,None],steady[:,None],np.broadcast_to(g,(len(h),2,len(time)))],axis=1).transpose(0,2,1)
def cv(label,base,F):
 def fit(m):
  a=F[m].reshape(-1,F.shape[-1]); yy=(y-base)[m].ravel(); return least_squares(lambda w:a@w-yy,np.linalg.lstsq(a,yy,rcond=None)[0],jac=lambda w:a,loss='cauchy',f_scale=.022).x
 w=fit(np.ones(len(y),bool)); e=y-base-F@w; cv=np.empty_like(y)
 for j in np.unique(q['lab_id']):
  m=q['lab_id']==j; cv[m]=y[m]-base[m]-F[m]@fit(~m)
 print(label,'coef',np.round(w,5),'fit med',np.round(np.median(abs(e),axis=0),4),'LOLO by lab (curve MAE,rate MAE)',[(j,*np.round([np.mean(abs(cv[q['lab_id']==j,:-1])),np.mean(abs(cv[q['lab_id']==j,-1]))],4)) for j in np.unique(q['lab_id'])]); return e
for f in ([.0,.0],[.45,.52],[.19431366,.52714783]):
 e=y-base-F[:,:,4:6]@f; print('fixed reset',f,'median curve/rate',np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])))
for ix in ([4,5],[2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5,6],list(range(9))): e=cv(str([names[i] for i in ix]),base,F[:,:,ix]);
print('broad residual sample',np.round(e[:5],3))
