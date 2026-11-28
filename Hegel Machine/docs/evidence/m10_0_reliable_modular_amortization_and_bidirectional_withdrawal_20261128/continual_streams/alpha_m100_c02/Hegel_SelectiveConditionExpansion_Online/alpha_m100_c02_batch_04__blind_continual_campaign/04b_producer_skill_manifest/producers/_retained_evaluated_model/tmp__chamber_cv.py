import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import downstream_solver as s
from _verified_upstream import calibrate_upstream
q=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=calibrate_upstream(dict(q['dataset'],chamber_verification_trials=q['campaign_diagnostics'])); z=d['chamber_verification_trials']; dt,t,_,tau,th=s._intrinsic(d)
p,h=s._stress(z); y=z['strain_curves_obs']-z['strain_0'][:,None]; r=z['instant_rate_obs']; rise=-np.expm1(-t[:,None]/tau); base=h[:,None]*(rise@th[:2]); drift_from_rate=r-h*np.sum(th[:2]/tau)
tr=d['train']; nt=len(tr['lab_id']); nz=len(h)
x_joint=s._features(tr,z)[nt:]; x_local=s._features(z,z)[:nz]; lab=z['lab_id']; miss=np.any(np.isnan(z['coupon_features_raw']),axis=1)
G=rise*tau-t[:,None]
print('missing diagnostic by lab',[(int(l),int(miss[lab==l].sum())) for l in np.unique(lab)])
print('rate errors from features:')
for tag,x in [('pooled',x_joint),('diagnostic only',x_local)]:
 e=abs(x@th[2:]-drift_from_rate); print(tag,np.quantile(e,[.5,.9,1]),'complete',np.quantile(e[~miss],[.5,.9,1]))
def fit(ids, mode):
 a=drift_from_rate[ids]; H=h[ids]; Y=y[ids]-base[ids]-a[:,None]*t
 A=a[:,None,None]*G[None,:,:]
 if mode=='hold': A=np.concatenate([A,H[:,None,None]*G],axis=2)
 if mode=='offset': A=np.concatenate([A,np.broadcast_to(G,A.shape)],axis=2)
 if mode=='archived': return np.array([.42947,.53764]),A[:,:,:2],Y
 f=np.linalg.lstsq(A.reshape(-1,A.shape[2]),Y.ravel(),rcond=None)[0]
 return f,A,Y
for mode in ('archived','drift','hold','offset'):
 co=[]; print('\n',mode,'full',fit(np.arange(nz),mode)[0]);
 for l in np.unique(lab):
  ids=np.flatnonzero(lab==l); f,_,_=fit(np.flatnonzero(lab!=l),mode); _,A,Y=fit(ids,mode)
  curve=base[ids]+(x_local[ids]@th[2:])[:,None]*(t+G@f[:2]);
  if mode in ('hold','offset'): curve+= (h[ids,None] if mode=='hold' else 1)*(G@f[2:]);
  print(int(l),'f',f.round(4),'boundary/feature based curve rmse',np.sqrt(np.mean((A@f-Y)**2)),np.sqrt(np.mean((curve-y[ids])**2)), 'long horizon',np.sqrt(np.mean((curve[:,-1]-y[ids,-1])**2)))
for x in (x_local,x_joint):
 a=x@th[2:]; print('feature implied coefficient (all, complete)')
 for ids in (np.arange(nz),np.flatnonzero(~miss)):
  A=a[ids,None,None]*G; Y=y[ids]-base[ids]-a[ids,None]*t; print(np.linalg.lstsq(A.reshape(-1,2),Y.ravel(),rcond=None)[0])
