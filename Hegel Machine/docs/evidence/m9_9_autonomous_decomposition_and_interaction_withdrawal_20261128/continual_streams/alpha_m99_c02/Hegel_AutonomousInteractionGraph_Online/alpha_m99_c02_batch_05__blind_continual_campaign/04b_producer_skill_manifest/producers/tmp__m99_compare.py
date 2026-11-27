import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import *
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
ds = calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
tr,te,d = (ds[k] for k in ('train','test','chamber_verification_trials'))
dt,t=ds['past_dt'],ds['eval_times']
x,xt,xd=features(tr,te,d)
taus,theta=material(tr,x,dt,t)
print('dt,tau,theta',dt,taus,theta)
Y=np.c_[d['instant_rate_obs'],d['strain_curves_obs']-d['strain_0'][:,None]]
P,H=loads(d); L=d['lab_id']; V=np.r_[1,t]; A=np.c_[np.vstack((1/taus,-np.expm1(-t[:,None]/taus))),V]; amp=Y@np.linalg.pinv(A.T)
xopts={'aligned':xd,'diagnostics_only':features(d,d)[0],'train_plus_diagnostics':features(tr,d)[1]}
Z=H[:,None]-gap_for(taus,P,H,dt)
for name,xx in xopts.items():
 D=theta[2]+xx@theta[3:]; print(name,'feature diffs',np.max(abs(xx-xd),axis=0).round(4));
 for name2,inputs in [('reset',[np.c_[H,D],np.c_[H,D],D[:,None]]),('hist',[np.c_[H,D],np.c_[H,D,Z],D[:,None]]),('slow_quad',[np.c_[H,D],np.c_[H,D,Z,H*D,D**2,H**2,Z*D[:,None],Z*H[:,None]],D[:,None]])]:
  B=np.concatenate([a[None,:,None]*J[:,None,:] for a,J in zip(A.T,inputs)],axis=-1)
  def fit(ix):
   b=np.linalg.lstsq(B[ix].reshape(-1,B.shape[-1]),Y[ix].ravel(),rcond=None)[0]; return b,B@b-Y
  rms=lambda e:np.array([np.sqrt(np.mean(e[:,0]**2)),np.sqrt(np.mean(e[:,1:]**2))]); b,e=fit(L==L)
  print(name2,b.round(4),rms(e).round(5),'LOLO worst',np.max([rms(fit(L!=l)[1][L==l]) for l in np.unique(L)],axis=0).round(5))
 if name=='aligned': print('slow residual by lab; hold, slow state, D, missing count:',np.c_[L,H,Z[:,1],D,np.isnan(d['coupon_features_raw']).sum(1),amp[:,1]-(.40*H+2.38*D+.036*Z[:,0]+.27*Z[:,1])].round(4))
print('telemetry',np.round(te['aux_chamber_telemetry'][:8],4)); print('diagnostic keys',list(d));print('timing',t)
