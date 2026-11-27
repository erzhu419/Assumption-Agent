import pickle,numpy as np
from rheo_robust_cv import m,robust
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; dt,t,(b,_)=m._prepare(d); s=d['train']; labs=s['lab_id']; on=b[-1]>0; missing=np.isnan(s['coupon_features_raw']).any(axis=1); y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; original=m._basis
rm=lambda x: np.sqrt(np.mean(x*x))
for mix in [0.,.5,1.]:
 def basis(tau,dt,t,b,log=True):
  C,A=original(tau,dt,t,b,True); idx=b[-1]==0
  terminal=(1-mix)*t+mix*np.log1p(t)
  C[idx,:,2:7]=b[2][idx,None,:]*terminal[None,:,None]
  return C,A
 m._basis=basis
 ec=np.empty_like(y); er=np.empty_like(r); params=[]
 for lab in np.unique(labs):
  te=labs==lab; tr=~te
  tau,w=robust(dt,t,tuple(x[tr] for x in b),y[tr],r[tr],.025,2); params.append(tau)
  C,A=basis(tau,dt,t,tuple(x[te] for x in b)); ec[te]=C@w-y[te]; er[te]=A@w-r[te]
 print('inactive log fraction',mix,'taus',np.round(params,4).tolist(),'curve pooled raw RMSE',rm(ec))
 for active in [False,True]:
  group=[(labs==lab)&(on==active) for lab in np.unique(labs)]
  print('active',active,'worst lab median / p90 / rate RMSE',max(np.median(abs(ec[z])) for z in group),max(np.quantile(abs(ec[z]),.9) for z in group),max(rm(er[z]) for z in group),'complete panel worst terminal median',max(np.median(abs(ec[z&~missing,-1])) for z in group))
 tau,w=robust(dt,t,b,y,r,.025,2); print('full',tau,w)
