import pickle, sys
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m99_c04_s00_solver as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=item['dataset']; q=item['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; s=m._stress_map(d['calibration_load_cells'])
x,_=m._features(tr,te); tau,th=m._fit(tr,x,s,dt,t)
_,xq=m._features(tr,q); xlocal,_=m._features(q,q)
p=s[0]+q['past_channel_raw']@s[1:]; h=s[0]+q['hold_channel_raw']@s[1:]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; z=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-age[:,None]/tau)); gap=h[:,None]-z
F={'hold':h,'z0':z[:,0],'z1':z[:,1],'g0':gap[:,0],'g1':gap[:,1],'g0*g1':gap.prod(1),'g0-g1':gap[:,0]-gap[:,1],'lab':q['lab_id'],'stratum':q['stratum']}
for i in range(p.shape[1]): F[f'p{i}']=p[:,i]; F[f'h-p{i}']=h-p[:,i]
for i in range(4): F[f'x{i}']=xq[:,i]; F[f'rawx{i}']=np.nan_to_num(q['coupon_features_raw'][:,i])
from scipy.optimize import minimize_scalar
keys=('coupon_features_raw','ref_coupons','lab_id','stratum')
def merge(*groups): return {k:np.concatenate([g[k] for g in groups]) for k in keys}
_,xd=m._features(te,q)
_,xtd=m._features(merge(tr,te),q)
_,xt=m._features(tr,te)
tm=-np.expm1(-t[:,None]/tau); intrinsic=(gap*th[:2])@tm.T; ir=(gap*th[:2]/tau).sum(1)
Y=q['strain_curves_obs']-q['strain_0'][:,None]-intrinsic; yr=q['instant_rate_obs']-ir
U=-np.expm1(-t)
a=Y@U/(U@U)
print('intrinsic tau,theta',tau,th)
print('pure temporal shape residual std',np.std(Y-a[:,None]*U),'projected rate rmse',np.sqrt(np.mean((a-yr)**2)))
miss=np.isnan(q['coupon_features_raw']).any(1)
for name,g in [('tr',tr),('te',te),('q',q)]:
 print(name,'missing by feature',np.isnan(g['coupon_features_raw']).sum(0).tolist())
 for lab in np.unique(g['lab_id']):
  rr=g['ref_coupons'][g['lab_id']==lab]; print('ref',int(lab),len(rr),round(np.median(rr.mean(0)),4),round(np.median(rr.std(0)),4))
for name,xx in [('q alone',xlocal),('train+q',xq),('test+q',xd),('train+test+q',xtd)]:
 r=th[2]+xx@th[3:]; E=Y-r[:,None]*U; er=yr-r
 print(name,'curve/rate RMSE',*[round(np.sqrt(np.mean(v**2)),6) for v in (E,er)],'complete',*[round(np.sqrt(np.mean(v[~miss]**2)),6) for v in (E,er)],'missing',*[round(np.sqrt(np.mean(v[miss]**2)),6) for v in (E,er)])
 if name=='q alone':
  print('missing coupon residuals',np.c_[a-r,yr-r,np.isnan(q['coupon_features_raw'])][miss].round(4).tolist()); print('largest complete',np.c_[a-r,yr-r,q['lab_id']][~miss][np.argsort(np.abs(a-r)[~miss])[-3:]].round(4).tolist())
 def loss(logtau,idx):
  tt=np.exp(logtau); u=-tt*np.expm1(-t/tt)
  return np.mean((Y[idx]-r[idx,None]*u)**2)
 print('best drift tau all/complete',*[round(np.exp(minimize_scalar(loss,args=(idx,),bounds=(-2,2),method='bounded').x),6) for idx in (np.ones(len(r),dtype=bool),~miss)])
