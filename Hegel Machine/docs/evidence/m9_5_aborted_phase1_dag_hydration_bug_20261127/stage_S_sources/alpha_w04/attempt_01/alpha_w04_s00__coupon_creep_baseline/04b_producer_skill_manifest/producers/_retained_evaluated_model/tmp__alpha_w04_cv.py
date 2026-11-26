import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_w04_s00_solver as m
D=pickle.load(open('/holdout_task/public_alpha_w04_s00.pkl','rb'))[0]['dataset']; tr=D['train']; t=D['eval_times']; dt=D['past_dt']
cal=m._calibration(D['calibration_load_cells']); g,b=cal[tr['lab_id']].T
p=tr['past_stress_cmd']*g[:,None]+b[:,None]; h=tr['stress_hold_cmd']*g+b
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
lab=np.r_[tr['lab_id'],D['test']['lab_id']]; s=np.r_[tr['stratum'],D['test']['stratum']]; raw=np.vstack([tr['coupon_features_raw'],D['test']['coupon_features_raw']]); ref=np.vstack([tr['ref_coupons'],D['test']['ref_coupons']]); z=raw.copy()
for c in range(4):
 a=ref[lab==c]; z[lab==c]=(raw[lab==c]-np.median(np.nanmean(a,axis=0)))/np.median(np.nanstd(a,axis=0))
keys=list(np.unique(np.c_[lab,s],axis=0)); means=np.array([np.nanmean(z[(lab==c)&(s==q)],axis=0) for c,q in keys]); balanced=np.nanmean(means,axis=0); print('balanced fill',balanced,'cell extrema',means.min(axis=0),means.max(axis=0))
xlocal=m._features(tr,D['test'])[0]; xglob=np.where(np.isfinite(z),z,balanced)[:len(y)]; miss=~np.isfinite(tr['coupon_features_raw']).all(axis=1)
for label,U in [('cell',xlocal),('balanced',xglob)]:
 for axis in [tr['lab_id'],tr['lab_id']//2]:
  for id in np.unique(axis):
   k=axis!=id
   tau,w=m._fit(p[k],h[k],U[k],dt,t,y[k]); pred=m._design(p[~k],h[~k],U[~k],dt,t,tau)@w; e=abs(pred-y[~k]);
   print(label,'axis',len(np.unique(axis)),int(id),'tau',np.round(tau,4),'curve median/p90 and rate median/p90',np.round(np.r_[np.quantile(e[:,:-1],[.5,.9]),np.quantile(e[:,-1],[.5,.9])],5),'horizon worst med',round(np.median(e[:,:-1],axis=0).max(),5),'missing rows',miss[~k].sum(),'missing med',np.round(np.median(e[miss[~k]],axis=0)[[0,-1]],5))
