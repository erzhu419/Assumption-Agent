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
U=-np.expm1(-t); R=q['strain_curves_obs']-q['strain_0'][:,None]-(gap*th[:2])@(-np.expm1(-t[:,None]/tau)).T
B=R@U/(U@U); raw=q['coupon_features_raw']; msk=np.isnan(raw); implied=xlocal.copy(); row,col=np.where(msk); implied[row,col]+=(B-th[2]-xlocal@th[3:])[row]/th[3+col]; print('diagnostic missing feature: column, local imputation, implied value:',np.c_[col,xlocal[row,col],implied[row,col]].round(5).tolist()); print('observed local feature means',np.nanmean(np.where(msk,np.nan,xlocal),axis=0).round(5).tolist()); print('test joint minus own RMS drift',np.sqrt(np.mean(((m._features(tr,te)[1]-m._features(te,te)[1])@th[3:])**2)))
