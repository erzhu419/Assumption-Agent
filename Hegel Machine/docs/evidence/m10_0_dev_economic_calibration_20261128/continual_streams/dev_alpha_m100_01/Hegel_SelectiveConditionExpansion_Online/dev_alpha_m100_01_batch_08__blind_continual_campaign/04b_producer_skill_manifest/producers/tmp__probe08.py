import sys,pickle,numpy as np
sys.path[:0]=['/holdout_task','/tmp/hegel_continual_archive']
import _verified_upstream as up
import dev_alpha_m100_01_batch_02_solver as old
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=up.calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
a,b,c=[d[k] for k in ('train','test','chamber_verification_trials')]; t,dt=d['eval_times'],d['past_dt']; n=len(a['lab_id']); x=old._features(a,b); xc=old._features(a,c)[n:]; p,h=old._loads(c)
tau,th=old._fit(*old._loads(a),x[:n],dt,t,a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']);print('tau theta',tau,th)
Y=c['strain_curves_obs']-c['strain_0'][:,None]; R=c['instant_rate_obs']; K=-np.expm1(-t[:,None]/tau); B=np.c_[K,t]; s=np.r_[1/tau,1.]; cc,rr=old._design(tau,p,h,xc,dt,t); base=cc@th; rbase=rr@th; dr=rr[:,2:]@th[2:]; g=rr[:,:2]*tau; print('baseline curve,rate RMSE',np.sqrt(np.mean((base-Y)**2)),np.sqrt(np.mean((rbase-R)**2)));print('diag missing',np.isnan(c['coupon_features_raw']).sum(0));print('rate comparison: observed,base,reset,drift',np.round(np.c_[R,rbase,h*(th[:2]/tau).sum()+dr,dr],4)); print('curve singular values',np.linalg.svd(Y,compute_uv=False))
def robust(A,y,scale):
 return least_squares(lambda w:A@w-y,np.linalg.lstsq(A,y,rcond=None)[0],loss='cauchy',f_scale=scale).x
print('entry curve/rate diagnostics')
for i in range(len(Y)):
 z=robust(np.r_[B,s[None]*.15],np.r_[Y[i],R[i]*.15],.0005); print(i,np.round(np.r_[h[i],g[i],dr[i],z,(z-np.r_[g[i]*th[:2],dr[i]]),np.median(abs(Y[i]-B@z)),R[i]-s@z],5))
for name,F in [('base',np.c_[np.ones(len(h)),g,h,dr]),('history',np.c_[np.ones(len(h)),p,h,dr]),('full',np.c_[np.ones(len(h)),p,h,xc])]:
 A=np.einsum('nf,kb->nkfb',F,B).reshape(Y.size,-1); Ar=np.einsum('nf,b->nfb',F,s).reshape(len(Y),-1); z=robust(np.r_[A,.15*Ar],np.r_[Y.ravel(),.15*R],.0005); e=(A@z).reshape(Y.shape)-Y; er=Ar@z-R; print(name,'error quantiles',np.quantile(abs(e),[.5,.9,.95,.99,1]),np.quantile(abs(er),[.5,.9,1]),'coef',np.round(z.reshape(-1,3),5))
