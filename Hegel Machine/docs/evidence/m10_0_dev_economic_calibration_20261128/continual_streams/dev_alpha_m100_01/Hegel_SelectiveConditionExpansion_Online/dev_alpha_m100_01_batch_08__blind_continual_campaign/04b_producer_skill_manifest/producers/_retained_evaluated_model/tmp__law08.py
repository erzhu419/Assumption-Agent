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
U=h[:,None]*th[:2]; D=Y-U@K.T; S=R-U@(1/tau)
print('taus theta',tau,th); print('baseline errors',np.sqrt(np.mean((base-Y)**2)),np.sqrt(np.mean((rbase-R)**2))); print('holdout stress range',old._loads(b)[1].min(),old._loads(b)[1].max());print('singular after reset',np.linalg.svd(D,compute_uv=False))
print('drift, reset rate diff, curve residual t[0,3,7,11], slope constrained modal amplitudes (excess fast/slow)');
for i in np.argsort(dr):
 f=robust(B[:,:2]-t[:,None]/tau,D[i]-S[i]*t,.002); print(i,np.round(np.r_[dr[i],S[i],D[i,[0,3,7,11]],f],5))
print('missing features',np.argwhere(np.isnan(c['coupon_features_raw'])));print('strain scale',np.median(abs(Y)), 'rate scale',np.median(abs(R)))
print('fitted diagnostic reference shift',np.max(abs(xc-old._features(a,dict(c, **{k:np.concatenate((c[k],b[k]),axis=0) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}))[n:n+len(h)])))
