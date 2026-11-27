import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
from scipy.optimize import least_squares
sp=m._specimen(c,m._features(tr,c)[1],bm); p,h,_=sp
Y=np.column_stack([c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]); B=np.vstack([np.column_stack([-np.expm1(-t[:,None]/tau),t]),np.r_[1/tau,1]])
amp=np.linalg.lstsq(B[:,:2],Y.T,rcond=None)[0].T; print('no-drift basis RMSE per output',np.sqrt(np.mean((Y-amp@B[:,:2].T)**2,axis=0)))
print('missing diag',np.where(~np.isfinite(c['coupon_features_raw']))); print('lab sweeps');
for l in np.unique(c['lab_id']):
 s=ca['single_channel_sweeps']; j=s['lab_id']==l; print(l,np.polyfit(s['cmd_stress'][j],s['true_stress'][j],1))
allg={k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
for name,a,b in [('diag-only',c,c),('train+diag',tr,c),('test+diag',te,c),('all+diag',allg,c)]:
 xx=m._features(a,b)[1]; drift=th[2]+xx@th[3:]; print('\n',name); print(np.round(np.column_stack([h,drift,amp,amp[:,0]-th[0]*h,amp[:,1]-th[1]*h]),4));
 for label,F in [('h',h[:,None]),('h+d',np.c_[h,drift]),('h+d+hd',np.c_[h,drift,h*drift]),('h+d+state',np.c_[h,drift,h[:,None]-m._branches(sp,dt,t,tau)[1]*tau]),('h+x',np.c_[h,xx]),('h+hx',np.c_[h,h[:,None]*xx]),('h+x+lab',np.c_[h,xx,np.eye(4)[c['lab_id']][:,1:]])]:
  A=np.c_[np.ones(len(h)),F]; w=np.linalg.lstsq(A,amp,rcond=None)[0]; err=amp-A@w; loo=err/(1-np.sum(A*np.linalg.pinv(A).T,axis=1))[:,None]; print(label,'coef',np.round(w.T,4),'loo rmse',np.round(np.sqrt(np.mean(loo**2,axis=0)),4))
