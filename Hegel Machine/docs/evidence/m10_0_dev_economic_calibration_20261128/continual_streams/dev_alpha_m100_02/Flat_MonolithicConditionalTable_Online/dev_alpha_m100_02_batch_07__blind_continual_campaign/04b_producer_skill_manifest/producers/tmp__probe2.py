import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_alpha_m100_02_batch_06_solver as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr,te=ds['train'],ds['test']; dg=item['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']
xtr,xte=m._features(tr,te)
features=('coupon_features_raw','ref_coupons','lab_id','stratum')
pool={k:np.concatenate([tr[k],te[k]]) for k in features}; _,xd=m._features(pool,dg)
cal=ds['calibration_load_cells']; c=m._bridge(cal)
a=cal['true_channel_loads']; y=cal['bridge_raw']; keep=np.max(np.abs(y),axis=1)<5
print('cal',c,'calibration load absolute residual quantiles',np.quantile(np.abs(a[keep]-m._loads(y[keep],c)),[.5,.9,1]))
def obs(p): return p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']
s=m._prepare(tr,xtr,c); tau,theta=m._fit(s,dt,t,*obs(tr)); print('tau,theta:',tau,theta)
np.set_printoptions(precision=4,suppress=True)
s=m._prepare(dg,xd,c); u,v=m._design(np.log(tau),s,dt,t); Y,R=obs(dg); D=s[2]@theta[2:]; C=u@theta; S=v@theta; H=Y-C
G=-np.expm1(-t[:,None]/tau); B=np.column_stack((G,t)); A=np.linalg.lstsq(np.vstack([B,np.r_[1/tau,1.]]),np.column_stack([H,R-S]).T,rcond=None)[0].T
print('lab stratum drift delta rate amplitude fast slow steady delta(t=.25,1.27,4)');
for i in np.argsort(D): print(dg['lab_id'][i],dg['stratum'][i],*[round(float(j),4) for j in np.r_[D[i],R[i]-S[i],A[i],H[i,[0,3,-1]]]])
print('rate discrepancy by feature-normalization reference:')
for label,x in [('joint',xd),('diag',m._features(dg,{k:dg[k][:0] for k in features})[0]),('trainDiag',m._features(tr,dg)[1])]:
 ss=m._prepare(dg,x,c); uu,vv=m._design(np.log(tau),ss,dt,t); print(label,np.quantile(abs(R-vv@theta),[.5,.9,1]),np.corrcoef(R-vv@theta,x.T)[0,1:]);
 print('lab slopes',[np.linalg.lstsq(np.column_stack([np.ones(l.sum()),x[l]]),(R-vv[:,:2]@theta[:2])[l],rcond=None)[0].round(3) for lab in np.unique(dg['lab_id']) if (l:=dg['lab_id']==lab).sum()>=5])
cal=ds['calibration_load_cells']; a=cal['true_channel_loads']; y=cal['bridge_raw']; ax=cal['single_channel_sweeps']; print('cal outliers',np.sum(~keep),np.sum(abs(ax['cmd_stress'])>=5),np.min(y),np.max(y));
for r in __import__('json').load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 z=np.asarray(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],z.mean(0).round(3),z[-1].round(3))
