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
x = m._features(dg,{k:dg[k][:0] for k in features})[0]
s=m._prepare(dg,x,c); u,v=m._design(np.log(tau),s,dt,t); Y,R=obs(dg); D=s[2]@theta[2:]; M=np.any(np.isnan(dg['coupon_features_raw']),axis=1)
base=(u@theta).reshape(Y.shape)-D[:,None]*t; memory=v[:,:2]@theta[:2]; DR=R-memory
Q=tau*-np.expm1(-t[:,None]/tau)-t[:,None]
print('missing rows and drift offsets',np.column_stack((np.where(M)[0],(DR-D)[M])))
for proxy,d,usable in [('features_complete',D,~M),('rate_all',DR,np.ones(len(D),bool)),('rate_for_missing',np.where(M,DR,D),np.ones(len(D),bool))]:
 A=d[:,None,None]*Q; target=Y-base-d[:,None]*t
 def fit(ok):
  a=A[ok].reshape(-1,2); b=target[ok].ravel(); z=np.linalg.lstsq(a,b,rcond=None)[0]; return least_squares(lambda z:(a@z-b)/.003,z,loss='soft_l1').x
 z=fit(usable); E=np.zeros(Y.shape)
 for lab in np.unique(dg['lab_id']):
  g=dg['lab_id']==lab; E[g]=A[g]@fit(usable&~g)-target[g]
 print(proxy,'coeff',z,'kernel',Q[[0,3,-1]]@z,'leave-lab-out peak',[(int(lab),np.max(abs(E[dg['lab_id']==lab]),axis=0)[[0,3,-1]].round(5).tolist()) for lab in np.unique(dg['lab_id'])],'complete p90',np.quantile(abs(E[~M]),.9),'missing p90',np.quantile(abs(E[M]),.9))
