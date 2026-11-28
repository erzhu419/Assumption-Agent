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
from itertools import product
ax=cal['single_channel_sweeps']; ar=np.asarray(ax['cmd_stress']); al=np.asarray(ax['true_stress']); labs=np.asarray(ax['lab_id']); print('axes',[np.linalg.lstsq(np.column_stack([np.ones(sum(labs==j)),al[labs==j]]),ar[labs==j],rcond=None)[0] for j in range(4)])
print('missing diag',np.argwhere(np.isnan(dg['coupon_features_raw'])).tolist())
for label,x in [('diag',m._features(dg,{k:dg[k][:0] for k in features})[0]),('joint',xd)]:
 s=m._prepare(dg,x,c); u,v=m._design(np.log(tau),s,dt,t); Y,R=obs(dg); D=s[2]@theta[2:]; C=u@theta; S=v@theta; E=np.column_stack([Y-C,R-S]); G=-np.expm1(-t[:,None]/tau); Z=s[1][:,None]-v[:,:2]*tau; T=np.r_[t,1.]; B=np.vstack([G,1/tau]); Q=B*tau-T[:,None];
 def outer(a,b): return a[:,None,None]*b[None,:,:]
 models={'conserved':outer(D,Q),'full_drift':outer(D,np.column_stack([B,T])),'conserved_plus_drift_offset':np.concatenate([outer(D,Q),outer(np.ones_like(D),Q)],axis=2),'conserved_plus_memory':np.concatenate([outer(D,Q),Z[:,None,:]*B[None,:,:]],axis=2),'conserved_plus_steady':np.concatenate([outer(D,Q),outer(D,T[:,None])],axis=2)}
 for name,A in models.items():
  def fit(mask):
   M=A[mask].reshape(-1,A.shape[-1]); y=E[mask].ravel(); b=np.linalg.lstsq(M,y,rcond=None)[0]; return least_squares(lambda v:(M@v-y)/0.004,b,loss='soft_l1').x
  mask=np.ones(len(D),bool); b=fit(mask); P=A@b-E; OO=np.zeros_like(E); OL=np.zeros_like(E)
  for i in range(len(D)): OO[i]=A[i]@fit(np.arange(len(D))!=i)-E[i]
  for lab in np.unique(dg['lab_id']):
   g=dg['lab_id']==lab; OL[g]=A[g]@fit(~g)-E[g]
  def met(M): return np.round(np.quantile(abs(M[:,:-1]),[.5,.9,1]),5),np.round(np.quantile(abs(M[:,-1]),[.5,.9,1]),5)
  print(label,name,np.round(b,5),'fit',met(P),'LOCO',met(OO),'LOLab',met(OL))
  if label=='diag' and name=='conserved': print('coupon residuals',[tuple(np.round([D[i],np.max(abs(OO[i,:-1])),OO[i,-1]],4)) for i in np.argsort(D)])
