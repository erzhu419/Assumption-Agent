import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];tr=d['train'];te=d['test'];c=i['campaign_diagnostics'];dt=d['past_dt'];t=d['eval_times'];cal=d['calibration_load_cells'];raw=cal['bridge_raw'];y=cal['true_channel_loads'];b=m._bridge(cal)
print('bridge',b,'raw range',np.min(raw,0),np.max(raw,0)); print('sweeps',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in cal['single_channel_sweeps'].items()])
from itertools import combinations_with_replacement
for degree in (1,2,3):
 A=np.column_stack([np.ones(len(raw))]+[np.prod(raw[:,j],axis=1) for j in range(1,degree+1) for j in combinations_with_replacement(range(4),j)])
 pred=np.zeros_like(y)
 for q in np.array_split(np.random.default_rng(5).permutation(len(y)),5):
  z=np.setdiff1d(np.arange(len(y)),q);pred[q]=A[q]@np.linalg.lstsq(A[z],y[z],rcond=None)[0]
 e=pred-y;print('bridge OOF degree',degree,'channel rmse',np.sqrt(np.mean(e**2,0)),'stress rmse',np.sqrt(np.mean(e.sum(1)**2)))
x,xt=m._features(tr,te);p,h=m._loads(tr,b);pc,hc=m._loads(c,b);tau,beta=m._fit(x,p,h,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,beta)
_,xc=m._features(dict((k,np.concatenate([tr[k],te[k]],axis=0)) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')),c)
import importlib.util
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); q=importlib.util.module_from_spec(s); s.loader.exec_module(q)
X=m._features(c,c)[1]; missing=~np.isfinite(c['coupon_features_raw']); complete=~missing.any(1); D=beta[2]+X@beta[3:]; H=hc
Y=np.c_[c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]; F=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); T=np.r_[t,1.]; B=(H[:,None]*beta[:2])@F.T
rms=lambda e: np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
folds=[c[k]==v for k in ('lab_id','stratum') for v in np.unique(c[k])]
folds += [H<=np.quantile(H,.25),H>=np.quantile(H,.75)]
for linear in (False,True):
 G=np.c_[F,T] if linear else F; A=D[:,None,None]*G
 for robust in (False,True):
  def fit(use):
   a=A[use].reshape(-1,G.shape[1]); y=(Y-B)[use].ravel(); coef=np.linalg.lstsq(a,y,rcond=None)[0]
   if robust: coef=least_squares(lambda z:a@z-y,coef,loss='soft_l1',f_scale=.004).x
   return coef
  co=fit(complete); rows=[]; missing_rows=[]
  for hold in folds:
   w=~hold; z=fit(w&complete); err=B+A@z-Y; rows.append(rms(err[hold&complete])); v=G@z
   # Estimate a diagnostic-only imputation correction from other missing coupons with the same missing column.
   shift=(Y-B-A@z)@v/(v@v); dshift=np.zeros(len(Y)); scored=np.zeros(len(Y),bool)
   for col in range(missing.shape[1]):
    ref=w&missing[:,col]; tgt=hold&missing[:,col]
    if ref.any(): dshift[tgt]=np.median(shift[ref]); scored|=tgt
   if scored.any(): missing_rows.append((rms(err[scored]),rms((err+dshift[:,None]*v)[scored])))
  print('law',linear,robust,'coef',co,'worst complete',np.max(rows,axis=0),'worst missing before/after diagnostic-only correction',np.max(missing_rows,axis=0))
for name,hold,feat in [('diagnostic',H,X),('test',m._loads(te,b)[1],xt)]:
 print(name,'hold/drift percentiles',np.round(np.percentile(np.c_[hold,beta[2]+feat@beta[3:]],[0,25,50,75,100],axis=0),4))
out=q.solve_batch(d); assert all(np.isfinite(v).all() for v in out.values()); assert .05<out['estimated_taus'][0]<out['estimated_taus'][1]<25
