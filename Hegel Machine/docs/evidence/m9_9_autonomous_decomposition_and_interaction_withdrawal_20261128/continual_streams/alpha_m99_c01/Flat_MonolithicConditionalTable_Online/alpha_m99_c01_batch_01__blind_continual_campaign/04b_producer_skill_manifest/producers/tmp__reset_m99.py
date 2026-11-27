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
Y=np.column_stack((c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs'])); labs=c['lab_id']; strata=c['stratum']; missing=np.isnan(c['coupon_features_raw']).any(1); n=len(labs)
keys=('coupon_features_raw','ref_coupons','lab_id','stratum')
def join(a,b): return {k:np.concatenate([a[k],b[k]],axis=0) for k in keys}
for name,g in [('train',tr),('test',te),('diag',c)]:
 print(name,list(g),'N',len(g['lab_id']),'missing',np.isnan(g['coupon_features_raw']).sum(0)); R=g['ref_coupons']; print('ref center scale',[(int(l),round(float(np.median(R[g['lab_id']==l].mean(0))),4),round(float(np.median(R[g['lab_id']==l].std(0))),4)) for l in np.unique(g['lab_id'])]); u=g.get('aux_chamber_telemetry'); print('telemetry',None if u is None else (u.shape,np.round(np.mean(u,axis=0),3),np.round(np.std(u,axis=0),3)))
for tag,xx in [('diag',m._features(c,c)[1]),('test_diag',m._features(te,c)[1]),('train_test_diag',m._features(join(tr,te),c)[1]),('train_diag',m._features(tr,c)[1])]:
 D=beta[2]+xx@beta[3:]; H=hc; F=np.vstack((np.column_stack((-np.expm1(-t[:,None]/tau),t)),np.r_[1/tau,1.])); base=H[:,None]*beta[:2]@F[:,:2].T
 for kind in ('zero','offset','full'):
  K=[D[:,None]*F[None,:,j] for j in (0,1)]
  if kind!='zero': K += [np.broadcast_to(F[None,:,j],Y.shape) for j in (0,1)]
  if kind=='full': K += [np.broadcast_to(F[None,:,2],Y.shape)]
  K=np.stack(K,axis=-1)
  def fit(idx):
   A=K[idx].reshape(-1,K.shape[-1]); yy=(Y-base)[idx].ravel(); z=np.linalg.lstsq(A,yy,rcond=None)[0]; return least_squares(lambda b:A@b-yy,z,loss='soft_l1',f_scale=.003).x
  def rms(e): return [np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]
  coef=fit(np.ones(n,dtype=bool)); e=K@coef+base-Y; mask=[labs==l for l in np.unique(labs)]+[strata==s for s in np.unique(strata)]; worst=np.max([rms((K@fit(~w)+base-Y)[w]) for w in mask],0); print(tag,kind,'coef',coef.round(5),'complete',np.round(rms(e[~missing]),5),'missing',np.round(rms(e[missing]),5),'worst',worst.round(5))
