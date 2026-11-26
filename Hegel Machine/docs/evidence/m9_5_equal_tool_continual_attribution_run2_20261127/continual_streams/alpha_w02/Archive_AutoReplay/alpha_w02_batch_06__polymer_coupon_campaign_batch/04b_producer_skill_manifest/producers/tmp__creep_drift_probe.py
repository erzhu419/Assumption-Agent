import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('candidate','/holdout_task/alpha_w02_batch_06_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
case=pickle.load(open('/holdout_task/public_alpha_w02_batch_06.pkl','rb'))[0]; d=case['dataset']; a,b=d['train'],d['test']; n=len(a['lab_id']); dt,t=d['past_dt'],d['eval_times']; cal=m._calibrate(d['calibration_load_cells']); f=m._features(a,b); v=m._inputs(a,cal,dt,d['yield_stress'],f[:n]); w=m._inputs(b,cal,dt,d['yield_stress'],f[n:]); y=np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']]; g=case['gt']; clean=np.c_[g['te_clean']-b['strain_0'][:,None],g['te_true_rate0']]; tt=np.r_[t,1.]; complete=np.isfinite(a['coupon_features_raw']).all(axis=1)
def model(keep, valid, outlab, bounds):
 tr=tuple(z[keep] for z in v); yt=y[keep]; labs=a['lab_id'][keep]; ok=complete[keep]; tau,c=m._fit(tr,dt,t,yt); x=tr[2]; design=m._design(tr,dt,t,np.log(tau)); pred=m._design(valid,dt,t,np.log(tau))@c; base=pred.copy(); flagged=[]
 r=yt-design@c; slope=np.nanmedian(r[:,:-1][:,t>=1.]/t[t>=1.],axis=1)
 for j in range(4):
  use=(labs==j)&ok&np.isfinite(slope)
  if use.sum()<10: continue
  Z=x[use]; q=np.linalg.lstsq(Z,slope[use],rcond=None)[0]; fitted=Z@q
  print('evidence',j,use.sum(),np.round([np.std(fitted),1.4826*np.median(abs(slope[use]-fitted))],5)) if bounds else None
  if np.std(fitted)>.006 and np.std(fitted)>3.5*max(.001,1.4826*np.median(abs(slope[use]-fitted))): flagged.append(j)
 if not flagged: return base,pred,tau,c,flagged
 main=~np.isin(labs,flagged)
 tau,c=m._fit(tuple(z[main] for z in tr),dt,t,yt[main]); pred=m._design(valid,dt,t,np.log(tau))@c; r=yt-m._design(tr,dt,t,np.log(tau))@c
 for j in flagged:
  use=(labs==j)&ok; Z=x[use]; R=r[use,:-1]; q=np.linalg.lstsq(Z,np.nanmedian(R[:,t>=1.]/t[t>=1.],axis=1),rcond=None)[0]; A=Z[:,None,:]*t[None,:,None]; A=A.reshape(-1,5); B=R.ravel(); good=np.isfinite(B); A,B=A[good],B[good]
  for _ in range(6):
   weight=1/np.sqrt(1+((B-A@q)/.015)**2); q=np.linalg.lstsq(A*weight[:,None],B*weight,rcond=None)[0]
  k=outlab==j; pred[k]+=(valid[2][k]@q)[:,None]*tt
 return base,pred,tau,c,flagged
for j in [-1,0,1,2]:
 if j<0: keep=np.ones(n,bool); val=w; lab=b['lab_id']; target=clean; ok=np.isfinite(b['coupon_features_raw']).all(axis=1)
 else: keep=a['stratum']!=j; val=tuple(z[~keep] for z in v); lab=a['lab_id'][~keep]; target=y[~keep]; ok=complete[~keep]
 p,q,tau,c,flag=model(keep,val,lab,j<0); print('fold',j,'flags',flag,'parameters',tau,c)
 for i in range(4):
  u=(lab==i)&ok
  if not u.any(): continue
  ep,eq=abs(target[u]-p[u]),abs(target[u]-q[u]); stats=(lambda e: [np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]) if j<0 else (lambda e: [np.median(e[:,:-1]),np.median(e[:,-1])]); print(' lab',i,'base/new',np.round([stats(ep),stats(eq)],5))
