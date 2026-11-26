import importlib.util,pickle,numpy as np
from pathlib import Path
root=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',root/'alpha_w03_batch_06_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.loads((root/'public_alpha_w03_batch_06.pkl').read_bytes())[0]['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id'])
x,h,p=m._prep(d); dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
from scipy.stats import median_abs_deviation
lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(1)
# A separate robust slope per coupon separates persistent drift from cell spikes.
u,v=np.triu_indices(len(t),1)
def slope(err): return np.median(err[:,v]/t[v],axis=1) if False else np.median(err/t,axis=1)
def mad(a): return median_abs_deviation(a,scale='normal')
fold=np.empty(n,int)
for a in range(4):
 j=np.flatnonzero(lab==a); fold[j]=np.random.default_rng(a+104).permutation(len(j))%5
obs_err=np.empty_like(y); rate_err=np.empty_like(r); shifted=np.empty_like(y); anchored=np.empty_like(y); intercept=np.empty(n)
for f in range(5):
 fit=(fold!=f)&complete; val=fold==f
 tau,co=m._fit(x[:n][fit],h[:n][fit],p[:n][fit],dt,t,sy,y[fit],r[fit]); c,rr=m._design(tau,x[:n],h[:n],p[:n],dt,t,sy)
 e=y-c@co; er=r-rr@co; dr=slope(e)
 for a in range(4):
  j=fit&(lab==a); q=val&(lab==a); b=np.median(dr[j]); e0=er[j]-np.median(er[j]); s0=dr[j]-b
  clean=(abs(e0)<3*max(mad(e0),1e-6))&(abs(s0)<3*max(mad(s0),1e-6))
  shrink=np.clip(np.dot(e0[clean],s0[clean])/max(np.dot(e0[clean],e0[clean]),1e-12),0,1)
  obs_err[q]=e[q]; rate_err[q]=er[q]; intercept[q]=b
  shifted[q]=e[q]-b*t
  anchored[q]=e[q]-(b+shrink*np.clip(er[q]-b,-.04,.04))[:,None]*t
  print('fold/lab/drift/shrink',f,a,round(b,5),round(shrink,3))
work=np.maximum(abs(p[:n])-sy,0)**2@dt; hold=abs(h[:n])>sy
for a in range(4):
 j=(lab==a)&complete; sc=slope(obs_err[j]); er=rate_err[j]; print('lab median drift/rate and MAD',a,np.round([np.median(sc),np.median(er),mad(sc),mad(er)],6));
 for s,q in [('elastic',(work==0)&~hold),('history',(work>0)&~hold),('hold',hold)]:
  q=q&j
  if q.any(): print('slice',a,s,int(q.sum()),'curve medAE / clean MAE / late medAE, unshifted/intercept/anchored',[(round(np.median(abs(z[q])),6),round(np.mean(abs(z[q][abs(obs_err[q])<.12])),6),round(np.median(abs(z[q,-3:])),6)) for z in (obs_err,shifted,anchored)])
q=~complete; print('missing slope/onset disagreement quantiles',np.quantile(abs(slope(obs_err[q])-rate_err[q]),[.25,.5,.75,1])); print('test onset residual by lab (complete only)'); tau,co=m._fit(x[:n][complete],h[:n][complete],p[:n][complete],dt,t,sy,y[complete],r[complete]); _,rt=m._design(tau,x[n:],h[n:],p[n:],dt,t,sy); e=te['instant_rate_obs']-rt@co
for a in range(4):
 q=(te['lab_id']==a)&np.isfinite(te['coupon_features_raw']).all(1); print(a,np.round([np.median(e[q]),mad(e[q])],6))
