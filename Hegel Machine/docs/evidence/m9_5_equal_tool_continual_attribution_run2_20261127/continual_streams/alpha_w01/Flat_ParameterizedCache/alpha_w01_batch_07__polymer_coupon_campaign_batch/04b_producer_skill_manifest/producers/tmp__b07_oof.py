import importlib.util as iu, pickle
import numpy as np
r='/holdout_task/'; s=iu.spec_from_file_location('m',r+'alpha_w01_batch_07_solver.py'); m=iu.module_from_spec(s); s.loader.exec_module(m)
c=pickle.load(open(r+'public_alpha_w01_batch_07.pkl','rb'))[0]; d=c['dataset']; tr,te=d['train'],d['test']; t,dt,sy=(d[k] for k in ('eval_times','past_dt','yield_stress')); mp=m._calibrate(d['calibration_load_cells']); lab=tr['lab_id']; x,xt,co,ct=m._align(tr,te); fold=np.empty(len(lab),int); rng=np.random.default_rng(707)
for l in range(4):
 for cc in (True,False):
  ids=np.flatnonzero((lab==l)&(co==cc)); fold[rng.permutation(ids)]=np.arange(len(ids))%5
base=np.empty(len(lab)); curve=np.empty_like(tr['strain_curves_obs']); tau=[]
for f in range(5):
 q=fold==f; sub={k:v[~q] for k,v in tr.items()}; val={k:v[q] for k,v in tr.items()}; xx,xv,cs,_=m._align(sub,val); ts,th=m._fit(sub,xx,cs,mp,dt,t,sy); tau.append(ts); h,v=m._stress(val,mp); A,B=m._design(ts,h,v,xv,dt,t,*m._plastic(h,v,dt,sy)); base[q]=B@th; curve[q]=val['strain_0'][:,None]+A@th
Y=tr['strain_curves_obs']-curve; onset=tr['instant_rate_obs']-base
# Independent slope measurement: reject gross point errors with a median,
# then average the consistent late-time strain measurements.
keep=t>=np.median(t); ratios=Y[:,keep]/t[keep]; center=np.median(ratios,axis=1); mask=np.abs(ratios-center[:,None])<.035; wt=mask*t[keep]**2; slope=np.sum(wt*ratios,axis=1)/wt.sum(axis=1)
print('fold taus',np.round(tau,5)); print('time',t)
for l in range(4):
 for tag,cc in [('complete',co),('missing',~co)]:
  q=(lab==l)&cc
  print(l,tag,q.sum(),'median abs base, onset vs curve slope',np.median(np.abs(np.c_[slope[q],onset[q]-slope[q]]),axis=0),'slope-onset quantiles',np.quantile(np.abs(onset[q]-slope[q]),[.5,.8,.9,1]),'blend robust errors',[round(float(np.median(abs(a*onset[q]-slope[q]))),6) for a in (0,.25,.5,.75,1)],'gross rate count',np.sum(abs(onset[q]-slope[q])>.1))
np.savez('/tmp/b07_oof.npz',base=base,curve=curve,slope=slope,onset=onset,complete=co,fold=fold)
