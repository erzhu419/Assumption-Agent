import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; lab=tr['lab_id']; n=len(lab); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; past,hold,x,active,act=m._prepare(d)
def join(k): return np.concatenate([tr[k],te[k]])
missing=~np.isfinite(join('coupon_features_raw')); labs=join('lab_id'); strata=join('stratum'); pooled=x.copy()
for j in range(x.shape[1]):
 means=[x[(labs==c)&(strata==s)&~missing[:,j],j].mean() for c,s in np.unique(np.column_stack((labs,strata)),axis=0) if np.any((labs==c)&(strata==s)&~missing[:,j])]
 pooled[missing[:,j],j]=np.mean(means)
print('max imputation change',np.max(abs(pooled-x)))
slices=[('active',active[:n]),('zero_damage',~active[:n]),('missing',missing[:n].any(axis=1))]+[(f'lab{c}',lab==c) for c in np.unique(lab)]
for law,gate in [('damage-gated-flow',active),('low-banding-flow',np.full(len(hold),d['shear_banding_index']<=.50))]:
 for fill,features in [('local',x),('cell-balanced pooled',pooled)]:
  state=(past,hold,features,gate,act); ec=np.empty_like(y); er=np.empty_like(r); taus=[]
  for c in np.unique(lab):
   a=lab!=c; b=~a
   tau,w=m._fit(tuple(v[:n][a] for v in state),dt,t,y[a],r[a]); A,B=m._design(tau,tuple(v[:n][b] for v in state),dt,t)
   ec[b]=A@w-y[b]; er[b]=B@w-r[b]; taus.append(tau)
  print(law,fill,'taus range',np.round(np.min(taus,axis=0),5),np.round(np.max(taus,axis=0),5))
  for name,k in slices:
   print(name,'curve median abs / rmse, rate rmse, late median abs',*[round(z,6) for z in [np.median(abs(ec[k])),np.sqrt(np.mean(ec[k]**2)),np.sqrt(np.mean(er[k]**2)),np.median(abs(ec[k,-3:]))]])
