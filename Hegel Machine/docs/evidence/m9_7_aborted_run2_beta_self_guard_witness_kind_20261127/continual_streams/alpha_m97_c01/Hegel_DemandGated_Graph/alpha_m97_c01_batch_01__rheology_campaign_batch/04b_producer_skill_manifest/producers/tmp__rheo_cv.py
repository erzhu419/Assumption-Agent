import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']; past,hold,x,s0=m._inputs(d); labs=d['train']['lab_id']; n=len(labs); past,hold,x=past[:n],hold[:n],x[:n]; y=np.column_stack((d['train']['strain_curves_obs']-s0[:n,None],d['train']['instant_rate_obs'])); groups=d['train']['stratum']; damage=np.maximum(abs(past)-d['crit_stress'],0)**2@d['past_dt'];
def model(mask,mode):
 z=m._Model(d,past[mask],hold[mask],x[mask]);
 if mode in ('global_log','linear_drift'):
  z.terminal[:,:-1]=np.log1p(z.t) if mode=='global_log' else z.t
  z.drift_basis=z.terminal[:,:,None]*z.x[:,None,:]
  if mode=='linear_drift':z.terminal[:,:-1]=np.log1p(z.t)
 return z
for mode in ('linear','gated','global_log','linear_drift'):
 pred=np.empty_like(y); params=[]
 for lab in np.unique(labs):
  te=labs==lab; tr=~te
  tau,w=m._fit(model(tr,mode),y[tr],plateau=mode!='linear'); params.append(np.r_[tau,w]); pred[te]=model(te,mode).design(np.log(tau),plateau=mode!='linear')[0]@w
 r=pred-y; a=abs(r); print('\n',mode,'fold tau and rw',np.round(np.asarray(params)[:,[0,1,-2,-1]],5)); print('pooled MAE, med strain/rate',a[:,:-1].mean(),a[:,-1].mean(),np.median(a[:,:-1]),np.median(a[:,-1])); print('horizon median',np.round(np.median(a,axis=0),5));
 for lab in np.unique(labs):
  for active in (False,True):
   mask=(labs==lab)&((damage>0)==active); b=a[mask]; print('lab,active,n,median strain/rate,90p strain/rate, late bias',lab,active,len(b),*[round(v,5) for v in (np.median(b[:,:-1]),np.median(b[:,-1]),np.quantile(b[:,:-1],.9),np.quantile(b[:,-1],.9),np.median(r[mask,-2]))])
 if mode=='gated':
  print('gated median signed residual by lab and stratum:');
  for lab in np.unique(labs):
   for g in np.unique(groups):
    q=(labs==lab)&(groups==g); print(lab,g,q.sum(),np.median(r[q],axis=0).round(4))
  np.save('/tmp/rheo_gated_oof.npy',r)
