import importlib.util,pickle,numpy as np
base='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',base+'alpha_w03_batch_03_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=pickle.load(open(base+'public_alpha_w03_batch_03.pkl','rb'))[0];d=r['dataset'];g=r['gt'];a,b=d['train'],d['test'];n=len(a['lab_id']); x,h,p=m._prep(d);dt,t=d['past_dt'],d['eval_times'];mem=np.r_[a['initial_memory_states'],b['initial_memory_states']];raw=np.r_[a['coupon_features_raw'],b['coupon_features_raw']];lab=np.r_[a['lab_id'],b['lab_id']];mask=np.isnan(raw);miss=mask.any(1);y=a['strain_curves_obs']-a['strain_0'][:,None];rate=np.r_[a['instant_rate_obs'],b['instant_rate_obs']]; q=b['critical_strain']-b['strain_0'];end=t[-1]
for clean_fit in (False,True):
 for held in (0,1,2,3,None):
  fit=np.flatnonzero((lab[:n]!=held)&(~miss[:n] if clean_fit else True)); val=np.flatnonzero(lab[:n]==held) if held is not None else np.arange(n,len(x)); tau,th=m._fit(x[fit],h[fit],p[fit],mem[fit],dt,t,y[fit],rate[fit]); c,rr,*_=m._design(tau,x,h,p,mem,dt,t); mu=x@th[2:]; fit_drift=(y-(c@th)[:n])@t/(t@t); innov=rate-rr@th; amp=rr[:,:2]*tau*th[:2];print('clean fit',clean_fit,'held',held,'tau',np.round(tau,5))
  for lab0 in (range(4) if held is None else [held]):
   j=val[lab[val]==lab0]; print(' lab',lab0,'n missing',int(miss[j].sum()))
   for z in (j[~miss[j]],j[miss[j]]):
    if not len(z): continue
    observed=fit_drift[z] if held is not None else ((g['te_clean'][z-n]-b['strain_0'][z-n,None]-(c@th)[z])@t/(t@t))
    print('    drift RMSE alpha',[(alpha,round(np.mean((observed-alpha*innov[z])**2)**.5,5)) for alpha in (0,.25,.5,.75,.95,1.)], 'innovation max',round(abs(innov[z]).max(),4))
  if held is None:
   for alpha0,alpha1 in ((0,0),(0,.75),(0,.95),(0,1),(.05,.95),(.2,.95)):
    alpha=np.where(miss[n:],alpha1,alpha0); dr=alpha*innov[n:]; curve=(c@th)[n:]+dr[:,None]*t+b['strain_0'][:,None]; sec=(curve[:,-1]-b['strain_0'])/np.maximum(abs(h[n:]),.5); cr=np.array([m._crossing(tau,A,v,Q,end) for A,v,Q in zip(amp[n:],mu[n:]+dr,q)]); delta=[curve-g['te_clean'],sec-g['te_secant'],cr-g['te_cross']]; groups=[lab[n:]==i for i in range(4)]; print(' alpha',alpha0,alpha1,'all',[round(np.mean(e**2)**.5,5) for e in delta],'worst lab',[round(max(np.mean(e[k]**2)**.5 for k in groups),5) for e in delta], 'disagree',np.flatnonzero((cr<end)!=(g['te_cross']<end)).tolist())
