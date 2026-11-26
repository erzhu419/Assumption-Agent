import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w03_batch_04_solver.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_alpha_w03_batch_04.pkl','rb'))[0]['dataset'];tr=d['train'];n=len(tr['lab_id']);t=d['eval_times'];dt=d['past_dt'];lab=tr['lab_id'];x,h,p=[a[:n] for a in s._prep(d)];hold,work=s._overstress(h,p,dt,d['yield_stress']);y=tr['strain_curves_obs']-tr['strain_0'][:,None];r=tr['instant_rate_obs'];miss=np.isnan(tr['coupon_features_raw']).any(axis=1)
for linear in (False, True):
 cy=np.zeros_like(y);rr=np.zeros_like(r)
 ho,wo=(np.zeros_like(hold),np.zeros_like(work)) if linear else (hold,work)
 for ell in sorted(set(lab)):
  a=lab!=ell;b=~a
  tau,w=s._fit(x[a],h[a],p[a],dt,t,ho[a],wo[a],y[a],r[a]);c,v,*_=s._design(tau,x[b],h[b],p[b],dt,t,ho[b],wo[b]);cy[b]=c@w;rr[b]=v@w
  print('linear' if linear else 'unified','lab',ell,'tau',tau.round(5),'weights',w.round(5),'RMSE',np.sqrt(np.mean((cy[b]-y[b])**2)),np.sqrt(np.mean((rr[b]-r[b])**2)))
 for title,m in [('complete',~miss),('missing',miss),('sub-yield',(hold==0)&(work==0)),('hold over yield',hold!=0),('history-only over yield',(hold==0)&(work!=0))]:
  if m.any():
   e=cy[m]-y[m];dr=r[m]-rr[m];print(title,m.sum(),'strain by horizon',np.sqrt(np.mean(e**2,axis=0)).round(5),'rate',np.sqrt(np.mean(dr**2)).round(5),'strain correction 1',np.sqrt(np.mean((e+dr[:,None]*t)**2,axis=0)).round(5))
