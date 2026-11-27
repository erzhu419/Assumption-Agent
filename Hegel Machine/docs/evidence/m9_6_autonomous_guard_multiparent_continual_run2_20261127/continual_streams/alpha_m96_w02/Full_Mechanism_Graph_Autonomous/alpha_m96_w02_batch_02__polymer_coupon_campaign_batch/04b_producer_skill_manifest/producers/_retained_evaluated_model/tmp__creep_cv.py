import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m96_w02_batch_02_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
e=pickle.load(open('/holdout_task/public_alpha_m96_w02_batch_02.pkl','rb'))[0]; d=e.get('dataset',e.get('data',e)); tr,te=d['train'],d['test']; n=len(tr['lab_id']); x=m._align(tr,te); cal=m._stress_calibration(d['calibration_load_cells']); dt,t=d['past_dt'],d['eval_times']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; inp=m._inputs(tr,x[:n],cal); rate=tr['instant_rate_obs']; lab=tr['lab_id']; miss=np.isnan(tr['coupon_features_raw']); missing=miss.any(1)
def mad(a):
 return 1.4826*np.median(abs(a-np.median(a)))
def fit(i,complete):
 i=i&(~missing if complete else True)
 return m._fit(m._Design(dt,t,tuple(a[i] for a in inp)),y[i],rate[i])
def diagnostics(p,r,lab,miss,y):
 for c in range(4):
  for missing in (False,True):
   j=(lab==c)&(miss.any(1)==missing)
   if not j.any():continue
   print(c,int(missing),j.sum(),'rmse(0,.85,.95,1)',[round(np.sqrt(np.mean((p[j]+k*r[j,None]*t-y[j])**2)),5) for k in (0,.85,.95,1)],'median coupon', [round(np.median(np.sqrt(np.mean((p[j]+k*r[j,None]*t-y[j])**2,axis=1))),5) for k in (0,.95)])
for complete in (False,True):
 print('complete fit',complete)
 p=np.empty_like(y); r=np.empty(n)
 for c in range(4):
  j=lab==c; tau,b=fit(~j,complete); a=m._Design(dt,t,tuple(v[j] for v in inp)).make(tau); p[j]=a[:,:-1]@b; r[j]=rate[j]-a[:,-1]@b
 diagnostics(p,r,lab,miss,y)
 # A median across late horizons isolates coupon-level drift from scattered curve outliers.
 slope=np.median((y[:,6:]-p[:,6:])/t[6:],axis=1)
 print('robust complete slope noise, rate innovation, rate minus slope',mad(slope[~missing]),mad(r[~missing]),mad((r-slope)[~missing]))
 for k in range(4):
  j=miss[:,k] & (miss.sum(1)==1);print('missing col',k,j.sum(),'slope mad',mad(slope[j]) if j.any() else None)
 tau,b=fit(np.ones(n,bool),complete); a=m._Design(dt,t,m._inputs(te,x[n:],cal)).make(tau); p=a[:,:-1]@b+te['strain_0'][:,None]; r=te['instant_rate_obs']-a[:,-1]@b
 print('tau/theta',tau,b);print('public clean diagnostics (not fitted):'); diagnostics(p,r,te['lab_id'],np.isnan(te['coupon_features_raw']),e['gt']['te_clean'])
