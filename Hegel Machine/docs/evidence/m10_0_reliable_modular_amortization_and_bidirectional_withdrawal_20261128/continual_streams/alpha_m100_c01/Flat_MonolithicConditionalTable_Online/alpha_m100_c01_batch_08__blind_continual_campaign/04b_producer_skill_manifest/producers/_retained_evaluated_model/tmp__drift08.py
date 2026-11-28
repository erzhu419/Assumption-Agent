exec(open('/tmp/chamber08.py').read().split('ramps=')[0])
m=runpy.run_path('/holdout_task/batch_solver.py')
key=('lab_id','stratum','ref_coupons','coupon_features_raw')
def join(*groups): return {k:np.concatenate([a[k] for a in groups]) for k in key}
def summarize(groups):
 for grp in groups:
  print('panel',len(grp['lab_id']),[(l,*np.round([np.median(grp['ref_coupons'][grp['lab_id']==l].mean(0)),np.median(grp['ref_coupons'][grp['lab_id']==l].std(0))],4)) for l in np.unique(grp['lab_id'])])
summarize([tr,te,g,join(tr,te),join(tr,g)])
for label,xx in [('tr+g',_features(tr,g)[1]),('te+g',_features(te,g)[1]),('g',_features({k:g[k][:0] for k in key},g)[1]),('all',_features(join(tr,te),g)[1])]:
 fs=np.c_[np.ones(len(H)),xx]; score=fs@th[2:];
 def ramp(u): return np.r_[2*t/(1+np.sqrt(1+t/np.exp(u))),1.]
 for kind in ['coupled','affine','full']:
  def pred(v):
   drift=score if kind=='coupled' else v[1]*score+v[2] if kind=='affine' else fs@v[1:]; return base+np.outer(drift,ramp(v[0]))
  ini=np.r_[0.,[] if kind=='coupled' else [1.,0.] if kind=='affine' else th[2:]]
  def fit(s): return least_squares(lambda v:((pred(v)-Y)[s]*W).ravel(),ini,loss='soft_l1',f_scale=.012).x
  err=np.empty_like(Y)
  for l in np.unique(labs):
   s=labs==l; err[s]=abs((pred(fit(~s))-Y)[s]);
  print(label,kind,fit(allrows).round(5), 'pooled',np.round([np.median(err[:,:-1]),np.median(err[:,-1])],5),'worst',np.round(np.max([[np.median(err[labs==l,:-1]),np.median(err[labs==l,-1])] for l in np.unique(labs)],axis=0),5))
