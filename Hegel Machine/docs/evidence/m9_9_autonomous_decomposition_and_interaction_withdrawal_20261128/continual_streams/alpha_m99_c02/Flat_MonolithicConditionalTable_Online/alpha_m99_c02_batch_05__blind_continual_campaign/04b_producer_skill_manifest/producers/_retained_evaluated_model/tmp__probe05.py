import numpy as np,pickle,runpy,itertools,json
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_batch_04_solver.py'); base=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_batch_02_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; dt,t=ds['past_dt'],ds['eval_times']; c=ds['calibration_load_cells']; l,r=c['true_channel_loads'],c['bridge_raw']; b=m['_bridge'](c)
print('cal range',l.min(),l.max()); print('archive coefficient',b.round(5)); print('archive cal residual percentiles',np.percentile(np.linalg.norm(m['_bridge_basis'](l)@b-r,axis=1),[0,25,50,65,75,90,100]))
for p,cols in [('affine',[np.ones(len(l)),l]),('archive',[m['_bridge_basis'](l)]),('all pairs',[np.ones(len(l)),l,np.array([l[:,a]*l[:,b] for a,b in itertools.combinations(range(4),2)]).T,l**3]),('all squares pairs cubes',[np.ones(len(l)),l,np.array([l[:,a]*l[:,b] for a,b in itertools.combinations_with_replacement(range(4),2)]).T,l**3]),('all third degree',[np.ones(len(l)),l,*[np.array([l[:,inds].prod(axis=1) for inds in itertools.combinations_with_replacement(range(4),k)]).T for k in [2,3]]])]:
 a=np.column_stack(cols); rng=np.random.default_rng(9); losses=[]
 for ids in np.array_split(rng.permutation(len(a)),5):
  train=np.setdiff1d(np.arange(len(a)),ids); ats,rs=a[train],r[train]; cut=int(.73*len(train)); best=(np.inf,None)
  for _ in range(70):
   s=rng.choice(len(train),min(len(train),a.shape[1]+6),False); coef=np.linalg.lstsq(ats[s],rs[s],rcond=None)[0]
   for j in range(5):
    ix=np.argsort(np.linalg.norm(ats@coef-rs,axis=1))[:cut]; coef=np.linalg.lstsq(ats[ix],rs[ix],rcond=None)[0]
   mse=np.mean((ats[ix]@coef-rs[ix])**2)
   if mse<best[0]:best=mse,coef
  losses.extend(np.linalg.norm(a[ids]@best[1]-r[ids],axis=1))
 print(p,'cal oof',np.percentile(losses,[25,50,65,75,90]))
xt,xe=m['_features'](tr,te); _,x=m['_features']({k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},d); tt,w=m['_material'](tr,b,xt,dt,t); print('material',tt,w)
for name,s,xx in [('train',tr,xt),('diagnostics',d,x)]:
 past,hold=m['_loads'](s,b); ac,ar=m['_basis'](tt,past,hold,xx,dt,t); y=s['strain_curves_obs']-s['strain_0'][:,None]; v=s['instant_rate_obs']; drift=w[2]+xx@w[3:]; f=-np.expm1(-t[:,None]/tt); F=np.vstack([np.column_stack([f,t]),np.r_[1/tt,1]]); Y=np.column_stack([y,v]); amps=np.linalg.lstsq(F,Y.T,rcond=None)[0].T; E=Y-amps@F.T
 def report(label,curve,rate):
  print(name,label,'median abs curve/rate',np.median(abs(y-curve)),np.median(abs(v-rate)),'max lab MAE',max(np.mean(abs(y[s['lab_id']==lab]-curve[s['lab_id']==lab])) for lab in np.unique(s['lab_id'])))
 report('baseline',ac@w,ar@w); report('batch04',*m['_chamber'](tt,w,hold,xx,t)); report('batch02',*base['_chamber'](tt,w,past,hold,xx,dt,t)); print('free intrinsic time fit percentiles',np.percentile(abs(E),[50,90,100]),'missing',np.isnan(s['coupon_features_raw']).sum(1));
 if name=='diagnostics':
  gap=ar[:,:2]*tt; state=hold[:,None]-gap; q=np.column_stack([hold,drift,state]); good=np.isfinite(s['coupon_features_raw']).all(1); print('amp regression on hold drift states',np.linalg.lstsq(q[good],amps[good],rcond=None)[0]); print('residual sign across time',np.quantile(E,[.1,.5,.9],axis=0).round(5)); print('amplitudes span',np.quantile(amps,[0,.5,1],axis=0).round(4))
