import contextlib,io,runpy,numpy as np
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path('/tmp/drift_grammar.py')
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
for j in np.unique(lab):
 q=(lab==j)&~missing.any(1)
 rawdot=di['coupon_features_raw'][q]@theta[3:]; target=inferred[q]-theta[2]; slope,ic=np.linalg.lstsq(np.c_[rawdot,np.ones(len(rawdot))],target,rcond=None)[0]
 print('lab',j,'inferred offset/scale',-ic/slope/theta[3:].sum(),1/slope,'amp mae',np.mean(abs(target-(rawdot*slope+ic))))
for name,sites in [('train',(tr,)),('test',(te,)),('diag',(di,)),('train-test',(tr,te)),('train-diag',(tr,di)),('test-diag',(te,di)),('all',(tr,te,di))]:
 for axis in (0,1):
  rr=np.concatenate([s['ref_coupons'] for s in sites]); ll=np.concatenate([s['lab_id'] for s in sites]); z0=di['coupon_features_raw'].copy(); out=[]
  for j in np.unique(lab):
   v=rr[ll==j]; center=np.median(v.mean(axis)); scale=np.median(v.std(axis)); q=lab==j; z0[q]=(z0[q]-center)/scale; p=(theta[2]+z0[q]@theta[3:]); out.append((round(center,4),round(scale,4),round(np.nanmean(abs(p-inferred[q])),5)))
  print(name,axis,out)
print('theta feature sum',theta[3:].sum())
