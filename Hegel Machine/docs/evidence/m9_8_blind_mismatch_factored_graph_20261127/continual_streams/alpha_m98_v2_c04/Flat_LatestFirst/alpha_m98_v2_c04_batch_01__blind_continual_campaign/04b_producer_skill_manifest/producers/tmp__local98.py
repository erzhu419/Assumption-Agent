import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/chamber98.py').read())
from scipy.optimize import least_squares
V=np.c_[Y,2*R]; raw=c['coupon_features_raw']; panels=c['ref_coupons']; X=raw.copy(); imp=np.isnan(X); split=np.c_[lab,c['stratum']]; _,ix,counts=np.unique(split,axis=0,return_inverse=True,return_counts=True)
for l in np.unique(lab):
 p=panels[lab==l]; X[lab==l]=(X[lab==l]-np.median(p.mean(0)))/np.median(p.std(0))
for name,w in [('plain',np.ones(len(X))),('cell',1/counts[ix]),('zero',np.zeros(len(X)))]:
 fill=np.nansum(X*w[:,None],0)/np.maximum(np.sum(w[:,None]*~imp,0),1e-15); xx=np.where(imp,fill,X); dr=th[2]+xx@th[3:]; print('impute',name,fill.round(4));
 for mode,F in [('simple',np.c_[H,dr]),('offset',np.c_[H,dr,np.ones(len(H))]),('hist',np.c_[H,dr,Z]),('nonlinear',np.c_[H,dr,dr**2]),('features',np.c_[H,dr,xx[:,1:]])]:
  def fit(mask):
   Q=np.linalg.lstsq(F[mask],V[mask],rcond=None)[0]
   def run(q,ret=False):
    ta=np.exp(q); T=np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]*2]); C=np.linalg.lstsq(T,Q.T,rcond=None)[0].T; p=F@C@T.T
    return (p,C) if ret else (p-V)[mask].ravel()
   q=least_squares(run,np.log([.47,2.5]),bounds=(np.log([.2,1]),np.log([.75,10]))).x
   return *run(q,True),np.exp(q)
  p,C,ta=fit(np.ones(len(H),bool)); e=[]
  for l in np.unique(lab):
   v=lab==l; e.append((fit(~v)[0]-V)[v]);
  sc=lambda z:[np.mean(z[:,:-1]**2)**.5,np.mean(z[:,-1]**2)**.5/2]
  print(mode,'tau',ta.round(5),'C',C.round(5).tolist(),'fit',np.round(sc(p-V),5),'worst',np.max([sc(z) for z in e],0).round(5),'missing',np.round(sc((p-V)[imp.any(1)]),5))
print('archive metadata'); import json
for q in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')): print({k:v for k,v in q.items() if k!='facility_telemetry_trace'})
print('archived prediction tail'); import inspect; print(inspect.getsource(m.solve_batch))
