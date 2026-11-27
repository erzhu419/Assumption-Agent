exec(open('/tmp/chamber_probe.py').read().split('for block,features')[0])
P,H,_=inp(diag,xd); T,Y=obs(diag); Q=np.c_[T,Y]; complete=np.isfinite(diag['coupon_features_raw']).all(1); labs=diag['lab_id']; idx=np.arange(len(Q)); age=np.cumsum(dt[::-1])[::-1]-dt
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); entries=log if isinstance(log,list) else next(v for v in log.values() if isinstance(v,list));
for row in entries:
 print('apparatus', {k:(v if np.isscalar(v) else str(v)[:160]) for k,v in row.items() if k!='facility_telemetry_trace'})

def features(refs):
 X=diag['coupon_features_raw'].copy(); lab=diag['lab_id']; _,cell,count=np.unique(np.c_[lab,diag['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/count[cell];
 for l in np.unique(lab):
  R=np.vstack([v['ref_coupons'][v['lab_id']==l] for v in refs]); X[lab==l]=(X[lab==l]-np.median(R.mean(axis=0)))/np.median(R.std(axis=0));
 good=np.isfinite(X); return np.c_[np.ones(len(X)),np.where(good,X,np.nansum(X*w[:,None],0)/(good*w[:,None]).sum(0))]

def kernel(s): return np.r_[-np.expm1(-t/s),1/s]
def memory(m): return -np.expm1(-dt/m)*np.exp(-age/m)
K=np.array([kernel(s) for s in taus]); hold=(H[:,None]*theta[:2])@K

def run(X,kind):
 D=X@theta[2:]; nonlinear=kind in ('signed','square','abs'); independent=kind in ('coeff','intercept');
 def basis(p):
  k=kernel(p[2])*p[2]; h=[np.outer(P@memory(p[j]),K[j]) for j in range(2)];
  if kind=='coeff': u=list(X.T)
  elif kind=='intercept': u=[D,np.ones(len(D))]
  elif kind=='signed': u=[D,D*np.abs(D)]
  elif kind=='square': u=[D,D**2]
  elif kind=='abs': u=[D,np.abs(D)]
  else: u=[D]
  return np.stack(h+[np.outer(z,k) for z in u],-1)
 def fit(ix):
  def res(p,full=False):
   B=basis(p); A=B[ix].reshape(-1,B.shape[-1]); y=(Q-hold)[ix].ravel(); c=np.linalg.lstsq(A,y,rcond=None)[0]; return (hold+B@c-Q,c) if full else A@c-y
  f=least_squares(res,[.8,2.,1.],bounds=([.15,.3,.25],[5,20,5])); return f.x,*res(f.x,True)
 def metrics(e): return np.array([np.mean(e[:,:-1]**2),np.mean(e[:,-1]**2)])**.5
 p,e,c=fit(idx); cv=np.array([metrics(fit(idx[labs!=l])[1][labs==l]) for l in np.unique(labs)]); print(kind,'p',p.round(4),'c',c.round(4),'fit',metrics(e).round(5),'worst lab',cv.max(0).round(5),'complete/missing',[metrics(e[i]).round(5).tolist() for i in [complete,~complete]])
for name,refs in [('train+diag',[a,diag]),('train+test',[a,b]),('train+test+diag',[a,b,diag]),('diag only',[diag]),('train only',[a])]:
 print(name); X=features(refs)
 for kind in ['linear','intercept','coeff','signed','square','abs']: run(X,kind)
