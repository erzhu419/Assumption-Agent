exec(open('/tmp/compact.py').read().split('results=[]')[0]);
from scipy.linalg import qr

def fit_groups(keys,I):
 fs=[F(k) for k in keys]; orth=[qr(f[I],mode='economic')[0] for f in fs]; target=T[I].ravel(); groups=len(fs)
 def project(lr,output=False):
  Q=-np.expm1(-np.exp(lr)[:,None]*t)
  design=np.column_stack([(a[:,:,None]*q[None,None,:]).transpose(0,2,1).reshape(len(target),-1) for a,q in zip(orth,Q)])
  w=np.linalg.lstsq(design,target,rcond=1e-11)[0]
  if not output:return design@w-target
  yhat=np.zeros_like(T); pos=0
  for f,a,q in zip(fs,orth,Q):
   p=a.shape[1]; yhat+=(f@np.linalg.lstsq(f[I],a@w[pos:pos+p],rcond=1e-10)[0])[:,None]*q;pos+=p
  return yhat,np.exp(lr),np.sqrt(np.mean((yhat[I]-T[I])**2))
 start=[.42,1.8] if groups==2 else [.42,1.8,.65]
 est=least_squares(project,np.log(start),bounds=(np.log([.12,1.0]+[.08]*(groups-2)),np.log([.9,3.8]+[4.]*(groups-2))),max_nfev=80)
 return project(est.x,True)
for keys in [('ovh','ozumhv'),('ov','oum','h'),('ov','ozum','h'),('ov','oum','oh'),('ov','oum','hm'),('ov','ou','hm'),('ovh','oum','vh'),('ovh','oum','v'),('ov','oum','vh'),('ov','oum','uz')]:
 E=[]
 for order in orders:
  I,J=order[:52],order[52:]; P,r,err=fit_groups(keys,I); e=abs(P[J]-T[J]); E.append([e.mean(),e.mean(0).max(),e.mean(1).max()]);
  if len(E)==1: first=(r,err)
 E=np.array(E);print(keys,first,'split',E[0],'random max',E[:26].max(0),'tail max',E[26:].max(0))
