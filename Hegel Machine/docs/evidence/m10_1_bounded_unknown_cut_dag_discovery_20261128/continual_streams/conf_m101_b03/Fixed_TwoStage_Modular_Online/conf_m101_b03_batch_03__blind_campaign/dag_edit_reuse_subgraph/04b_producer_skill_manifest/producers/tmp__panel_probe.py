import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v2']; n=len(u)
def probe(D,tag):
 D=np.column_stack((np.ones(n),D)); r=D@np.linalg.lstsq(D,v,rcond=None)[0]-v; H=D@np.linalg.pinv(D); loo=r/np.maximum(1-H.diagonal(),1e-12)[:,None]; print(tag,D.shape[1],np.round(np.sqrt(np.mean(r*r,0)),5),np.round(np.sqrt(np.mean(loo*loo,0)),5))
for deg in (2,3,4):
 for parity in (0,1):
  p=[q for d in range(1,deg+1) if not parity or d%2==0 for q in itertools.combinations_with_replacement(range(4),d)];
  if len(p)<22:probe(np.stack([np.prod(u[:,q],1) for q in p],1),('poly',deg,parity))
W=np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)])
for f in (.8,1,1.2,1.5,1.8,2,2.2,2.5,2.8,3,3.5,4,5):
 for D,tag in [(np.cos(f*u),'cos'),(np.c_[np.cos(f*u),np.cos(f*W)],'cos+cosprod'),(np.c_[np.cos(f*u),W],'cos+prod'),(np.c_[np.cos(f*u),u*u],'cos+sq'),(np.c_[np.cos(f*u),u],'cos+linear')]:probe(D,(tag,f))
for norm in ('separated','combined'):
 def design(f):
  Z=np.c_[np.cos(u*f[:4]),np.cos(W*f[4:])] if norm=='separated' else np.cos(u*f)
  return np.c_[np.ones(n),Z]
 def fun(f):
  D=design(f); return (D@np.linalg.lstsq(D,v,rcond=None)[0]-v).ravel()
 fit=least_squares(fun,np.full(10 if norm=='separated' else 4,2.),bounds=(.2,10)); probe(design(fit.x)[:,1:],(norm,fit.x.round(4)))
p=s['panel_a']; A=np.c_[np.ones(len(p['u'])),p['u']]; Y=p['v']; w=np.ones(len(A))
for _ in range(35):
 B=np.linalg.lstsq(A*w[:,None],Y*w[:,None],rcond=None)[0]; r=np.linalg.norm(A@B-Y,axis=1); w=1/np.hypot(1,r/max(np.median(r),1e-7))
print('A coef',B,'residual sorted',np.sort(r).round(6),'cond',np.linalg.cond(B[1:]))
