import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True,linewidth=150)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(u):return np.c_[np.ones(len(u)),u]
def score(a,v):
 c=ls(a,v);h=np.sum(np.linalg.qr(a)[0]**2,axis=1);r=(a@c-v)/(1-h) if v.ndim==1 else (a@c-v)/(1-h[:,None]);return np.mean(r*r)**.5,c
p=s['panel_a'];u=p['u'];v=p['v'];q=np.roll(u,-1,axis=1);base=np.stack((np.ones_like(u),u,q,u*q),axis=-1); e=np.stack([base[:,j]@ls(base[:,j],v[:,j])-v[:,j] for j in range(4)],axis=1)
for _ in range(20):
 r=np.linalg.norm(e,axis=1); w=np.sqrt(np.minimum(1,1.3*np.median(r)/r));e=np.stack([base[:,j]@ls(w[:,None]*base[:,j],w*v[:,j])-v[:,j] for j in range(4)],axis=1)
g=r<4*np.median(r); print('A outliers',np.where(~g)[0]);u,q,v=u[g],q[g],v[g];
funcs={'raw':lambda a:a,'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda a:a/(1+a*a),'sqrt':lambda a:a/np.sqrt(1+a*a)}
for j in range(4):
 a,b,t=u[:,j],q[:,j],v[:,j]; z=np.c_[a,b]; out=[]
 for nm,f in funcs.items():
  for k in [.25,.5,.75,1,1.25,1.5,2]:
   features=aff(np.c_[z,f(k*a*b)]); er,c=score(features,t); out.append((er,nm,k,c))
 print(j,sorted(out,key=lambda a:a[0])[:5]);bc=sorted(out,key=lambda a:a[0])[0];e=t-aff(np.c_[z,funcs[bc[1]](bc[2]*a*b)])@bc[3];print('res corr with u',np.corrcoef(np.c_[e,u].T)[0,1:]);
 cand=[]
 for k,l in itertools.combinations_with_replacement(range(4),2):
  for nm,f in funcs.items():
   feat=f(u[:,k]*u[:,l]); cand.append((score(aff(np.c_[z,funcs[bc[1]](bc[2]*a*b),feat]),t)[0],k,l,nm))
 print('extras',sorted(cand)[:6])
for pn in ['panel_b','panel_c']:
 p=s[pn]; u=p['u'];a=aff(u)
 for vn,v in p.items():
  if vn=='u':continue
  print(pn,vn,'range',v.min(0),v.max(0));
  for j,t in enumerate(v.T):
   out=[]
   for nm,f in funcs.items():
    for k in range(1,5):
     def fun(c,A=a):return c[0]+c[1]*f(A@c[2:]);
     p0=np.r_[0, max(.4,max(abs(t)))*k, ls(a,t)/(max(.4,max(abs(t)))*k)];res=least_squares(lambda c:fun(c)-t,p0,max_nfev=250);h=np.sum(np.linalg.svd(res.jac,full_matrices=False)[0][:,:np.linalg.matrix_rank(res.jac)]**2,axis=1);out.append((np.mean((res.fun/np.maximum(1-h,.05))**2)**.5,np.sqrt(np.mean(res.fun**2)),nm,res.x))
   print(j,sorted(out,key=lambda a:a[0])[:3])
