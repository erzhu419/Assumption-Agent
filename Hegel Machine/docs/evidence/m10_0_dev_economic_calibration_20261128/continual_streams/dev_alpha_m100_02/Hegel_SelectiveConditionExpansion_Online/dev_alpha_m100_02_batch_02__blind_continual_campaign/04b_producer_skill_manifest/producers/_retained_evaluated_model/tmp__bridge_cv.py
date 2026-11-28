from bridge_probe import x,y,D,design,fit
import numpy as np
from itertools import combinations_with_replacement
terms=[t for k in range(3) for t in combinations_with_replacement(range(4),k)] + [(i,)*3 for i in range(4)]
a=design(x,terms)
def est(mask,mode):
 f=a[mask]; b=y[mask]; p,r=fit(f,b)
 clean = r < max(8*np.median(r),.002)
 f,b=f[clean],b[clean]
 q=np.linalg.lstsq(f,b,rcond=None)[0]
 if mode=='dense': return q
 err=f@q-b
 se=np.sqrt(np.sum(err**2,axis=0)/max(1,len(b)-f.shape[1]))
 se=np.sqrt(np.diag(np.linalg.pinv(f.T@f)))[:,None]*se
 support=np.abs(q)>6*np.maximum(se,1e-8)
 support[:5]=True
 if mode=='sparse':
  for j in range(4):
   k=support[:,j]; q[~k,j]=0; q[k,j]=np.linalg.lstsq(f[:,k],b[:,j],rcond=None)[0]
  return q
 # Physics candidate: output gain multiplies a common cyclic bridge response.
 z=x[mask][clean]; j=np.arange(4); nxt=(j+1)%4; prev=(j-1)%4
 from scipy.optimize import least_squares
 def residual(p):
  intercept,g,common=p[:4],p[4:8],p[8:]; s,t,u,v=common
  return (intercept+g*(z+s*z[:,nxt]+t*z[:,prev]+u*z*z[:,nxt]+v*z**3)-b).ravel()
 p=least_squares(residual,np.r_[q[0],np.diag(q[1:5]),-.3,.15,-.2,.2]).x
 out=np.zeros_like(q)
 for j in range(4):
  i,h=(j+1)%4,(j-1)%4; g=p[4+j]
  out[[0,1+j,1+i,1+h,terms.index(tuple(sorted((j,i)))),terms.index((j,)*3)],j]=np.r_[p[j],g,g*p[8:]]
 return out

def inv(p,b):
 z=(b-p[0])@np.linalg.inv(p[1:5]); jac=np.zeros((len(z),4,4))
 for _ in range(18):
  r=design(z,terms)@p-b; jac[:]=p[1:5].T
  for t,q in zip(terms[5:],p[5:]):
   for k,j in enumerate(t): jac[:,:,j]+=np.prod(z[:,t[:k]+t[k+1:]],axis=1)[:,None]*q
  dz=np.linalg.solve(jac,r[...,None])[...,0]; z-=dz/np.maximum(1,np.max(np.abs(dz),axis=1,keepdims=True))
  if np.max(np.abs(dz))<1e-11:break
 return z
idx=np.arange(len(x)); groups=[idx[i:i+4] for i in range(0,16,4)]+list(np.array_split(idx[16:],8))
for mode in ['dense','sparse','shared']:
 p=est(idx>=0,mode); print(mode,'coeff',[(terms[i],np.round(p[i],6).tolist()) for i in [0,1,5,6,15]], 'train max',np.max(np.linalg.norm(a@p-y,axis=1)[np.linalg.norm(a@p-y,axis=1)<1]))
 for label,splits in [('LOO',[np.array([i]) for i in idx]),('blocked',groups)]:
  e=np.zeros(len(x)); sm=e.copy(); fw=e.copy()
  for g in splits:
   q=est(~np.isin(idx,g),mode); fw[g]=np.linalg.norm(a[g]@q-y[g],axis=1); k=g[fw[g]<1]; delta=inv(q,y[k])-x[k] if len(k) else np.zeros((0,4)); e[k]=np.linalg.norm(delta,axis=1); sm[k]=np.abs(delta.sum(axis=1))
  print(label,'axis/mixed max channel norm',e[:16].max(),e[16:].max(),'max sum',sm.max(),'forward max',fw[fw<1].max())
 print('trajectory ranges and closure',[(part,key, (lambda z:(z.min(),z.max(),np.max(np.abs(design(z,terms)@p-D[part][key].reshape(-1,4)))))(inv(p,D[part][key].reshape(-1,4)))) for part in ['train','test'] for key in ['past_channel_raw','hold_channel_raw']])
