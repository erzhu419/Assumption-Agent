exec(open('/tmp/spatial.py').read().split('coords=')[0])
from scipy.optimize import minimize_scalar
from scipy.special import erf
terms=[j for i in range(1,4) for j in comb(range(4),i)]
links={'soft':lambda q,k:q/np.sqrt(1+k*q*q),'rat':lambda q,k:q/(1+k*q*q),'tanh':lambda q,k:np.tanh(np.sqrt(k)*q)/np.sqrt(k),'sin':lambda q,k:np.sin(np.sqrt(k)*q)/np.sqrt(k),'atan':lambda q,k:np.arctan(np.sqrt(k)*q)/np.sqrt(k),'erf':lambda q,k:erf(np.sqrt(k)*q)/(np.sqrt(k)*2/np.sqrt(np.pi))}
def G(q,name,k): return poly(links[name](q,k),3)
def pred(q,name,k,i,j,R,alpha):
 f=G(q,name,k); h=f[i]; pen=alpha*np.eye(h.shape[1]); return f[j]@solve(h.T@h+pen,h.T@R[i],assume_a='pos')
def one(q,name,group,i,j,alpha,fit_train=False):
 R=Y[:,group]; R=R if R.ndim==2 else R[:,None]
 if fit_train:
  def obj(k):
   h=G(q,name,k)[i]; w=solve(h.T@h+alpha*np.eye(h.shape[1]),h.T@R[i],assume_a='pos'); return np.mean((h@w-R[i])**2)
 else:
  def obj(k):
   h=G(q,name,k)[i]; K=h@h.T; P=np.linalg.inv(K+alpha*np.eye(len(i))); return np.mean((P@R[i]/P.diagonal()[:,None])**2)
 grid=np.geomspace(.008,3,28); vals=[obj(k) for k in grid]; m=np.argmin(vals); k=minimize_scalar(obj,bounds=(grid[max(0,m-1)],grid[min(len(grid)-1,m+1)]),method='bounded').x
 return k,np.abs(pred(q,name,k,i,j,R,alpha)-R[j]).mean(),obj(k)
res=[]
for cname,q in [('raw',x),('cal',xc)]:
 for name in links:
  for mode in [False,True]:
   for alpha in [1e-7,1e-4,.002,.02]:
    for group,gname in [(0,'y'),(slice(1,None),'traj')]:
     r=np.array([one(q,name,group,i,j,alpha,mode) for i,j in ss]); res.append((gname,r[:,1].max(),r[:,1].mean(),r[0,1],cname,name,mode,alpha,np.round(r[:,0],4),r[:,2].mean()))
for name in ['y','traj']:
 for sortkey in [1,3,9]:
  print(name,sortkey); print(*sorted((r for r in res if r[0]==name),key=lambda r:r[sortkey])[:7],sep='\n')
