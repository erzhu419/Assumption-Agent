import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
from sklearn.linear_model import Lasso,OrthogonalMatchingPursuit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; T=tr['traj']; t=s['eval_times']; y=tr['y']; rng=np.random.default_rng(4)
def design(u):return np.c_[np.ones(len(u)),u]
p=s['panel_a']; h=design(p['u']); v=p['v']; print('panel A row residuals and fits')
for diagonal in (True,False):
 best=None
 for k in range(180):
  ii=rng.choice(len(h),5,replace=False); w=np.zeros((5,4))
  for j in range(4):
   jj=[0,j+1] if diagonal else list(range(5)); w[jj,j]=np.linalg.lstsq(h[ii][:,jj],v[ii,j],rcond=None)[0]
  e=np.linalg.norm(h@w-v,axis=1); val=np.sum(np.minimum(e,.10)**2)
  if best is None or val<best[0]:best=(val,e,w)
 keep=best[1]<.10; w=best[2];
 for j in range(4):
  jj=[0,j+1] if diagonal else list(range(5)); w[jj,j]=np.linalg.lstsq(h[keep][:,jj],v[keep,j],rcond=None)[0]
 print(diagonal,keep.sum(),w.round(5),np.sort(np.linalg.norm(h@w-v,axis=1)).round(4))
 if not diagonal: A=w
p=s['panel_b']; B=np.linalg.lstsq(design(p['u']),p['v'],rcond=None)[0]; print('B singular',np.linalg.svd(B[1:],compute_uv=False))
def basis(r):return -np.expm1(-t[:,None]*np.sort(r))
for k in (1,2,3):
 def er(r):
  q=basis(r); return (np.linalg.lstsq(q,T.T,rcond=None)[0].T@q.T-T).ravel()
 opt=least_squares(er,np.geomspace(.25,2.5,k),bounds=(.015,15),max_nfev=300); print('rates',opt.x,'mae',np.mean(abs(opt.fun)))
 if k==3: q=basis(opt.x)
Y=np.column_stack([y,np.linalg.lstsq(q,T.T,rcond=None)[0].T]); print('Y std',Y.std(0)); print('corr',np.corrcoef(Y.T).round(2))
z=(x-A[0])@np.linalg.inv(A[1:]); coords={'raw':x,'A':z,'AB':design(z)@B,'AiB':(z-B[0])@np.linalg.inv(B[1:]),'B':design(x)@B,'iB':(x-B[0])@np.linalg.inv(B[1:])}; print('ranges',[(a,b.min(0).round(2),b.max(0).round(2)) for a,b in coords.items()]); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(rng.permutation(72),4)]
def lib(z, size):
 terms=[(f'{i}',u) for i,u in enumerate(z.T)]
 for deg in range(2,4):
  terms.extend(('*'.join(map(str,ii)),np.prod(z[:,ii],axis=1)) for ii in it.combinations_with_replacement(range(4),deg))
 if size==0:return terms
 args=terms[:4].copy()
 for i,j in it.combinations(range(4),2):
  u,v=z[:,i],z[:,j]; args.extend([(f'{i}+{j}',u+v),(f'{i}-{j}',u-v)])
  terms.extend([(f'{i}{j}/norm',u*v/(1+u*u+v*v)),(f'{i}{j}/sum',u*v/(1+(u+v)**2)),(f'{i}{j}/diff',u*v/(1+(u-v)**2))])
  for k in (.25,.5,1,2):
   for a,b,p,r in ((i,j,u,v),(j,i,v,u)): terms.append((f'{a}/(1+{k}*{b}^2)',p/(1+k*r*r)))
 for name,u in args:
  for k in (.5,1,1.5,2,3):
   for f in (np.tanh,np.sin,np.cos,np.arctan):terms.append((f'{f.__name__}({k}*{name})',f(k*u)))
  terms.extend([(f'sat{name}',u/(1+abs(u))),(f'abs{name}',abs(u))])
 return terms
for nm,Z in coords.items():
 for size in (0,1):
  terms=lib(Z,size); A0=np.column_stack([v for _,v in terms]); scores=[]
  for method,vals in [('ridge',[.001,.03,.2,1,5]),('omp',[5,9,14,21]),('lasso',[.0005,.002,.008,.03])]:
   for reg in vals:
    err=[]
    for a,b in folds:
     F=(A0-A0[a].mean(0))/np.maximum(A0[a].std(0),1e-9); avg=Y[a].mean(0); r=Y[a]-avg
     if method=='ridge': p=F[b]@F[a].T@np.linalg.solve(F[a]@F[a].T+reg*np.eye(len(a)),r)+avg
     else:
      M=OrthogonalMatchingPursuit(n_nonzero_coefs=reg) if method=='omp' else Lasso(alpha=reg,max_iter=4000,tol=.0003)
      p=np.column_stack([M.fit(F[a],u).predict(F[b]) for u in Y[a].T]);
     e=np.abs(p[:,0]-y[b]).mean(),np.abs(p[:,1:]@q.T-T[b]).mean();err.append(e)
    scores.append((np.mean(err,0).sum(),method,reg,np.round(err[0],4),np.round(np.mean(err[1:],0),4),np.round(np.max(err[1:],0),4)))
  print(nm,size,sorted(scores,key=lambda k:k[0])[:4],flush=True)
