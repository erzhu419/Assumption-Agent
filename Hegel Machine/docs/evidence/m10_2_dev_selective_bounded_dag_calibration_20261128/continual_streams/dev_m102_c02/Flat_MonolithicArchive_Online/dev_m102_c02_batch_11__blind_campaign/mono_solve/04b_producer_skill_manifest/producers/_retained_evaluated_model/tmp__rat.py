import numpy as np,pickle,itertools
from scipy.linalg import svd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; A=np.load('/tmp/composed.npz'); P=dict(A); B=np.load('/tmp/cached.npz'); P.update(cs=B['panel_cv1soft'],dd=B['panel_cv2dead']); n=52
mi,mj=np.triu_indices(4)
F={}; pairs={}; norms={}
for k,x in P.items():
 F[k]=x; F[k+'q']=x[:,mi]*x[:,mj]; F[k+'cub']=x**3
 pairs[k]=np.c_[x[:,0]*x[:,2]+x[:,1]*x[:,3],x[:,0]*x[:,2]-x[:,1]*x[:,3]]
 norms[k]=np.c_[(x*x).sum(1),x*x,(x[:,0]+x[:,2])**2,(x[:,1]+x[:,3])**2,(x[:,0]-x[:,2])**2,(x[:,1]-x[:,3])**2]
 F[k+'p']=pairs[k][:,:1]; F[k+'pm']=pairs[k]
for k,l in itertools.product(P,P):
 if k!=l:F[k+'m'+l]=P[k]*P[l]
for k,l in itertools.product(P,P):
 for mode in [0,1,2]: F[k+'r'+l+str(mode)]=P[k]/(1+.22*np.roll(P[l],mode,axis=1)**2)
for k in P:
 for l in P:
  F[k+'d'+l]=pairs[k][:,:1]/(1+.22*norms[l]); F[k+'Q'+l]=F[k+'q']/(1+.22*norms[l][:,:1]); F[k+'s'+l]=P[k]/(1+.22*norms[l][:,:1]); F[k+'t'+l]=pairs[k][:,:1]/(1+.22*pairs[l]**2)
F['const']=np.ones((len(y),1));
def score(seq):
 h=np.column_stack([F[k] for k in seq]); X=h[:n]; m=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1; h=(h-m)/sd; U,S,V=svd(h[:n],full_matrices=False); a=U.T@(y[:n]-y[:n].mean()); Q=S[:,None]**2; L=np.array([.0001,.01,.10,.5,1.5,4]); g=Q/(Q+L); er=y[:n,None]-y[:n].mean()-U@(a[:,None]*g); E=np.mean(abs(er/np.maximum(1-1/n-(U*U)@g,1e-8)),0); j=E.argmin(); pred=y[:n].mean()+h[n:]@(V.T@(a*S/(S*S+L[j]))); return E[j],np.mean(abs(pred-y[n:])),L[j],np.std(er[:,j])
beam=[('z',)]; seen=set(beam); out=[]
for dep in range(5):
 R=[]
 for seq in beam:
  for k in F:
   q=tuple(sorted(set(seq+(k,))));
   if q in seen:continue
   seen.add(q); R.append((score(q),q))
 R.sort(); out+=R[:25]; beam=[]
 for stat,q in R:
  if any(abs(stat[0]-old[0][0])<1e-8 for old in beam):continue
  beam.append((stat,q))
  if len(beam)==35:break
 beam=[q for _,q in beam]; print('depth',dep,*R[:3],flush=True)
print(*sorted(out)[::2][:25],sep='\n'); print('low CV',*sorted(out,key=lambda q:q[0][1])[:8],sep='\n')
