import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task'); from batch_solver import *
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=align(s['panel_a'],tr['x']); w,c,d=features(s,z); T=tr['traj']; y=tr['y']; Y=np.c_[y,T]; n=len(y); a=np.arange(52); h=np.arange(52,72)
def nrm(x):return x/np.maximum(np.std(x,axis=0),.1)
def gram(x):
 x=nrm(x); return x@x.T/x.shape[1]
q=np.cos(2.4*z); p=pair(z)
P=np.column_stack([np.prod(z[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),3)]); print('coefficients fit cubic joint'); cf=ls(one(np.c_[z,p,q,P])[:52],Y[:52]); print(np.round(cf,2));
folds=[(a,h)]+[(i[:52],i[52:]) for i in [np.random.default_rng(k).permutation(n) for k in range(6)]]
def cv(K,lam,fs=folds):
 es=[]
 for i,j in fs:
  D=abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);es.append((D[:,0].mean(),D[:,1:].mean()))
 return np.array(es)
groups={'z':z,'z+w':np.c_[z,w],'z+c+d':np.c_[z,c,d],'z+p+q':np.c_[z,p,q],'w':w,'c+d':np.c_[c,d]}; base=1+gram(z)+gram(p)+gram(q)
mods=[]
for name,X in groups.items():
 X=nrm(X); G=X@X.T/X.shape[1]; ncol=X.shape[1]; ds=(X[:,None]-X[None,:])**2
 for k in [.07,.2,.55,1.5,4,10]:
  U=np.exp(-k*ds); K=U.mean(2); K2=(U.sum(2)**2-(U**2).sum(2))/max(1,ncol*(ncol-1)); K3=(U.sum(2)**3-3*U.sum(2)*(U**2).sum(2)+2*(U**3).sum(2))/max(1,ncol*(ncol-1)*(ncol-2)); R=np.exp(-k*ds.mean(2))
  for form,M in enumerate([K,K+.1*K2,K+K2,K+.5*K2+.15*K3,K2,K2+K3,K3,R,K+.5*K2+.5*R]):
   for pk,N in enumerate([np.ones_like(K),base,(1+.4*G)**3,base+.25*(1+.4*G)**3]):
    for l in [1e-6,3e-5,.001,.025]:
     e=cv(N+M,l); mods.append((e[1:].mean(0),e[0],e.max(0),(name,k,form,pk,l)))
for q0 in [0,1]:
 print('outer-robust best',q0, sorted(mods,key=lambda a:a[0][q0])[:4]); print('fixed best',q0,sorted(mods,key=lambda a:a[1][q0])[:4])
