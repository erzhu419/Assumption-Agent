from h import *
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from itertools import combinations
R=np.linalg.svd(T[:n],full_matrices=False)[2][:3]; Z=np.c_[y,T@R.T];
print('z correlations',np.corrcoef(Z.T)); print('singular joint',np.linalg.svd(Y[:n],compute_uv=False));print('linear y on traj',metrics(A(T@R.T))[:,0]); print('calibrated ranges',q.min(0),q.max(0));
xs={'q':q,'Bq':z,'Cq':C(q),'Dq':D(q),'CBq':C(z),'DBq':D(z),'DCq':D(C(q)),'CDq':C(D(q))};
def eval(f):
 if len(f)!=72:return None
 tr=f[:n]; U,sv,V=np.linalg.svd(tr,full_matrices=False); good=sv>sv[0]*1e-9; U=U[:,good]; sv=sv[good]; V=V[good]; e=U@(U.T@Y[:n])-Y[:n]; h=(U**2).sum(1); l=np.mean(abs(e/np.maximum(1-h,1e-10)[:,None]),0); v=np.mean(abs(f[n:]@(V.T@((U.T@Y[:n])/sv[:,None]))-Y[n:]),0); return np.array([l[0],l[1:].mean(),v[0],v[1:].mean()]);
Gs=[];Fs=[]
for k,p in xs.items():
 Fs.extend([(k,A(p)),(k+'2',poly(p,2))]);
 v=np.c_[p,np.sum(p,1),np.prod(p,1),np.linalg.norm(p,axis=1),*[p[:,i]+sgn*p[:,j] for i,j in combinations(range(4),2) for sgn in (-1,1)],*[p[:,i]*p[:,j] for i,j in combinations(range(4),2)]]; Gs.extend([(k,i,g) for i,g in enumerate(v.T)]);
for k,l in combinations(xs,2):
 a,b0=xs[k],xs[l];Fs.append((k+l,A(np.c_[a,b0])));Fs.append((k+l+'dot',A(np.c_[a,b0,(a*b0).sum(1)])));
for i in range(4): Gs.append(('preBprod',i,q[:,i]*z[:,i]));
out=[]
for k,i,g in Gs:
 for cut in [0.,-.22,.22,.7,1.25,*np.quantile(g[:n],[.25,.50,.75])]:
  h=g>cut
  if min(sum(h[:n]),sum(~h[:n]))<9:continue
  for name,p in Fs:
   if p.shape[1]>9:continue
   f=np.c_[p,h[:,None]*p];l0=eval(f);out.append((l0,(k,i,cut,name,'hard')))
   f=np.c_[p,np.maximum(g-cut,0),np.maximum(g-cut,0)[:,None]*p[:,1:]];l0=eval(f);out.append((l0,(k,i,cut,name,'hinge')))
for j in range(2):
 print('gates',j,sorted(out,key=lambda o:o[0][j])[:5]);
# GPs with coordinate length scales, combinations of panel kernels and polynomial trends.
for kind in ('se','m32','m52'):
 for space in ('q','Bq','qCq','qDq','qCqDq'):
  p=np.c_[tuple(xs[k] for k in ['q','Cq','Dq'] if k in space)] if space!='Bq' else z
  p=(p-p[:n].mean(0))/p[:n].std(0); base=p[:,:4]; d=(p[:,None,:]-p[None,:,:])**2; W=np.c_[y,T[:,-1]]; W=(W-W[:n].mean(0))/W[:n].std(0); m=2
  for mode in ('single','ard','group'):
   k=1 if mode=='single' else p.shape[1] if mode=='ard' else p.shape[1]//4
   def calc(v,pr=False):
    w=np.exp(v[:k]); w=np.repeat(w,4) if mode=='group' else w; dist=np.sqrt(np.maximum(0,d@w)); K=np.exp(-dist**2/2) if kind=='se' else (1+dist)*np.exp(-dist) if kind=='m32' else (1+dist+dist**2/3)*np.exp(-dist);K+=np.exp(v[-3])*(1+base@base.T/4)+np.exp(v[-2])*(1+base@base.T/4)**2
    a=K[:n,:n]+np.exp(v[-1])*np.eye(n); L=np.linalg.cholesky(a); alpha=np.linalg.solve(L.T,np.linalg.solve(L,W[:n]));
    if pr:
     pred=K[n:,:n]@alpha;E=abs(pred-W[n:])*np.c_[y,T[:,-1]][:n].std(0);loo=alpha/np.sum(np.linalg.inv(L)**2,axis=0)[:,None];print(kind,space,mode,'val',E.mean(0),'loo',np.mean(abs(loo),0),'param',np.round(v,2))
    return (alpha*W[:n]).sum()/2+m*np.log(np.diag(L)).sum()+.01*np.sum(v*v)
   v=minimize(calc,np.r_[np.zeros(k),-2,-2,-4],method='L-BFGS-B',bounds=[(-8,5)]*k+[(-12,5)]*3,options={'maxiter':100}).x;calc(v,True)
