from probe import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
Y=np.c_[y,T]; n=len(x); a=np.arange(n); folds=[np.arange(56,72),*np.array_split(a,4),a[np.argsort(np.linalg.norm(x,axis=1))[-14:]]]
for k in ['panel_a','panel_b']:
 p=s[k]; w=affine(p); e=np.linalg.norm(poly(p['u'],1)@w-p['v'],axis=1); print(k,w,'row-error percentiles',np.quantile(e,[0,.5,.8,.95,1]))
D=poly(x,2); ec=Y-D@np.linalg.lstsq(D,Y,rcond=None)[0]; U,S,V=np.linalg.svd(ec,full_matrices=False); print('joint quadratic residual modes',S[:4],V[:3]); print('amplitude correlations',np.corrcoef(np.c_[y,T@np.linalg.pinv(timefit(T).T)].T)); print('norm ranges',np.quantile(np.linalg.norm(x,axis=1),[0,.5,.8,1]))
K0=D@D.T; Ks={};
for name,z in {'x':x,'a':za,'b':x@B[1:]+B[0],'ab':za@B[1:]+B[0],'bi':zb}.items():
 z=(z-z.mean(0))/z.std(0); G=z@z.T/z.shape[1]; dd=cdist(z,z,'sqeuclidean');
 for q in [3,4,5,6]:
  Ks[(name,'hom'+str(q))]=G**q; Ks[(name,'poly'+str(q))]=(1+G)**q
 for l in [.6,1.,1.6,2.5,4.]:
  d=np.sqrt(dd)/l; Ks[(name,'rbf'+str(l))]=np.exp(-d*d/2); Ks[(name,'mat'+str(l))]=(1+d+d*d/3)*np.exp(-d)
 for l in [.8,1.6,3.]:
  W=np.exp(-((z[:,None,:]-z[None,:,:])/l)**2/2); Ks[(name,'add'+str(l))]=W.sum(2); Ks[(name,'pair'+str(l))]=sum(W[:,:,i]*W[:,:,j] for i in range(4) for j in range(i,4))
for j in range(4):
 for k in range(j+1,4):
  for v in [.5,1,2]:
   z=np.column_stack([x[:,j]+sg*v*x[:,k] for sg in [-1,1]]); Ks[('x',f'sumpair{j}{k}/{v}')]=sum((1+np.outer(zi,zi))**5 for zi in z.T)
results=[]
for name,F in Ks.items():
 F=F/np.mean(np.diag(F)); K=K0+F
 for reg in [1e-6,3e-5,.0003,.003,.03,.3]:
  E=[]
  for jj in folds:
   ii=np.setdiff1d(a,jj); w=solve(K[np.ix_(ii,ii)]+reg*np.eye(len(ii)),Y[ii],assume_a='pos'); pred=K[np.ix_(jj,ii)]@w; Vt=np.linalg.svd(T[ii],full_matrices=False)[2][:2]; pred[:,1:]=(pred[:,1:]@Vt.T)@Vt; e=np.abs(pred-Y[jj]); E.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
  E=np.array(E); results.append((E.mean(0),E.max(0),E[0],E[-1],name,reg))
for col in range(3):
 print('criterion',col)
 for r in sorted(results,key=lambda v:(v[0][col]+v[1][col])*.5)[:9]: print(r)
