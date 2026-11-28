exec(open('/tmp/dag.py').read().split('ops=')[0])
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve
M=ts@np.linalg.pinv(H.T); Z=np.c_[y,M];print('corr outputs',np.corrcoef(np.c_[y,ts,M].T)[[0,-3,-2,-1]].round(3)); print('amplitude stats',M.mean(0),M.std(0)); radi=euclidean_distances(z); np.fill_diagonal(radi,np.inf); k=np.argsort(radi.min(1))[:16]; j=radi.argmin(1)[k];print('nearest',np.c_[radi[k,j],z[k],Z[k],Z[j]].round(3));
for name,w in [('z',z),('z+C',np.c_[z,Q(z),R(z)]),('z+B+C(B)',np.c_[z,Q(L(z)),R(L(z))])]:
 d=euclidean_distances(w,squared=True); best=[]
 for p in [.5,1,1.5,2]:
  for h in np.geomspace(.3,8,9):
   for lam in [1e-6,.001,.01,.1]:
    for form in [0,1]:
     K=(np.exp(-(d/h**2)**(p/2)) if not form else (1+d/h**2)**(-p)); F=K[:52,:52]+lam*np.eye(52); wgt=solve(F,np.c_[y,ts][:52],assume_a='pos'); err=abs(K[52:,:52]@wgt-np.c_[y,ts][52:]); best.append((err[:,0].mean(),err[:,1:].mean(),p,h,lam,form))
 print('kernel',name,sorted(best)[:3]);
for w in [z,L(z)]:
 for d in range(1,6):
  F=PolynomialFeatures(d).fit_transform(w); b=ls(F[:52],np.c_[y,ts][:52]); e=abs(F@b-np.c_[y,ts]); print('poly',d,e[:52,0].mean(),e[52:,0].mean(),e[52:,1:].mean())
