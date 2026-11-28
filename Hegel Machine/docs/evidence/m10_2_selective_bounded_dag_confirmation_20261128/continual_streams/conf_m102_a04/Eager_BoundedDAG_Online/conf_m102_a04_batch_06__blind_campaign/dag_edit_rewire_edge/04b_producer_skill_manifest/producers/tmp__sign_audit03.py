exec(open('/tmp/explore03.py').read().split('for n in (48')[0]); from scipy.spatial.distance import cdist
z=z[:N]; n=52
m=cdist(z[:n],-z[:n]); np.fill_diagonal(m,np.inf)
print('training-only near-antipode mismatch:')
for k in np.argsort(m.min(1))[:8]:
 j=m[k].argmin()
 if j>k: print(k,j,m[k,j],y[k,0]-y[j,0],np.mean(abs(y[k,1:]-y[j,1:])))
import itertools
for h in range(1,5):
 P=np.column_stack([np.prod(z[:,ix],axis=1) for ix in itertools.combinations_with_replacement(range(4),h)]); print('order',h,'shape',P.shape)
for f in [1,3,5,7,10,15]:
 inds=[ix for h in range(1,5) for ix in itertools.combinations_with_replacement(range(4),h)]; X=np.column_stack([np.ones(N)]+[np.prod(z[:,ix],axis=1) for ix in inds]); d=np.array([0]+[len(ix) for ix in inds]); B=np.linalg.solve(X[:n].T@X[:n]+np.diag(.1*f**d),X[:n].T@y[:n]); print('parity residual',f,[(X[:n,d%2==p]@B[d%2==p]).std(0)[[0,1,8]] for p in [0,1]],abs(X[n:]@B-y[n:]).mean(0)[[0,1,8]])
