exec(open('/tmp/explore03.py').read().split('for n in (48')[0]); from scipy.spatial.distance import cdist
z=z[:N]; n=52
for k in range(1,5):
 X=np.column_stack([np.ones(N)]+[np.prod(z[:,ix],axis=1) for h in range(1,k+1) for ix in combinations_with_replacement(range(4),h)])
 beta=np.linalg.lstsq(X[:n],y[:n],rcond=1e-9)[0];g=abs(X[n:]@beta-y[n:]); print('signed poly',k,g[:,0].mean(),g[:,1:].mean());
 for deg,h in enumerate([1,5,15,35,70][:k+1]):
  a0=[np.arange(1),np.arange(1,5),np.arange(5,15),np.arange(15,35),np.arange(35,70)][deg];print('deg',deg,'term std',np.std(X[:n,a0]@beta[a0],axis=0)[[0,1,8]])
print('z range and mean', z.min(0),z.max(0),z.mean(0)); print('features oddness algebraically exact:');
for ff in (a._interactions,lambda u:a._curvature(u,.25)): print(np.max(abs(ff(z)-ff(-z))))
m=cdist(z,-z); np.fill_diagonal(m,np.inf);i,j=np.unravel_index(np.argmin(m),m.shape); print('nearest antipodes',i,j,m[i,j],z[i],z[j],y[[i,j]][:,[0,1,8]]);
