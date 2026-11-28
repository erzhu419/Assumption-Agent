exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
np.set_printoptions(precision=4,suppress=True,linewidth=150)
print('u | v2');print(np.c_[u,p['v2']]);print('train x | b | y | traj');print(np.c_[x[:n],b[:n],y,tr[:,[0,3,7]]][:14]);print('condition',np.linalg.svd(p['v2'],compute_uv=False))
def pol(a,k):
 return np.column_stack([np.ones(len(a))]+[np.prod(a[:,ii],axis=1) for h in range(1,k+1) for ii in itertools.combinations_with_replacement(range(a.shape[1]),h)])
def eval(a,z,k):
 w=np.linalg.lstsq(a[:k],z[:k],rcond=None)[0]; err=a@w-z;return np.round(np.mean(abs(err[:k]),axis=0),5),np.round(np.mean(abs(err[k:]),axis=0),5)
for a,z in [(u,p['v2']),(s['panel_b']['u'],s['panel_b']['v'])]:
 for k in (1,2): print('panel poly',k,eval(pol(a,k),z,19))
 w=np.linalg.lstsq(pol(a,2),z,rcond=None)[0];print('poly2 coefficients',w,'residuals',np.sqrt(np.mean((pol(a,2)@w-z)**2,axis=0)))
for q in [x[:n],np.c_[x[:n],b[:n]]]:
 for k in ([1,2,3] if q.shape[1]==4 else [1,2]):
  a=pol(q,k);print('train poly',q.shape[1],k,eval(a,np.c_[y,tr[:,[0,3,7]]],52))
T=s['eval_times']; from scipy.optimize import least_squares
for name,fun,k in [('two exp',lambda z: -np.expm1(-T[:,None]*np.exp(z)),2),('exp + linear',lambda z: np.c_[-np.expm1(-T*np.exp(z[0])),T],1),('exp + constant',lambda z: np.c_[-np.expm1(-T*np.exp(z[0])),np.ones_like(T)],1),('three exp',lambda z: -np.expm1(-T[:,None]*np.exp(z)),3)]:
 def resid(z):
  A=fun(z);return (A@np.linalg.lstsq(A,tr.T,rcond=None)[0]-tr.T).ravel()
 R=least_squares(resid,np.linspace(-1.5,0.8,k),max_nfev=300); print(name,np.exp(R.x),np.sqrt(np.mean(R.fun**2)))
