import pickle,numpy as np
from scipy.optimize import least_squares, root
np.set_printoptions(precision=5,suppress=True,linewidth=140)
d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw']; j=np.arange(4); src=[2,3,0,1]; good=np.max(abs(Y-X),axis=1)<2
D=lambda z:z+np.sin(2.5*z)/2.5
B=lambda z:np.cos(2*z)-1
A=lambda x,k:np.c_[np.ones(len(x)),D(x[:,k]),x[:,src[k]],B(x[:,src[k]])]
p=np.array([np.linalg.lstsq(A(X[good],k),Y[good,k],rcond=None)[0] for k in j]); print('compact coefficients',p,'normalized',p[:,2:]/p[:,1,None]); print('clean absolute residual quantiles',np.quantile(abs(np.column_stack([A(X,k)@p[k] for k in j])-Y)[good],[.5,.95,1]))
print('CV full axis and mixed slices:')
for v in [np.where(good & (np.arange(len(X))<16))[0],*np.array_split(np.random.default_rng(63).permutation(np.where(good & (np.arange(len(X))>=16))[0]),5)]:
 t=np.where(good & ~np.isin(np.arange(len(X)),v))[0]; e=np.column_stack([A(X[v],k)@np.linalg.lstsq(A(X[t],k),Y[t,k],rcond=None)[0] for k in j])-Y[v]; print(len(v),np.max(abs(e)),np.sqrt(np.mean(e**2)))
for s in ('train','test'):
 for key in ('past_channel_raw','hold_channel_raw'):
  y=d[s][key].reshape(-1,4); approx=np.empty_like(y); mult=[]
  for pair in ([0,2],[1,3]):
   w=p[pair]; u=y[:,pair]; sol=[]
   for i,b in enumerate(u):
    def f(q): return w[:,0]+w[:,1]*D(q)+w[:,2]*q[::-1]+w[:,3]*B(q[::-1])-b
    def jac(q): return np.diag(w[:,1]*(1+np.cos(2.5*q)))+np.diag(w[:,2]-2*w[:,3]*np.sin(2*q[::-1]))[:,::-1]
    roots=[]
    for seed in ([0.,0.],(b-w[:,0])/(w[:,1]+w[:,2]), [.7,1.5],[1.5,.7],[1.5,1.5],[-1.5,-1.5],[-1.5,0.],[0.,-1.5],[-1.5,1.5],[1.5,-1.5]):
     r=root(f,seed,jac=jac); q=r.x
     if max(abs(f(q)))<1e-8 and all(max(abs(q-prev))>1e-4 for prev in roots): roots.append(q)
    if not roots:
     r=least_squares(f,[0.,0.],jac=jac,gtol=1e-14); roots=[r.x]; print('fallback',b,r.fun)
    if len(roots)>1: mult.append((pair,i,np.array(roots)))
    sol.append(roots[0])
   approx[:,pair]=sol
  print(s,key,'ranges',np.min(approx,axis=0),np.max(approx,axis=0),'ambiguous',len(mult),'first',mult[:5]); print('derived load range/correlation',np.quantile(approx,[.01,.25,.5,.75,.99],axis=0),np.corrcoef(approx.T))
print('public keys',d.keys()); print('asset metadata excerpt',open('public_asset_index.json').read()[:1200])
