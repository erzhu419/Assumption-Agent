exec(open('/tmp/probe_readout.py').read().split("print('warp'")[0])
from scipy.optimize import least_squares
f0,a0,W=b['_fit_panel'](s['panel_b']['u'],s['panel_b']['v'])
print('panel C', [(k,np.shape(v),np.round(np.ravel(v)[:5],3)) for k,v in s['panel_c'].items()])
idx=np.arange(n); folds=[idx[52:]]+list(np.array_split(np.random.default_rng(917).permutation(n),5))+[np.argsort(np.linalg.norm(x,axis=1))[-14:]]
for kind in range(9):
 E=[]; pars=[]
 for v in folds:
  t=np.setdiff1d(idx,v)
  def fit(par,ret=False):
   f,a,h,g=par
   U=x+a*np.sin(f*x); Z=W[0]+U@W[1:]; Q=np.c_[np.ones(n),Z]; P=np.c_[Z[:,0]*Z[:,2], Z[:,1]*Z[:,3]]
   contrast=np.sin(h*(x[:,1]-g*x[:,3])); M=np.c_[Q,P,contrast]; R=Q
   if kind==1: M=np.c_[M,z[:,0]*x[:,1]**2]
   if kind==2: M=np.c_[M,z[:,0]*x[:,1]**2,z]
   if kind==3: M=np.c_[M,z[:,0]*x[:,1]**2,x]
   if kind==4: M=np.c_[Q,P,np.sin(h*(U[:,1]-g*U[:,3]))]
   if kind==5: M=np.c_[Q,P,np.sin(h*(Z[:,1]-g*Z[:,3]))]
   if kind==6: M=np.c_[M,z[:,0]*x[:,1]**2,np.sin(h*x)]
   if kind==7: M=np.c_[Q,P,np.sin(h*x[:,1]),np.sin(h*x[:,3])]
   if kind==8: M=np.c_[M,z[:,0]*x[:,1]**2,np.tanh(h*(Z[:,1]-Z[:,3]))]
   er=M@np.linalg.lstsq(M[t],y[t],rcond=None)[0]-y; tr=R@np.linalg.lstsq(R[t],T[t],rcond=None)[0]-T
   if ret:return [np.mean(abs(er[v])),np.mean(abs(tr[v])),np.std(er[t])]
   return np.r_[er[t]*(1 if kind in (2,3,6,8) else .4), (tr[t]*2).ravel()]
  sol=min((least_squares(fit,[f0,a0,h,1.],bounds=([.8,-1.5,.4,.15],[6.,.3,6.,3.]),max_nfev=90) for h in [1.,2.,3.5]),key=lambda z:z.cost)
  E.append(fit(sol.x,True));pars.append(sol.x)
 print('warp',kind,np.round(E,4).tolist(),'params',np.round(np.median(pars,axis=0),3),flush=True)
