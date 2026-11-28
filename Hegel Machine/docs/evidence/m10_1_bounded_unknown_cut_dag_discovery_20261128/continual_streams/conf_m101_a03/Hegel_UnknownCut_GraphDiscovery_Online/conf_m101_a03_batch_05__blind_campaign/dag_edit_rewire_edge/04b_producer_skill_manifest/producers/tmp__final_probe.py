exec(open('/tmp/probe_readout.py').read().split("print('warp'")[0])
from scipy.optimize import least_squares, minimize_scalar
f0,a0,W=b['_fit_panel'](s['panel_b']['u'],s['panel_b']['v'])
idx=np.arange(n); folds=[idx[52:]]+list(np.array_split(np.random.default_rng(327).permutation(n),5))+[np.argsort(np.linalg.norm(x,axis=1))[-14:]]
def reg(A,y): return np.linalg.lstsq(A,y,rcond=None)[0]
for k in range(21):
 E=[]; weights=[]
 for v in folds:
  t=np.setdiff1d(idx,v)
  def tr(p,ret=False):
   f,a=p; u=x+a*np.sin(f*x); A=np.c_[np.ones(n),u]; R=A@reg(A[t],T[t])-T
   return (u,R) if ret else R[t].ravel()
  p=least_squares(tr,[f0,a0],bounds=([1.5,-.8],[4.5,-.1])).x; u,r=tr(p,True); Z=W[0]+u@W[1:]; P=np.c_[np.ones(n),Z,Z[:,0]*Z[:,2],Z[:,1]*Z[:,3]]
  C=Z[:,0]*x[:,1]**2; H=np.tanh(2*(Z[:,1]-Z[:,3])); Q=np.c_[Z[:,0]*Z[:,1],Z[:,0]*Z[:,3],Z[:,1]*Z[:,2],Z[:,2]*Z[:,3],Z**2]; cb=[C,H,Z[:,0]*u[:,1]**2,Z[:,0]*x[:,1]*u[:,1],x[:,0]*x[:,1]*x[:,2]]
  extras=[(),(C,),(H,),(C,H),(cb[2],),(cb[3],),(C,cb[4]),(C,Q),(Q,),(),(C,),(H,),(C,H),(cb[2],),(cb[3],),(C,cb[4]),(C,Q),(Q,),(C,H),(C,H),(C,H)][k]
  B=np.column_stack((P,*extras)); land=x if k<9 else u
  def fit(par,ret=False):
   h,j=par; A=np.c_[B,np.sin(h*land[:,1]-j*land[:,3])];
   if k==18: A=np.c_[A,np.cos(h*land[:,1]-j*land[:,3])]
   if k==19: A=np.c_[A,np.sin(h*x[:,1]-j*x[:,3])]
   if k==20: A=np.c_[A,Z[:,0]*np.sin(h*land[:,1]-j*land[:,3])]
   w=reg(A[t],y[t]); er=A@w-y
   return (np.mean(abs(er[v])),np.std(er[t]),w) if ret else er[t]
  par=min((least_squares(fit,[h,h],bounds=([.4,.3],[5.,5.]),max_nfev=55) for h in [1.2,2.2,3.6]),key=lambda r:r.cost).x
  e,sd,w=fit(par,True); E.append(e); weights.append(np.r_[p,par,sd,w]);
 print(k,np.round(E,4), 'max',round(max(E),4),'par',np.round(np.median(weights,axis=0),3),flush=True)
