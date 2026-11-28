from probe import *
from scipy.linalg import orthogonal_procrustes

def orth_branch(u,v,i):
 def f(z,p): return p[0]*z+np.tanh(p[1]*z) if i==1 else z+p[0]*z*z
 def profile(p):
  g=f(u,p); m=g.mean(0); g=g-m; vm=v.mean(0)
  rot,sc=orthogonal_procrustes(g,v-vm); mat=rot*(sc/np.sum(g*g)); b=vm-m@mat
  return mat,b
 def residual(p):
  mat,b=profile(p); return (f(u,p)@mat+b-v).ravel()
 p=least_squares(residual,[.45,2.2] if i==1 else [.45],bounds=([.001,.1],[3,5]) if i==1 else (0,2)).x
 mat,b=profile(p); print('orth',i,p,np.linalg.norm(mat[:,0]),b,'panel rms',np.std(residual(p))); return lambda z:f(z,p)@mat+b

tr=s['train']; t=s['eval_times']; z=calibration(s['panel_a'])(tr['x']); pc=s['panel_c']; y=tr['y']; obs=tr['traj']; i=np.arange(len(y)); rng=np.random.default_rng(562)
folds=[(i[:56],i[56:])]+[(np.setdiff1d(i,h),h) for h in np.array_split(rng.permutation(i),5)]
q=np.linalg.norm(z,axis=1); ex=np.argsort(q)[-16:]; folds.append((np.setdiff1d(i,ex),ex))
for branchfun in [branch,orth_branch]:
 a,b=[branchfun(pc['u'],pc['v'+str(j)],j)(z) for j in [1,2]]; dot=np.sum(a*b,axis=1); cross=a[:,0]*b[:,1]; E=af(np.c_[a,b,dot]); S=np.c_[b,cross]; F=np.c_[a,dot]; W=af(np.c_[a,b,dot,cross]);
 def time(r):return -np.expm1(-t[:,None]*r)
 for mode in ['linear','full_time','separate','shared','shared0','shared_no_dot','separate_no_dot']:
  errs=[]; pars=[]
  for u,v in folds+[(i,i)]:
   py=E[v]@fit(E[u],y[u]); o=obs[u]
   if mode=='linear': pred=W[v]@fit(W[u],o)
   else:
    if mode=='full_time': D,G=W,W
    else:
     D=S if mode!='shared0' else af(S)
     G=F if 'no_dot' not in mode else a
     if mode.startswith('separate'):G=np.c_[G,S]
     if mode in ['shared0','separate','separate_no_dot']: G=af(G)
    def design(j,p):
     ts,tf=time(p[:2]).T; ts=ts+(p[2]*tf if len(p)>2 else 0)
     return np.column_stack(((D[j,None,:]*ts[None,:,None]).reshape(len(j)*len(t),-1),(G[j,None,:]*tf[None,:,None]).reshape(len(j)*len(t),-1)))
    def profile(p,j=u):
     X=design(j,p); coef=fit(X,o.ravel()); return X@coef-o.ravel(),coef
    params=[.4,1.8,.12] if mode.startswith('shared') else [.4,1.8]; lo=[.1,1.,-.4][:len(params)]; hi=[.85,3.,.6][:len(params)]
    r=least_squares(lambda p:profile(p)[0],params,bounds=(lo,hi)).x; err,co=profile(r); pars.append(r); pred=(design(v,r)@co).reshape(len(v),len(t))
   er=np.mean(abs(pred-obs[v]),axis=0); errs.append([np.mean(abs(py-y[v])),er.mean(),er[-1],er.max()]);
  print(mode,'56/16',np.round(errs[0],6),'worst',np.round(np.max(errs[:-1],axis=0),6),'insample',np.round(errs[-1],6));
  if mode in ['shared','shared0','separate']:
   print('rates/mix',np.round(pars,5),'training coefficients',np.round(co,5))
