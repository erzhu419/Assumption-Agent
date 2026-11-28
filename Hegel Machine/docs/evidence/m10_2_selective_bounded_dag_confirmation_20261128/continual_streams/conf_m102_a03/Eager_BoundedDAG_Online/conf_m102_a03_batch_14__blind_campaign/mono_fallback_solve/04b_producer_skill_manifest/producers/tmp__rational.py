from batch_solver import *
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; z,q=coords(s,t['x']); y=t['y']; T=t['traj']; times=s['eval_times']; I=np.arange(52);J=np.arange(52,72)
B=np.c_[-np.expm1(-times),times]; A=ls(B,T.T).T; Y=np.c_[y,A[:,0]]
L=aff(q); Q=quad(q); Qz=quad(z); X=np.c_[L,Q]
np.set_printoptions(precision=4,suppress=True)
# First determine whether sharing one arbitrary quadratic component is supported at all.
ci=ls(X[I],Y[I]);print('base',np.mean(abs(X@ci-Y),0));
for ln,b in [('q',L),('qz',aff(np.c_[q,z])),('qsin',aff(np.c_[q,np.sin(2.15*q)])),('qZsin',aff(np.c_[q,np.sin(2.15*z)]))]:
 d=np.c_[b,A[:,0]]; c=ls(d[I],y[I]); print('observed-amp',ln,np.mean(abs(d[I]@c-y[I])),np.mean(abs(d[J]@c-y[J])),c)
res=[]
for bname,b in [('q',q),('z',z),('ql',np.c_[q,z])]:
 for mode in ['rquad','rall','rtarg','ralt','squad','tquad','pquad','rden2']:
  if bname=='ql' and mode not in ['rquad','rall','ralt']:continue
  for fix in [False,True]:
   if mode in ['squad','tquad','pquad'] and fix:continue
   h=b.shape[1]+int(not fix); B0=aff(b) if not fix else b
   if mode=='rden2': B0=np.c_[B0,b*b];h=B0.shape[1]
   def design(p):
    if mode in ['squad','tquad','pquad']:
     d=X@p; g={'squad':np.sin,'tquad':np.tanh,'pquad':lambda d:d/(1+d*d)}[mode];return np.c_[L,g(d)]
    d=B0@p; den=1+d*d
    if mode=='rquad':return np.c_[L,Q/den[:,None]]
    if mode=='rall':return X/den[:,None]
    if mode=='rtarg':return np.c_[L,X/den[:,None]]
    if mode=='ralt':return np.c_[X,1/den]
    return np.c_[L,Q/den[:,None]]
   def err(p):
    a=design(p);return (a[I]@ls(a[I],Y[I])-Y[I]).ravel()
   rng=np.random.default_rng(314); starts=[rng.normal(0,k,h) for k in [.15,.45,.8] for j in range(7)] if mode not in ['squad','tquad','pquad'] else [v*ci[:,0]+rng.normal(0,.08,len(X.T)) for v in [.15,.45,.75,1.2,-.6] for j in range(2)];
   r=min((least_squares(err,p,max_nfev=100) for p in starts),key=lambda r:r.cost);a=design(r.x);E=abs(a@ls(a[I],Y[I])-Y);res.append((E[I].mean(0),E[J].mean(0),bname,mode,fix,r.x));
for j in [0,1]:print('rank',j,*sorted(res,key=lambda r:r[j][0])[:12],sep='\n')
