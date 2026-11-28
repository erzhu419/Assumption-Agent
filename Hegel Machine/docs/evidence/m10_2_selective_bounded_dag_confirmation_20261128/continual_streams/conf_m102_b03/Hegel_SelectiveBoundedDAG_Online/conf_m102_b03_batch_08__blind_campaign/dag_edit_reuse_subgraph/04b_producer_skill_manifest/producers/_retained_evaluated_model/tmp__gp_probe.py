import numpy as np, pickle, importlib.util
from scipy.linalg import svd
spec=importlib.util.spec_from_file_location('a','/tmp/hegel_continual_archive/conf_m102_b03_batch_01_solver.py'); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; n=len(y); z,l,q,r,p,c=a.features(s,s['train']['x']); dot=(q*r).sum(1); cross=(p[:,:,None]*c[:,None,:]).reshape(n,-1); mix=(q[:,:,None]*r[:,None,:]).reshape(n,-1); zp=(z[:,:,None]*p[:,None,:]).reshape(n,-1); zc=(z[:,:,None]*c[:,None,:]).reshape(n,-1)
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
F=[np.arange(52,72),*np.array_split(np.random.default_rng(325).permutation(n),4)]
B=np.c_[np.ones(n),p,c,cross]; Bz=np.c_[B,z]; W=np.c_[y,T]
for i,te in enumerate(F):
 tr=np.setdiff1d(np.arange(n),te); A=B[tr]; E=W[tr]-A@np.linalg.lstsq(A,W[tr],rcond=None)[0]; _,ss,V=np.linalg.svd(E,full_matrices=False); h=V[0]/V[0,0]; coef=np.linalg.lstsq(A,W[tr],rcond=None)[0]; U=B@coef; target=U[:,0].copy(); target[tr]+=(E@h)/(h@h); print('slice',i,'h',np.round(h,3),'sig',np.round(ss[:3],4),flush=True)
 opts=[]
 for mean,BB in [('none',B[:,:1]),('z',np.c_[B[:,:1],z]),('branch',B),('branchz',Bz),('dot',np.c_[B[:,:9],dot])]:
  for coord,XX in [('z',z),('L',l),('qr',np.c_[q,r]),('zqr',np.c_[z,q,r]),('zL',np.c_[z,l])]:
   if i and (mean not in ('branch','branchz','dot') or coord not in ('z','L','zqr')):continue
   X=XX-XX.mean(0); ds=(X[:,None,:]-X[None,:,:])**2; A=BB[tr]; qmat,rmat=np.linalg.qr(A,mode='complete'); null=qmat[:,A.shape[1]:]; v=null.T@target[tr]; Dt=ds[tr][:,tr]; dim=X.shape[1]
   for fam in ['rbf','m52','m32','rq']:
    for ard in [False,True] if coord in ('z','L') else [False]:
     k=dim if ard else 1
     def kernel(d,b):
      d=d@np.exp(-2*np.broadcast_to(b[:k],(dim,))); r=np.sqrt(np.maximum(d,0));
      if fam=='rbf':return np.exp(-.5*d)
      if fam=='m52':return (1+np.sqrt(5)*r+5/3*d)*np.exp(-np.sqrt(5)*r)
      if fam=='m32':return (1+np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r)
      return (1+.5*d/np.exp(b[-1]))**(-np.exp(b[-1]))
     def obj(b):
      K=kernel(Dt,b)+np.exp(b[k])*np.eye(len(tr)); f=cho_factor(null.T@K@null,check_finite=False); sig=v@cho_solve(f,v,check_finite=False)/len(v); return len(v)*np.log(sig)+2*np.log(np.diag(f[0])).sum()+.2*np.sum(b[:k]**2)
     b=min([minimize(obj,[np.log(h0)]*k+[-10]+([0.] if fam=='rq' else []),method='L-BFGS-B',bounds=[(-3,3)]*k+[(-17,-2)]+([(-4,5)] if fam=='rq' else []),options={'maxiter':80}) for h0 in [1.,3.]],key=lambda a:a.fun).x
     K=kernel(Dt,b)+np.exp(b[k])*np.eye(len(tr)); cf=cho_factor(K); AX=cho_solve(cf,A); beta=np.linalg.lstsq(A.T@AX,AX.T@target[tr],rcond=1e-12)[0]; alpha=cho_solve(cf,target[tr]-A@beta); pred=BB[te]@beta+kernel(ds[te][:,tr],b)@alpha; st=U[te,1:]+(pred-U[te,0])[:,None]*h[1:]; er=np.mean(abs(pred-y[te])); et=np.mean(abs(st-T[te])); opts.append((er,et,mean,coord,fam,ard,np.round(b,2).tolist(),np.round(obj(b),2)))
 print('best',sorted(opts)[:14],flush=True)
 print('branch,z',[x for x in opts if x[2:4]==('branch','z')],flush=True)
