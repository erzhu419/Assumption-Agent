import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; ix=list(itertools.combinations_with_replacement(range(4),2))
Q=np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in ix]])
A=Q[:,:5]; r=np.random.default_rng(293); N=np.arange(len(x)); splits=[(N[:52],N[52:])]+[(np.setdiff1d(N,v),v) for v in np.array_split(r.permutation(len(x)),4)]
links={'tanh':np.tanh,'sin':np.sin,'rational':lambda a:a/(1+a*a),'softsign':lambda a:a/np.sqrt(1+a*a),'atan':np.arctan}
for target in ('scalar','trajectory'):
 for label,phi in links.items():
  for bn,ln in ((5,15),(15,5),(15,15),(1,15)):
   errors=[]; fits=[]
   for i,j in splits:
    if target=='scalar': Y=y[i]
    else:
     _,_,V=np.linalg.svd(T[i],full_matrices=False); Y=T[i]@V[0]
    B=Q[i,:bn]; L=Q[i,:ln]; Bp=np.linalg.pinv(B); Z=Y-B@(Bp@Y); slope=np.linalg.lstsq(L,Y,rcond=None)[0]; pairs=[]
    for k in range(8):
     p=np.r_[1.5,np.zeros(ln)]; p[1:]=slope/(1+np.std(Y)) if not k else r.normal(0,.55,ln)
     def res(p):
      g=p[0]*phi(L@p[1:]); return np.r_[g-B@(Bp@g)-Z, .002*p]
     opt=least_squares(res,p,max_nfev=200); pairs.append((np.linalg.norm(opt.fun),opt.x))
    err,p=min(pairs,key=lambda x:x[0]); b=Bp@(Y-p[0]*phi(L@p[1:])); pred=Q[j,:bn]@b+p[0]*phi(Q[j,:ln]@p[1:]); fits.append(err/len(i)**.5)
    if target=='trajectory': pred=pred[:,None]*V[0]+ Q[j]@np.linalg.lstsq(Q[i],T[i]-Y[:,None]*V[0],rcond=None)[0]
    errors.append(abs(pred-(y[j] if target=='scalar' else T[j])).mean())
   print(target,label,bn,ln,'fit',round(np.mean(fits),4),'errors',np.round(errors,4),flush=True)
