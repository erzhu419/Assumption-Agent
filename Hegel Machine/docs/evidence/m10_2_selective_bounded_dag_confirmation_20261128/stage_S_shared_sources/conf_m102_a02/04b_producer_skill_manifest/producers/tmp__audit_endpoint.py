import pickle,itertools
import numpy as np
from scipy.optimize import least_squares
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
x=s['train']['x']; y=s['train']['y']; h=s['train']['traj']; t=s['eval_times']; n=len(x)
a=np.c_[np.ones(n),x]; pairs=list(itertools.combinations_with_replacement(range(4),2)); p=np.column_stack([a]+[x[:,i]*x[:,j] for i,j in pairs]); ix=np.arange(n)
splits=[ix[52:],*np.array_split(np.random.default_rng(555).permutation(ix),4),*[np.argsort(abs(x[:,k]-x[:,k].mean()))[-15:] for k in range(4)],np.argsort(np.linalg.norm(x,axis=1))[-15:]]
for f,j in enumerate(splits):
 i=np.setdiff1d(ix,j); w=np.linalg.lstsq(p[i],y[i],rcond=None)[0]; Q=np.zeros((4,4))
 for c,(u,v) in zip(w[5:],pairs): Q[u,v]=Q[v,u]=c/(1+(u!=v))
 ev,V=np.linalg.eigh(Q); order=np.argsort(-abs(ev)); ev=ev[order]; V=V[:,order]; er=[np.mean(abs(a[j]@np.linalg.lstsq(a[i],y[i],rcond=None)[0]-y[j])),np.mean(abs(p[j]@w-y[j]))]
 for rank in (2,3):
  sig=np.sign(ev[:rank]); L=(V[:,:rank]*np.sqrt(abs(ev[:rank]))).ravel();
  def features(z): return ((x@z.reshape(4,rank))**2)@sig
  def profile(z):
   q=features(z); c=np.linalg.lstsq(a[i],y[i]-q[i],rcond=None)[0]; return a@c+q
  r=least_squares(lambda z:profile(z)[i]-y[i],L,max_nfev=100)
  er.append(np.mean(abs(profile(r.x)[j]-y[j])))
 W=np.linalg.lstsq(a[i],h[i],rcond=None)[0]; vt=np.linalg.svd(a[i]@W,full_matrices=False)[2][:2]; base=a[j]@W@vt.T@vt
 def time_fit(log_rate):
  B=-np.expm1(-t[None,:]*np.exp(log_rate)[:,None]); C=np.linalg.lstsq(B.T,W.T,rcond=None)[0].T; return C@B
 r=least_squares(lambda k:(a[i]@time_fit(k)-h[i]).ravel(),np.log([.3,2]),bounds=(-5,4))
 phys=a[j]@time_fit(r.x); e0=abs(base-h[j]); e1=abs(phys-h[j]); print(f, 'eigenvalues',np.round(ev,5), 'endpoint MAE affine/full/rank2/rank3',np.round(er,6),'trajectory mean',np.round([e0.mean(),e1.mean()],6),'final',np.round([e0[:,-1].mean(),e1[:,-1].mean()],6),'worst horizon',np.round([e0.mean(0).max(),e1.mean(0).max()],6))
