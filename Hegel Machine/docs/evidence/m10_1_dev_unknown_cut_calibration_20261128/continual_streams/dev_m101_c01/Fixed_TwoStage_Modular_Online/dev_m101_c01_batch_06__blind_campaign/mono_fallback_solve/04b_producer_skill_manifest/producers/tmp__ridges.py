from joint import *
from scipy.linalg import eigh

def act(a,kind):
 if kind=='sin': return np.sin(a)
 if kind=='tanh': return np.tanh(a)
 if kind=='soft': return a/(1+abs(a))
 if kind=='cos': return np.cos(a)
 if kind=='recip': return 1/(1+a*a)
 return a*np.exp(-a*a/2)

def fit_ridges(x,Y,q,kind,count,reg=.025):
 n,d=x.shape; D=design(x,2); T=design(q,2); Y=np.asarray(Y).reshape(n,-1)
 base=np.linalg.lstsq(D,Y,rcond=None)[0]; R=Y-D@base
 rng=np.random.default_rng(882); eye=np.eye(D.shape[1]+count); scales=np.r_[np.full(D.shape[1],.005),np.full(count,reg)]
 def f(V,p):
  P=p.reshape(count,d+1); return act(V@P[:,1:].T+P[:,0],kind)
 def fun(p,ret=False):
  F=np.c_[D,f(x,p)]; W=np.linalg.lstsq(np.r_[F,scales*eye],np.r_[Y,np.zeros((len(eye),Y.shape[1]))],rcond=None)[0]
  if ret:return np.c_[T,f(q,p)]@W
  return np.r_[(F@W-Y).ravel(),(scales[:,None]*W).ravel(),.004*p]
 starts=[rng.normal(0,scale,(count,d+1)).ravel() for scale in [0.6,1.0,1.5,1.8] for _ in range(3)]
 fs=sorted((least_squares(fun,p,bounds=(-5,5),max_nfev=110,ftol=1e-6) for p in starts),key=lambda f:f.cost)
 return np.mean([fun(f.x,True) for f in fs[:2]],axis=0),np.sqrt(np.mean(fs[0].fun[:Y.size]**2)),fs[0].x

if __name__=='__main__':
 for i,j in ss:
  r,a,b=amplitude(t,z[i]); R=z[i]-a[:,None]*b; v=np.linalg.svd(R,full_matrices=False)[2][0]
  for kind in ['sin','tanh','soft','cos','recip','gauss']:
   for k in [1,2,3]:
    pred,err,pars=fit_ridges(x[i],y[i],x[j],kind,k); print(kind,k,round(err,5),round(abs(pred[:,0]-y[j]).mean(),5),np.round(pars,2) if i[0]==0 and len(j) and j[0]==56 else '',end='; ')
   print(flush=True)
  print('---',flush=True)
