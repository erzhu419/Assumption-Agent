import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; n=52

def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def score(d,y,ix=None):
 if ix is None: ix=np.arange(len(d))
 return np.mean(abs(d[ix[n:]]@ls(d[ix[:n]],y[ix[:n]])-y[ix[n:]]))
forms={'sin':(np.sin,.1,15),'cos':(np.cos,.1,15),'tanh':(np.tanh,.1,10),'asinh':(np.arcsinh,.1,10),'atan':(np.arctan,.1,10),'soft':(lambda z:z/np.sqrt(1+z*z),.05,8),'rat':(lambda z:z/(1+z*z),.05,8),'exp':(lambda z:np.expm1(-z*z),.05,8),'gaussodd':(lambda z:z*np.exp(-z*z),.05,8),'sig':(lambda z:np.logaddexp(0,z),.05,8),'sinh':(np.sinh,.05,4)}

def fitpanel(u,v,robust=False):
 out=[]
 for name in ['aff','quad','cube']+list(forms):
  for extra in ([0,1] if name in forms else [0]):
   if name in forms:
    fun,lo,hi=forms[name]
    def df(x,k,fun=fun,extra=extra): return aff(np.c_[x,fun(k*x)] if extra else fun(k*x))
   else:
    def df(x,k,d={'aff':1,'quad':2,'cube':3}[name]): return poly(x,d)
   def calc(k,ret=False):
    a=df(u,k); keep=np.arange(len(u)); w=ls(a,v)
    if robust:
     for _ in range(8):
      keep=np.argsort(np.linalg.norm(a@w-v,axis=1))[:41]; w=ls(a[keep],v[keep])
    a0=a[keep]; err=a0@w-v[keep]; lev=np.sum(a0*np.linalg.pinv(a0).T,axis=1)
    mse=np.mean((err/np.maximum(1-lev[:,None],.01))**2)+1e-7*np.sum(w*w)
    return (mse,np.sqrt(np.mean(err**2)),w) if ret else mse
   k=1.
   if name in forms:
    g=np.linspace(lo,hi,122); m=np.array([calc(i) for i in g]); j=m.argmin()
    k=minimize_scalar(calc,bounds=(g[max(0,j-1)],g[min(len(g)-1,j+1)]),method='bounded').x
   mse,rm,w=calc(k,True); out.append((np.sqrt(mse),rm,name+str(extra),k,lambda x,df=df,k=k,w=w:df(x,k)@w,w))
 return sorted(out,key=lambda i:i[0])

if __name__=='__main__':
 for p,key in [('a','v'),('b','v'),('c','v1'),('c','v2')]:
  q=s['panel_'+p]; r=fitpanel(q['u'],q[key],p=='a'); print(p,key,[i[:4] for i in r[:10]]);print(np.round(r[0][-1],4))
 print('raw',[(d,score(poly(tr['x'],d),tr['y']),score(poly(tr['x'],d),tr['traj'])) for d in (1,2,3)])
