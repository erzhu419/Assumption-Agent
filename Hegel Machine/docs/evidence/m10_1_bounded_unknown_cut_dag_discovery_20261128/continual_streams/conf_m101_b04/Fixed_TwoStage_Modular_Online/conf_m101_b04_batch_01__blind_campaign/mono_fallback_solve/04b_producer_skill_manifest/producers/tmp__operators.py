import numpy as np
from scipy.optimize import least_squares

def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def lin(x):return np.c_[np.ones(len(x)),x]

def panels(s,verbose=False):
 a=s['panel_a']; U=lin(a['u']); v=a['v']; keep=np.arange(len(U));
 for _ in range(8):
  A=ls(U[keep],v[keep]); r=np.linalg.norm(U@A-v,axis=1); keep=np.argsort(r)[:len(U)-max(1,len(U)//5)]
 keep=r<10*np.median(r); A=ls(U[keep],v[keep]); b=s['panel_b']; B=ls(lin(b['u']),b['v']); c=s['panel_c']; u=c['u']
 acts={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'root':lambda z:z/np.sqrt(1+z*z)}
 def features(x,p,f,degree,linear):
  base=lin(x) if linear else np.ones((len(x),1)); z=f(x*p[:4]+p[4:]); return np.c_[base,x*x,z] if degree==2 else np.c_[base,z]
 models=[]
 for key,degree in [('v1',1),('v2',2)]:
  v=c[key]; best=None
  for name,f in acts.items():
   for linear in ([False,True] if degree==1 else [True]):
    def residual(p):
     z=features(u,p,f,degree,linear); return (z@ls(z,v)-v).ravel()
    opts=[least_squares(residual,np.r_[np.full(4,w),np.zeros(4)],bounds=(np.r_[np.full(4,.06),np.full(4,-1.5)],np.r_[np.full(4,5.),np.full(4,1.5)]),max_nfev=230) for w in [.7,1.4,2.6]]; res=min(opts,key=lambda r:np.linalg.norm(r.fun)); error=np.std(res.fun); z=features(u,res.x,f,degree,linear); coef=ls(z,v)
    if verbose: print(key,name,linear,round(error,7),res.x.round(3),flush=True)
    if best is None or error<best[0]:best=(error,res.x,f,linear,coef)
  _,p,f,linear,coef=best
  models.append(lambda x,p=p,f=f,degree=degree,linear=linear,coef=coef:features(x,p,f,degree,linear)@coef)
 return {'A':lambda x:lin(x)@A,'B':lambda x:lin(x)@B,'C':models[0],'D':models[1]}
if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; panels(s,True)
