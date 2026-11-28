import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda D,V:np.linalg.lstsq(D,V,rcond=None)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]
W={
 'dead':(lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.22,1.25],([0,.15],[1.,5.])),
 'softsign':(lambda x,p:x/(1+p[0]*abs(x))**p[1],[.22,1.25],([0,.15],[15.,5.])),
 'rational':(lambda x,p:x/(1+p[0]*x*x)**p[1],[.22,1.25],([0,.15],[15.,5.])),
 'tanh':(lambda x,p:np.tanh(p[0]*x),[2.15],([.02],[12.])),
 'sin':(lambda x,p:np.sin(p[0]*x),[2.15],([.02],[12.])),
 'asinh':(lambda x,p:np.arcsinh(p[0]*x),[2.15],([.02],[30.])),
 'atan':(lambda x,p:np.arctan(p[0]*x),[2.15],([.02],[30.])),
 'pow':(lambda x,p:np.sign(x)*abs(x)**p[0],[1.25],([.1],[7.])),
 'xcos':(lambda x,p:x+p[1]*np.cos(p[0]*x),[2.15,.22],([.02,-4.],[12.,4.])),
 'xsin':(lambda x,p:x+p[1]*np.sin(p[0]*x),[2.15,.22],([.05,-4.],[12.,4.])),
 'rect':(lambda x,p:np.maximum(x-p[0],0)**p[1],[.22,1.25],([-1.,.1],[1.,7.])),
 'sat':(lambda x,p:np.clip(x,-p[0],p[0]),[.6],([.05],[1.3])),
}
def get_models(u,v):
 def point(n,f,ini,bounds):
  def res(p):
   D=f(u,p); return (D@fit(D,v)-v).ravel()
  p=least_squares(res,ini,bounds=bounds,max_nfev=180).x
  D=f(u,p); H=D@np.linalg.pinv(D); E=v-H@v; return (np.mean(abs(E)/(1-np.diag(H))[:,None]),np.mean(abs(E)),n,p,lambda x:f(x,p)@fit(D,v),f)
 models=[]
 for n,(w,ini,bo) in W.items():
  for kind in ('pre','post','add'):
   def f(x,p,w=w,kind=kind):
    return aff(w(x,p)) if kind=='pre' else np.c_[aff(x),w(x,p)]
   if kind=='post':
    # Invert monotone output warps where possible, assessed on original scale.
    continue
   models.append(point(n+'_'+kind,f,ini,bo))
 for a,b in [('atan','sin'),('tanh','sin'),('dead','sin'),('rational','sin'),('tanh','xcos'),('dead','xcos')]:
  w,p,bo=W[a]; w2,p2,bo2=W[b]; m=len(p)
  models.append(point(a+'+'+b,lambda x,q,w=w,w2=w2,m=m:np.c_[aff(x),w(x,q[:m]),w2(x,q[m:])],p+p2,[bo[0]+bo2[0],bo[1]+bo2[1]]))
 return sorted(models,key=lambda q:q[0])
if __name__=='__main__':
 for k in ('panel_b','panel_c'):
  d=s[k]
  for n,v in d.items():
   if n=='u': continue
   print(k,n,flush=True)
   for cv,err,name,p,*_ in get_models(d['u'],v)[:12]:print(name,p,cv,err,flush=True)
