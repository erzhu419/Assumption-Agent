import pickle,numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=[s['panel_a'][k] for k in ('u','v')];n,d=u.shape

def poly(x,k): return np.column_stack([np.ones(len(x))]+[x[:,p].prod(axis=1) for j in range(1,k+1) for p in cwr(range(x.shape[1]),j)])
def fn(x):return np.c_[np.ones(len(x)),x]
def fit(z,v):
 w=np.ones(len(z));
 for j in range(14):
  b=np.linalg.lstsq(z*w[:,None],v*w[:,None],rcond=1e-10)[0]; e=np.linalg.norm(v-z@b,axis=1); w=1/np.sqrt(1+(e/max(.001,np.median(e)*1.5))**2)
 return b,e
funcs={'tanh':np.tanh,'sin':np.sin,'erf':erf,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'cub':lambda x:x**3}
if __name__=='__main__':
 np.set_printoptions(precision=6,suppress=True,linewidth=150)
 k=np.linalg.norm(v-u,axis=1)<2.5;print('cleanish',sum(k)); print('raw deviations',np.round((v-u)[k][:12],3));print('outliers',np.flatnonzero(~k))
 models={}
 for name,f in funcs.items():
  for mode in ['plain','quad','pair_sum','pair_diff','pair_both']:
   def des(x,w):
    a=x if mode in ('plain','quad') else (x+np.roll(x,2,axis=1))[:,:2] if mode=='pair_sum' else (x-np.roll(x,2,axis=1))[:,:2] if mode=='pair_diff' else np.c_[(x+np.roll(x,2,axis=1))[:,:2],(x-np.roll(x,2,axis=1))[:,:2]]
    return np.c_[poly(x,2 if mode=='quad' else 1),f(w*a)]
   def loss(w):
    z=des(u[k],w); b,e=fit(z,v[k]);return np.median(e)+.3*np.quantile(e,.8)
   g=np.linspace(.25,6,35); vals=np.array([loss(w) for w in g]); j=np.argmin(vals); o=minimize_scalar(loss,bounds=(g[max(0,j-1)],g[min(len(g)-1,j+1)]),method='bounded');z=des(u[k],o.x);b,e=fit(z,v[k]);models[name,mode]=(o.fun,o.x,np.quantile(e,[.5,.8,.95,1]),b)
 for nm,(l,w,e,b) in sorted(models.items(),key=lambda p:p[1][0])[:13]:
  print(nm,w,e)
  if l<.025: print(b)
 for deg in [1,2,3]:
  z=poly(u[k],deg);b,e=fit(z,v[k]);print('poly',deg,np.quantile(e,[.5,.8,.9,1]));
  if deg==3:
   for j in range(4):
    p=[()]+[p for a in range(1,4) for p in cwr(range(4),a)]; print(j,sorted(zip(p,np.round(b[:,j],4)),key=lambda a:-abs(a[1]))[:12]);
