import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def quad(x):return np.array([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]).T
def fit(f,y):
 w=ls(f,y); e=f@w-y; h=(np.linalg.svd(f,full_matrices=False)[0]**2).sum(1); return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2))
def panels():
 for key,vkey in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u=s[key]['u']; v=s[key][vkey]; a=aff(u); q=quad(u);
  print(key,vkey)
  cand={}
  for n,f in [('identity',u),('cube',u**3),('pair',u*np.roll(u,-1,axis=1)),('square',u**2),('sinpair',np.sin(u+np.roll(u,-1,axis=1)))]+[(f'{kind}{w}',fun(w*u)) for kind,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)] for w in [0.5,1,1.5,2,2.5,3,3.5,4,5,6]]:
   for nq,base in [('a',a),('q',np.c_[a,q])]:
    if n in ['identity','square','pair'] and nq=='q':continue
    F=base if n=='identity' else np.c_[base,f]; cand[n,nq]=fit(F,v)
   cand[n,'direct']=fit(aff(f),v)
  print('basis LOO',sorted(cand.items(),key=lambda t:t[1][1])[:14])
  # Sparse terms per output, with a shared harmonic frequency if supported.
  for j in range(4):
   ans=[]
   cand2=[(f'q{i}',q[:,i]) for i in range(10)]+[(f'c{i}',u[:,i]**3) for i in range(4)]+[(f'{name}{i}:{w}',fun(w*u[:,i])) for i in range(4) for w in (1,1.5,2,2.5,3,3.5,4) for name,fun in [('s',np.sin),('c',np.cos),('t',np.tanh)]]
   for n,f in cand2: ans.append((fit(np.c_[a,f],v[:,j:j+1]),n))
   err=np.std(v[:,j]-a@ls(a,v[:,j])); print(j,err,sorted(ans,key=lambda t:t[0][1])[:4]);
  # Single-index output nonlinearities, before versus after a dense affine mix.
  for kind,fun in [('sin',np.sin),('tanh',np.tanh),('square',lambda z:z*z),('cubic',lambda z:z**3),('rational',lambda z:z/(1+z*z))]:
   vals=[]
   for y in v.T:
    def r(w):
     f=np.c_[a,fun(a@w)]; return f@ls(f,y)-y
    rng=np.random.default_rng(4); seeds=[ls(a,y)]+list(rng.normal(size=(12,5))); vals.append(min(np.std(least_squares(r,p,max_nfev=120).fun) for p in seeds));
   print('index',kind,np.round(vals,5))
if __name__=='__main__':panels()
