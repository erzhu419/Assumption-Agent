import numpy as np,pickle,importlib.util
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=np.linalg.lstsq
def aff(*xs):return np.column_stack((np.ones(len(xs[0])),*xs))
def poly(u,d):return np.column_stack([u[:,j].prod(1) for i in range(1,d+1) for j in combinations_with_replacement(range(4),i)])
def score(a,v):return np.sqrt(np.mean((a@fit(a,v,rcond=None)[0]-v)**2))
p=s['panel_a'];u,v=p['u'],p['v']; keep=np.max(abs(v),axis=1)<3; print('a kept',keep.sum());
for name,f in [('lin',lambda x,j: aff(x)),('arch',lambda x,j:aff(x[:,j],x[:,(j+1)%4],x[:,j]*x[:,(j+1)%4],x[:,j]**3)),('poly2',lambda x,j:aff(poly(x,2),x[:,j]**3))]:
 print('a',name,[round(score(f(u[keep],j),v[keep,j]),6) for j in range(4)])
fs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'sat':lambda x:np.sign(x)*-np.expm1(-abs(x)),'gauss':lambda x:np.exp(-x*x),'rational':lambda x:x/(1+x*x),'abs':abs,'softsign':lambda x:x/(1+abs(x))}
for p in ('panel_b','panel_c'):
 u=s[p]['u']
 for k,v in s[p].items():
  if k=='u':continue
  res=[]
  for name,a in [('lin',aff(u)),('quad',aff(poly(u,2))),('cyclic',aff(u,u*u,u*np.roll(u,1,axis=1),u**3))]:res.append((score(a,v),name))
  for name,f in fs.items():
   for base in [False,True]:
    def loss(r):return score(aff(u,f(r*u)) if base else aff(f(r*u)),v)
    g=np.linspace(.15,9,150);i=np.argmin([loss(r) for r in g]);r=minimize_scalar(loss,bounds=(g[max(i-1,0)],g[min(i+1,len(g)-1)]),method='bounded').x
    res.append((loss(r),name+(' +lin' if base else ''),r))
  print(p,k,sorted(res)[:8]); print('linear residual per output',np.std(v-aff(u)@fit(aff(u),v,rcond=None)[0],axis=0))
print('trajectory singulars',np.linalg.svd(s['train']['traj'],compute_uv=False))
