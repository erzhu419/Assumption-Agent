import pickle
import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations, combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(*a): return np.column_stack((np.ones(len(a[0])),*a))
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def pairs(x): return np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)])
def calbasis(x,kind):
 if kind=='affine': return D(x)
 if kind=='cubic': return D(x,x**3)
 if kind=='cross': return D(x,pairs(x),x**3)
 if kind=='quadratic': return D(x,pairs(x),x**2)
 return D(x,pairs(x),x**2,x**3)
def calfit(kind):
 p=s['panel_a']; u,v=p['u'],p['v']; g=calbasis(u,kind); ix=np.arange(len(u)); rng=np.random.default_rng(6); best=(np.inf,None)
 for it in range(64):
  k=ix if it==0 else rng.choice(len(u),len(u)-10,replace=False)
  for j in range(6):
   c=fit(g[k],v[k]); r=np.sum((g@c-v)**2,axis=1); k=np.argsort(r)[:45]
  c=fit(g[k],v[k]); loss=np.mean((g[k]@c-v[k])**2)**.5
  if loss<best[0]: best=(loss,c)
 return best
fs={'linear':lambda u,k:D(u),'quadratic':lambda u,k:D(u,pairs(u),u*u),'pairs':lambda u,k:D(pairs(u)),'archpair':lambda u,k:D(pairs(u)[:,[0,2,3,5]]),'square':lambda u,k:D(u,u*u),'cubic':lambda u,k:D(u,u**3),'sin':lambda u,k:D(np.sin(k*u)),'lin_sin':lambda u,k:D(u,np.sin(k*u)),'lin_cos':lambda u,k:D(u,np.cos(k*u)),'cos':lambda u,k:D(np.cos(k*u)),'tanh':lambda u,k:D(np.tanh(k*u)),'oddexp':lambda u,k:D(-np.sign(u)*np.expm1(-k*abs(u))),'shrink':lambda u,k:D(np.sign(u)*np.maximum(abs(u)-k,0)**1.5),'shrink_lin':lambda u,k:D(u,np.sign(u)*np.maximum(abs(u)-k,0)**1.5),'evenrat':lambda u,k:D(u,u*u/np.sqrt(1+k*u*u)),'evenrat2':lambda u,k:D(u,u*u/(1+k*u*u)),'oddrat':lambda u,k:D(u/(1+k*u*u)),'oddroot':lambda u,k:D(u/np.sqrt(1+k*u*u)),'lin_abs':lambda u,k:D(u,abs(u)),'lin_pairs':lambda u,k:D(u,pairs(u)),'pairrat':lambda u,k:D(pairs(u)/(1+k*pairs(u)**2)),'linsat':lambda u,k:D(u,np.tanh(k*u)),'lin_oddexp':lambda u,k:D(u,-np.sign(u)*np.expm1(-k*abs(u)))}
def rank(u,v):
 out=[]
 for name,f in fs.items():
  def loss(k,what=0):
   a=f(u,k); U,S,V=np.linalg.svd(a,full_matrices=False); U=U[:,S>S[0]*1e-11]; r=v-U@(U.T@v); return np.mean((r/(1-(U*U).sum(axis=1))[:,None])**2)**.5 if what==0 else np.mean(r*r)**.5
  grid=np.linspace(.01,.65,25) if 'shrink' in name else np.geomspace(.01,7,32); j=np.argmin([loss(k) for k in grid]); k=minimize_scalar(loss,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded').x
  out.append((loss(k),loss(k,1),name,k))
 return sorted(out)
if __name__=='__main__':
 for kind in ['affine','cubic','quadratic','cross','full']:
  e,c=calfit(kind); print('cal',kind,e); print(np.round(c,4)) if kind=='full' else None
 for pname,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  p=s[pname]; print(pname,key,rank(p['u'],p[key])[:10])
