from panel import *
from itertools import product
from scipy.optimize import root
funcs.update(cube=lambda x:x**3,soft=lambda x:np.logaddexp(x,-x),id=lambda x:x)
keys=['sin','tanh','erf','asinh','atan','rat','sat','cube']
def cal(panel,verbose=False):
 u,v=panel['u'],panel['v']; keep=np.abs(v).max(1)<3; u,v=u[keep],v[keep]; models=[]
 for j in range(4):
  k=(j+2)%4; y=v[:,j]; scores=[]
  for a,b in product(keys,repeat=2):
   def D(w,p,a=a,b=b,j=j,k=k):return I(np.c_[w,funcs[a](w[:,j]*np.exp(p[0])),funcs[b](w[:,k]*np.exp(p[1]))])
   def R(p):
    d=D(u,p);return d@ls(d,y)-y
   r=min((least_squares(R,np.log(p),bounds=(-2.5,1.6),max_nfev=45) for p in [(1,1),(2.15,2.15)]),key=lambda z:z.cost)
   scores.append((np.sqrt(2*r.cost/len(y)),a,b,np.exp(r.x),D,r.x,ls(D(u,r.x),y)))
  scores.sort(key=lambda z:z[0]); e,a,b,p,d,q,c=scores[0]; models.append((d,q,c));
  if verbose: print('A',j,[(v[:4],v[-1]) for v in scores[:3]])
 def forward(w):return np.column_stack([d(w,q)@c for d,q,c in models])
 c=ls(I(u),v)
 def inv(x):return np.array([root(lambda w:forward(w[None,:])[0]-y,np.linalg.solve(c[1:].T,y-c[0])).x for y in x])
 return inv,forward

def cfit(panel,verbose=False):
 u,v=panel['u'],panel['v1']; results=[]
 for a,b in product(keys, ['id','sin','asinh','cube','tanh','rat','sqrt','cos','sat']):
  if a==b:continue
  def D(w,p,a=a,b=b):return I(np.c_[w,funcs[a](w*np.exp(p[0])),funcs[b](w*np.exp(p[1]))] if b!='id' else np.c_[w,funcs[a](w*np.exp(p[0]))])
  def R(p):
   d=D(u,p);return (d@ls(d,v)-v).ravel()
  r=min((least_squares(R,np.log(p),bounds=(-2.5,1.7),max_nfev=55) for p in [(2.15,1),(1,2.15),(3.5,2.15)]),key=lambda r:r.cost)
  c=ls(D(u,r.x),v); results.append((np.sqrt(2*r.cost/v.size),a,b,np.exp(r.x),lambda w,D=D,p=r.x,c=c:D(w,p)@c,c))
 results.sort(key=lambda z:z[0]);
 if verbose:
  for z in results[:10]:print('C1',z[:4]);
  print(results[0][-1])
 return results[0][-2]
if __name__=='__main__':
 a,f=cal(s['panel_a'],True); x=np.r_[s['train']['x'],s['test']['x']]; z=a(x);print('inversion',np.max(abs(f(z)-x)),z.min(0),z.max(0)); cfit(s['panel_c'],True)
