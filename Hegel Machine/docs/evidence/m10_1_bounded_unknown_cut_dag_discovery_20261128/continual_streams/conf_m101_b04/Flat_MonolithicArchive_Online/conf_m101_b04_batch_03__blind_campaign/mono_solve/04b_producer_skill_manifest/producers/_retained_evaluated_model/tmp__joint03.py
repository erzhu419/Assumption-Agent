from operators03 import *
import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _forecast,_panel_b
u,v=(S['panel_b'][k] for k in ('u','v'));x=S['train']['x'];y=S['train']['y'];tr=S['train']['traj'];t=S['eval_times'];z=_panel_b(S['panel_b'],x)
# A common warp implies proportional linear and odd coefficients and does not require separate frequencies.
def warp(x,p):return x+p[0]*np.sin(p[1]*x)
def shape(x,p,k):
 w=warp(x,p)
 return np.c_[aff(w),*[f for f in [x**2,np.cos(p[1]*x),x*np.roll(x,1,axis=1)] if k & (1<<[x.size,x.size+1,x.size+2].index(x.size) )]] if False else np.c_[aff(w),*([x*x] if k&1 else []),*([np.cos(p[1]*x)] if k&2 else []),*([x*np.roll(x,1,axis=1)] if k&4 else [])]
for k in (0,1,2,4,5,6):
 def res(p):
  a=shape(u,p,k);return (a@fit(a,v)-v).ravel()
 r=least_squares(res,[-.4,2.8],bounds=([-1,1],[-.02,5]));a=shape(u,r.x,k);e=r.fun.reshape(v.shape);h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,1)
 print('B warp',k,np.round(r.x,5),round((e**2).mean()**.5,6),round(((e/(1-h[:,None]))**2).mean()**.5,6))
rng=np.random.default_rng(516);splits=[np.arange(72),*[rng.permutation(72) for _ in range(9)]]; ds=cdist(x,x,'sqeuclidean');dz=cdist(z,z,'sqeuclidean');tasks=[]
for ix in splits:
 a,b=ix[:52],ix[52:];Y,B=relax(tr[a],t);tasks.append((a,b,Y,B))
base={'quadB':quad(z),'affXB':np.c_[aff(z),x], 'quadB+X':np.c_[quad(z),x], 'quadB+quadX':np.c_[quad(z),quad(x)[:,1:]],'quadB+diagXZ':np.c_[quad(z),x,x*z], 'quadjoint':quad(np.c_[x,z]) if False else np.c_[quad(z),quad(x)[:,1:],(x[:,:,None]*z[:,None,:]).reshape(len(x),16)]}
for j in (0,1):
 scores=[]
 for nm,p in base.items():
  for l,alpha,w in itertools.product([.035,.075,.15,.3],[.0003,.002,.015],[0.,.25,1.]):
   d=ds+w*dz;rec=[]
   for a,b,Y,B in tasks:
    pred=_forecast(d[np.ix_(a,a)],d[np.ix_(b,a)],p[a],p[b],y[a] if j==0 else Y,l,alpha)
    rec.append(abs(pred-y[b]).mean() if j==0 else abs(pred@B.T-tr[b]).mean())
   scores.append((np.mean(rec),max(rec),rec[0],nm,l,alpha,w))
 print('target',j)
 for r in sorted(scores)[:12]:print(tuple(round(o,5) if isinstance(o,float) else o for o in r))
