import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def fit(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def maps(s,x):
 a=s['panel_a']; u,v=a['u'],a['v']; keep=np.linalg.norm(v,axis=1)<4
 u,v=u[keep],v[keep]; w=np.roll(u,2,axis=1)
 def cal(p,ret=False):
  h=u+p[0]*np.arctan(p[1]*u); r=np.sin(p[2]*w); hc=h-h.mean(0);rc=r-r.mean(0);vc=v-v.mean(0)
  g0=(hc*vc).sum(0)/(hc**2).sum(0); g1=(hc*rc).sum(0)/(hc**2).sum(0)
  rr=rc-hc*g1; c=(rr*(vc-hc*g0)).sum()/(rr**2).sum(); g=g0-c*g1; b=(v-h*g-c*r).mean(0)
  return (b,g,c) if ret else (b+h*g+c*r-v).ravel()
 p=least_squares(cal,[.26,1.75,2.4],bounds=([0,.3,1],[1,5,4]),gtol=1e-13).x
 b,g,c=cal(p,True)
 def fw(x): return b+g*(x+p[0]*np.arctan(p[1]*x))+c*np.sin(p[2]*np.roll(x,2))
 z=np.array([least_squares(lambda q:fw(q)-r,(r-b)/g,gtol=1e-13).x for r in x]);print('cal',p,b,g,c,np.std(cal(p)),np.max(abs(np.array([fw(q) for q in z])-x)),z.min(),z.max())
 def op(a,k,fn,p,lo,hi):
  u,v=a['u'],a[k]
  def resid(p):
   h=aff(fn(u,p));return (h@fit(h,v)-v).ravel()
  p=least_squares(resid,p,bounds=(lo,hi),gtol=1e-13).x
  return aff(fn(z,p))@fit(aff(fn(u,p)),v)
 b=op(s['panel_b'],'v',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.35,1.5],[0,.2],[.7,4])
 c=op(s['panel_c'],'v1',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.62,1.07,1.57],[0,.1,.1],[5,5,5])
 d=op(s['panel_c'],'v2',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.2,0],[.1,-3],[8,3])
 return dict(z=z,b=b,c=c,d=d)

def run():
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; T=t['traj']; M=maps(s,t['x']); Y=np.c_[y,T]; tr=np.arange(52); te=np.arange(52,72)
 print('svd',np.linalg.svd(T,compute_uv=False));print('ranks',np.linalg.svd(np.c_[y,T],compute_uv=False))
 funcs={**M}
 for name,a in M.items():
  for name2,b in M.items():
   for lam in [.22,.5,1.25,2.15]:
    funcs[f'{name}/{name2}{lam}']=a/(1+lam*b*b)
    funcs[f'{name}*tanh{name2}{lam}']=a*np.tanh(lam*b)
   funcs[f'{name}*{name2}']=a*b
 def score(h):
  h=aff(h); return np.abs(h@fit(h[tr],Y[tr])-Y).mean(0), np.abs(h[te]@fit(h[tr],Y[tr])-Y[te]).mean(0)
 for base in ('','b','c','d','z','bz','bc','bd'):
  scores=[]
  for k,a in funcs.items():
   h=np.column_stack([M[j] for j in base]+[a]); train,test=score(h);scores.append((k,*(float(q) for q in [train[0],train[1:].mean(),test[0],test[1:].mean()])))
  print(base,'Y',sorted(scores,key=lambda q:q[3])[:5]);print(base,'T',sorted(scores,key=lambda q:q[4])[:5])
 np.savez('/tmp/m14.npz',**M,Y=Y,t=s['eval_times'])
if __name__=='__main__':run()
