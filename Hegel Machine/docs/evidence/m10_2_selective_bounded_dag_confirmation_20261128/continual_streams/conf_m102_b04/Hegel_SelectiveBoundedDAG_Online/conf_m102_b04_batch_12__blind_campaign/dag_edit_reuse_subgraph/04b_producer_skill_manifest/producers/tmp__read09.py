from probe09 import *
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def op(u,v,f,k,lo,hi):
 def res(k):
  a=f(u,k);return (a@ls(a,v)-v).ravel()
 p=least_squares(res,k,bounds=(lo,hi),max_nfev=300,gtol=1e-12).x;c=ls(f(u,p),v);return np.mean(res(p)**2)**.5,p,c,f(q,p)@c
out={}
for pn,vn,w0,name in [('panel_b','v',2.6,'B'),('panel_c','v2',1.3,'W')]:
 u=s[pn]['u'];v=s[pn][vn]
 for kind in ['c','lc','cp','lcp','lsc','l2c']:
  r,p,c,o=op(u,v,lambda x,k:features(x,kind,k[0]),[w0],[w0-.3],[w0+.3]);print(name,kind,r,p);print(c if kind in ('lc','lcp','lsc') else '')
  if kind=='lc':out[name]=o
u=s['panel_c']['u'];v=s['panel_c']['v1']
for fn in [np.arctan,np.tanh,np.sin]:
 for base in [False,True]:
  for sh in [False,True]:
   def f(x,k):
    z=fn(k[0]*x+(k[1:] if sh else 0)); return aff(np.c_[x,z] if base else z)
   r,p,c,o=op(u,v,f,[2.15]+[0]*4*sh,[.4]+[-1]*4*sh,[5]+[1]*4*sh);print('C',fn.__name__,base,sh,r,p);print(c if r<.001 else '')
   if fn==np.arctan and sh and not base:out['C']=o
Y=s['train']['y']; T=s['train']['traj']; _,_,vt=np.linalg.svd(T,full_matrices=False); Z=T@vt[:2].T
sets={'Q':q,**out}; sets.update({a+b:np.c_[sets[a],sets[b]] for a,b in itertools.combinations(sets,2)})
for name,x in sets.items():
 for kind in ['l','l2','l2p','l23p','ls','lsc','lt','lr']:
  a=features(x,kind,1.25)
  if a.shape[1]>=49:continue
  pred=[a[52:]@ls(a[:52],t[:52]) for t in (Y,Z)]; print(name,kind,*np.round([np.mean(abs(pred[0]-Y[52:])),np.mean(abs(pred[1]@vt[:2]-T[52:])),*np.mean(abs(pred[1]-Z[52:]),axis=0)],5))
