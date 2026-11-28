import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v,axis=1)<5; u,v=u[keep],v[keep];g=np.roll(u,2,axis=1)
FUN={'tan':np.tanh,'sin':np.sin,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'atan':np.arctan,'cos':np.cos,'exp':lambda x:np.exp(-x*x),'cube':lambda x:x**3}
for k,h in FUN.items():
 for extra in [False,True]:
  def prof(p,ret=False):
   ff=h(p[0]*u); gg=np.sin(p[1]*g); F=[np.c_[np.ones(len(u)),u[:,j],ff[:,j],gg[:,j],*( [u[:,j]*g[:,j]] if extra else [])] for j in range(4)]; b=np.array([ls(x,v[:,j]) for j,x in enumerate(F)]);r=np.column_stack([x@b[j]-v[:,j] for j,x in enumerate(F)]);return (r,b) if ret else r.ravel()
  r=least_squares(prof,[.8,2.4],bounds=([.09,1],[5,4])); print('A',k,extra,np.std(r.fun),r.x,prof(r.x,True)[1][0])
u,v=[s['panel_c'][k] for k in ('u','v1')]; print('C',u.shape, 'orthogonal latent?',np.linalg.svd(v-v.mean(0),compute_uv=False)); one=np.ones(len(u)); modes={'u':u, **{f'd{i}':u-np.roll(u,i,axis=1) for i in (1,2,3)}, **{f's{i}':u+np.roll(u,i,axis=1) for i in (1,2,3)}, **{f'p{i}':u*np.roll(u,i,axis=1) for i in (1,2,3)}}; F0=np.c_[one,modes['d1']**2]; res=v-F0@ls(F0,v)
for deg in range(1,5):
 ids=[()] + [q for d in range(1,deg+1) for q in combinations_with_replacement(range(4),d)]; f=np.column_stack([np.prod(u[:,q],axis=1) for q in ids]); b=ls(f,v); print('Cp',deg,np.std(f@b-v),[(q,round(np.linalg.norm(c),3),np.round(c,2)) for q,c in zip(ids,b) if np.linalg.norm(c)>.12][:23])
Funs={**FUN,'sq':lambda x:x*x,'id':lambda x:x,'sqrt':lambda x:np.sqrt(1+x*x),'inv':lambda x:1/(1+x*x),'abs':np.abs}; opts=[]
for k,h in Funs.items():
 for m,U in modes.items():
  for hz in (.5,1,1.8,2.4,3):
   F=np.c_[F0,h(hz*U)];opts.append((np.std(F@ls(F,v)-v),k,m,hz))
print('C additive span',sorted(opts)[:22]);opts=[]
for k,h in Funs.items():
 for m,U in modes.items():
  for l,V in modes.items():
   def proj(p):
    F=np.c_[one,h(p[0]*U+p[1]*V)];return (F@ls(F,v)-v).ravel()
   r=least_squares(proj,[1.4,.35],bounds=([.05,-4],[4,4]),max_nfev=24);opts.append((np.std(r.fun),k,m,l,r.x.tolist()))
print('C mixed inputs',sorted(opts)[:22]);
