import pickle, runpy, numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
f=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py'); rng=np.random.default_rng(271)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(a,y):return np.linalg.lstsq(a,y,rcond=None)[0]
a=s['panel_a']; d=aff(a['u']); v=a['v']; c=ls(d,v)
for _ in range(20):
 r=np.linalg.norm(d@c-v,axis=1); w=1/np.hypot(1,r/max(np.median(r),1e-8)); c=ls(d*w[:,None],v*w[:,None])
r=np.linalg.norm(d@c-v,axis=1); print('A',np.sort(r)[-5:]); c=ls(d[np.argsort(r)[:-3]],v[np.argsort(r)[:-3]])
b=s['panel_b']; e=ls(aff(b['u']),b['v']); print('A coef',c,'B coef',e)
p=s['panel_c']; u=p['u']; fs={'cos':(lambda x,t:np.cos(t*x),(.4,5.)), 'rat':(lambda x,t:x/(1+t*x*x),(.05,15.)), 'tanh':(lambda x,t:np.tanh(t*x),(.2,8.)), 'sin':(lambda x,t:np.sin(t*x),(.2,6.)), 'atan':(lambda x,t:np.arctan(t*x),(.2,15.)), 'gauss':(lambda x,t:np.exp(-t*x*x),(.05,8.))}
def basis(x,g,t): return np.c_[aff(x),g(x,t)]
models={}
for k in ('v1','v2'):
 v=p[k]; rows=[]
 for n,(g,lim) in fs.items():
  def obj(t):
   a=basis(u,g,t); return np.linalg.norm(a@ls(a,v)-v)
  grid=np.linspace(*lim,101); j=np.argmin([obj(t) for t in grid]); res=minimize_scalar(obj,bounds=(grid[max(0,j-1)],grid[min(j+1,100)]),method='bounded'); t=res.x; a=basis(u,g,t); coef=ls(a,v); h=(np.linalg.qr(a)[0]**2).sum(1); loo=np.sqrt(np.mean(((a@coef-v)/(1-h[:,None]))**2)); rows.append((loo,n,t,coef,g,res.fun))
 rows.sort(key=lambda z:z[0]); print(k,[(round(z[0],7),z[1],round(z[2],6),round(z[-1],7)) for z in rows]); print('best coef',rows[0][3]); _,_,t,m,g,_=rows[0]; models[k]=lambda x,t=t,m=m,g=g:basis(x,g,t)@m
x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; banks={'x':x}
def fw(x,c):return aff(x)@c
def iv(x,c):return np.linalg.solve(c[1:].T,(x-c[0]).T).T
banks.update({'Ai':iv(x,c),'Af':fw(x,c)})
for n,q in list(banks.items()):
 banks[n+' Bf']=fw(q,e); banks[n+' Bi']=iv(q,e)
splits=[np.arange(len(x))]+[rng.permutation(len(x)) for _ in range(5)]
def score(fn,key,target,q):
 loss=[]
 for ix in splits:
  a=ix[:52]; b=ix[52:]; z={key:q[ix],'x':x[ix]}; loss.append(np.mean(np.abs(fn(z,target[a])-target[b])))
 return np.array(loss)
from scipy.optimize import least_squares
node=runpy.run_path('/holdout_task/node_00.py')
V=node['_readout'](p,'v2',node['_logcosh'],(1.1,.9),((.1,0),(4,5)))
G=lambda x: x+node['_logcosh'](1.1*x)/1.1
m=ls(aff(G(u)),p['v2']); print('v2 fixed-shape train RMSE',np.sqrt(np.mean((aff(G(u))@m-p['v2'])**2)))
# The shared warp is log((1+exp(2*k*x))/2)/k, so inversion is only defined above its lower asymptote.
def ung(w):
 t=1.1*w+np.log(2); return np.log(np.expm1(t))/2.2
out=[]
for n,q in banks.items():
 for name,w in [('decode',iv(q,m)),('inverse warp',q),('decode forward branch',iv(V(q),m))]:
  t=1.1*w+np.log(2); valid=(t>0).all(1); print('inverse range',n,name,valid.sum(),len(q))
  if not valid.all() or name=='decode forward branch':continue
  r=ung(w)
  for tag,z in [('raw',r),('Bf',fw(r,e)),('Bi',iv(r,e))]:
   l=score(f['_scalar'],'i2b',y,z);out.append((l.max(),l[0],l[1:].mean(),n,name,tag))
print('inverse handoffs',sorted(out)[:10]);
for kind,op,lim in [('asinh',np.arcsinh,(.01,15)),('erf',__import__('scipy').special.erf,(.05,8)),('softsign',lambda x:x/(1+np.abs(x)),(.05,15)),('atan',np.arctan,(.05,15))]:
 def fit(U,T):
  def residual(t):
   A=aff(U+t[1]*op(t[0]*U));return (A@ls(A,T)-T).ravel()
  r=min((least_squares(residual,s0,bounds=([lim[0],-30],[lim[1],30])) for s0 in [(1,1),(3,7),(6,-2)]),key=lambda r:r.cost)
  A=aff(U+r.x[1]*op(r.x[0]*U));c=ls(A,T);return r,lambda W:aff(W+r.x[1]*op(r.x[0]*W))@c
 r,_=fit(u,p['v1']); err=[]
 for i in range(len(u)):
  keep=np.arange(len(u))!=i; _,F=fit(u[keep],p['v1'][keep]);err.extend((F(u[i:i+1])-p['v1'][i]).ravel())
 print('v1',kind,r.x,'nested loo',np.sqrt(np.mean(np.square(err))))
