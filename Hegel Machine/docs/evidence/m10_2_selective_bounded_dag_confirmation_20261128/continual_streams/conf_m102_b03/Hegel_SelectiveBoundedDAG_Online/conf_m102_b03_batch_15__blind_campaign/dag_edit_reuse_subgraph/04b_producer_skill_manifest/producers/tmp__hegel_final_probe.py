import pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/node_00.py'); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; one=m['_ones']; ls=m['_ls']; t=s['train']; n=len(t['x']); rows=np.arange(n); hs=[rows[-20:]]+list(np.array_split(np.random.default_rng(123).permutation(rows),4)); pc=s['panel_c']; u=pc['u']; v=pc['v2']; pf=np.array_split(np.random.default_rng(23).permutation(len(u)),4)
def adj(x):return np.roll(x,-1,axis=1)
def nbr(x):return x*adj(x)
def design(x,kind,k):
 p=nbr(x); w=adj(x)
 if kind=='neighbors': h=p
 elif kind=='nbsq': h=np.c_[p,x*x]
 elif kind=='nbsat': h=np.c_[p,x*x/(1+k*x*x)]
 elif kind=='nbcos': h=np.c_[p,np.cos(k*x)]
 elif kind=='sincross': h=np.sin(k*x)*np.sin(k*w)
 elif kind=='ratself': h=p/(1+k*x*x)
 elif kind=='ratnext': h=p/(1+k*w*w)
 elif kind=='ratprod': h=p/(1+k*abs(p))
 elif kind=='nbthird': h=np.c_[p,p*np.roll(x,-2,axis=1)]
 elif kind=='nbsum': h=np.c_[p,np.cos(k*(x+w))]
 elif kind=='nbdiff': h=np.c_[p,np.cos(k*(x-w))]
 return one(np.c_[x,h])
def fit(a,b,kind):
 def obj(k):
  A=design(a,kind,k); return np.mean((A@ls(A,b)-b)**2)
 if kind in ('neighbors','nbsq','nbthird'): k=1.
 else:
  grid=np.geomspace(.015,7,42); j=np.argmin([obj(k) for k in grid]); k=minimize_scalar(obj,bounds=(grid[max(j-1,0)],grid[min(j+1,len(grid)-1)]),method='bounded').x
 A=design(a,kind,k); w=ls(A,b); return k,w
feat=m['run_node']({'__input__':s}); z=feat['z'][:n]; q=feat['Q'][:n]; base={k:w[:n] for k,w in feat.items()}; kinds='neighbors nbsq nbsat nbcos sincross ratself ratnext ratprod nbthird nbsum nbdiff'.split(); stats=[]; rs={}
for kind in kinds:
 k,w=fit(u,v,kind); rs[kind]=design(z,kind,k)@w; errs=[]
 for h in pf:
  a=np.setdiff1d(np.arange(len(u)),h); j,b=fit(u[a],v[a],kind); errs.append(np.mean(abs(design(u[h],kind,j)@b-v[h])))
 stats.append((np.mean(errs),max(errs),np.mean(abs(design(u,kind,k)@w-v)),kind,k))
print('PANEL',sorted(stats),flush=True)
rs={k:rs[k] for k in [r[3] for r in sorted(stats)[:4]]}; rs['baseline']=base['R']
oldp=nbr(z); oldc=np.cos(2.4*z)
def score(r,prod,cos):
 f=dict(base,R=r,product=prod,cosine=cos); ret=[]
 for h in hs:
  a=np.setdiff1d(rows,h); ix=np.r_[a,h]; st=dict(s,train={k:t[k][a] for k in ('x','y','traj')},test={'x':t['x'][h]}); o=sink({'__input__':st,'node_00':{k:w[ix] for k,w in f.items()}}); ret.append([np.mean(abs(o['y_hat']-t['y'][h])),np.mean(abs(o['state_pred']-t['traj'][h]))])
 return np.round(ret,6)
res=[]
for kind,r in rs.items():
 for pn,p in [('q',q),('q+nbr',np.c_[q,oldp]),('nbr',oldp)]:
  for cn,c in [('r',r),('r+nbr',np.c_[r,oldp]),('r+cos',np.c_[r,oldc]),('nbr',oldp),('cos',oldc)]:
   e=score(r,p,c); item=(e.max(),e.mean(),kind,pn,cn,e.tolist()); res.append(item)
   if e[0].max()<=.038: print('PASS',item,flush=True)
print('WORST RANK',sorted(res)[:12],flush=True)
