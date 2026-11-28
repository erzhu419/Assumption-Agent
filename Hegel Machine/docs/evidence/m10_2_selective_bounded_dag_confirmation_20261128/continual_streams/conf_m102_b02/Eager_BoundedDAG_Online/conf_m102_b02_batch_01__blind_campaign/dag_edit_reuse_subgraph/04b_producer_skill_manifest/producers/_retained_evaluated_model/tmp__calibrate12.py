import itertools, pickle, runpy, shutil
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; base='/tmp/baseline11.py'; x=runpy.run_path(base)['_latent'](s); n=len(s['train']['x']); kernel=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['_kernel']; y=s['train']['y']; traj=s['train']['traj']; folds=[]
for j in [np.arange(52,n),*np.array_split(np.random.default_rng(42).permutation(n),4),*np.array_split(np.random.default_rng(550).permutation(n),4)]:
 i=np.setdiff1d(np.arange(n),j); v=np.linalg.svd(traj[i],full_matrices=False)[2][:2]; folds.append((i,j,v,np.c_[y[i],traj[i]@v.T]))
def evaluate(z):
 try:
  g=kernel(z[:n],z[:n]); errors=[]
  for i,j,v,t in folds:
   a=g[np.ix_(i,i)].copy(); a.flat[::len(i)+1]+=1e-6*np.diag(a).mean(); p=g[np.ix_(j,i)]@solve(a,t,assume_a='pos',check_finite=False); errors.append([np.mean(abs(p[:,0]-y[j])),np.mean(abs(p[:,1:]@v-traj[j]))])
  return np.asarray(errors)
 except (ValueError,np.linalg.LinAlgError): return np.full((len(folds),2),np.inf)
exec(open('/tmp/maps12.py').read()); candidate='/tmp/node12.py'; start=evaluate(.5*x); print('BASE',start.round(6).tolist(),flush=True)
def quality(r):
 if not np.isfinite(r).all(): return np.inf
 if max(r[0])<=.038: return -1+max(r.max(0))
 return max(r.max(0))+.35*max(r[0])+.20*r.mean()
best=(quality(start),None,start)
settings=[('', '',k,a,'raw') for k,aa in [('identity',[1]),('cube',[.15,.5,1,2,4]),('asinh',[1,2,3.5,6,10]),('tanh',[.6,1.25,2,3]),('power',[.35,.55,.75,1.5,2,2.5,3]),('softcube',[1,2,4]),('sinres',[.7,1.5,2.5])] for a in aa]
for pk,vk,k,aa in [('panel_b','v','cube',[.25,.55,.8,1.25,2]),('panel_b','v','softcube',[1,2]),('panel_c','v1','asinh',[2,3.5,5]),('panel_c','v2','square',[.165,.3]),('panel_b','v','identity',[1]),('panel_c','v1','identity',[1]),('panel_c','v2','identity',[1])]:
 for a in aa:
  for mode in ['single','unmix','polar','inverse','nonlinear','full','blend']:
   if (mode in ('inverse','unmix') and k=='square') or (mode in ('nonlinear','full','blend') and k=='identity'): continue
   settings.append((pk,vk,k,a,mode))
results=[]
def check(spec):
 global best
 z=_coordinates(x,s,spec,n)
 if not np.isfinite(z).all(): return
 r=evaluate(z); q=quality(r); results.append((q,spec))
 if q<best[0] and (max(r[0])<=.038 or (np.all(r[0]<=start[0]) and np.all(r.max(0)<=start.max(0)))):
  best=(q,spec,r); print('IMPROVED',spec,r[0].round(6),r.max(0).round(6),flush=True)
for prefix in settings:
 for center,post,b,sc in itertools.product([0,1],['none','axis'],[0],[.07,.14,.28,.50,.8,1.2]): check((*prefix,center,post,b,sc))
for _,spec in sorted(results,key=lambda t:t[0])[:65]:
 prefix=spec[:5]; center=spec[5]; scale=spec[-1]
 for post,b in [('add',.15),('add',.35),('add',.7),('resid',-.5),('resid',-.2),('resid',.25),('resid',.5),('compress',1.25),('compress',3.5)]:
  for f in (.65,1,1.5): check((*prefix,center,post,b,scale*f))
_,spec,r=best
if spec is not None:
 body=open(base).read().split('def run_node')[0]+open('/tmp/maps12.py').read()+f"\n\ndef run_node(inputs: dict) -> dict:\n    study=inputs['__input__']; n=len(study['train']['x']); x=_latent(study)\n    z=_coordinates(x,study,{spec!r},n)\n    return {{'z':z[:n].copy(),'zt':z[n:].copy()}}\n"
 open(candidate,'w').write(body); runpy.run_path(candidate); shutil.copyfile(candidate,'/holdout_task/node_00.py')
print('FINAL SLICES',r.round(6).tolist(),flush=True)
