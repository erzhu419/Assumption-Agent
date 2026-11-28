import copy, itertools, pickle, runpy, subprocess, sys, re, json
import numpy as np
from pathlib import Path
root=Path('/holdout_task'); template=Path('/tmp/node00_next.py').read_text(); mod=runpy.run_path('/tmp/node00_next.py'); blocks=mod['_blocks']; kernel=mod['_kernel']; sink=runpy.run_path(str(root/'_frozen_nodes/node_01.py')); ridge=sink['_ridge']; src=(root/'_frozen_nodes/node_01.py').read_text(); penalties=list(map(float,re.findall(r'target,\s*(\.[0-9]+)\)',src))); print('sink penalties',penalties,flush=True)
s=pickle.load(open(root/'public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; traj=s['train']['traj']; n=len(y)
fold=[]
for v in [*np.array_split(np.random.default_rng(38).permutation(n),5),*np.array_split(np.random.default_rng(72).permutation(n),5),np.arange(n-20,n)]:
 t=np.setdiff1d(np.arange(n),v); basis=np.linalg.svd(traj[t],full_matrices=False)[2][:2]; T=np.c_[y[t],traj[t]@basis.T]; mu=T.mean(0); fold.append((t,v,basis,T-mu,mu))
def predict(k,t,v,T,mu,lam):
 a=k[np.ix_(t,t)]; b=k[np.ix_(v,t)]; m=a.mean(0); a=a-m[:,None]-m+m.mean(); b=b-b.mean(1)[:,None]-m+m.mean(); a.flat[::len(t)+1]+=lam
 return mu+b@np.linalg.solve(a,T)
def score(k,f,c):
 E=[]
 for t,v,basis,T,mu in fold:
  P=.5*predict(k,t,v,T,mu,penalties[0])+(predict(k+f,t,v,T,mu,penalties[1])+predict(k+c,t,v,T,mu,penalties[2])+predict(k+f+c,t,v,T,mu,penalties[3]))/6
  E.append([np.mean(abs(y[v]-P[:,0])),np.mean(abs(traj[v]-P[:,1:]@basis))])
 return np.array(E)
def write(conf):
 (root/'node_00.py').write_text(re.sub(r'^CONFIG = .*$', 'CONFIG = '+repr(conf), template, flags=re.M))
write(mod['CONFIG'])
assert len(penalties)==4,src
results=[]
for m in (0,1,2):
 Q=blocks(s,m); P=[X@X.T for X in Q]; bf,bc=P[1],P[2]+P[3]; J=[(1,2,3,4),(0,1,2,3,4),(1,4),(1,3,4),(1,4,5),(1,2,3,4,5),(1,4,6,7),(0,1,4,5,8),(4,),(4,5),(1,4,9),(4,5,8)]
 for ids in J:
  for kind,w in itertools.product(('rbf','blend','poly','matern'),(.045,.12,.28,.65,1.4)):
   K=kernel(Q,ids,kind,w,.7)
   for amp,lin,scale in ((.4,(0,1,2,3,4),1.),(1.2,(0,1,2,3,4),1.),(3.,(0,1,2,3,4),1.),(1.2,(0,1,4,5),1.),(1.2,(0,1,4,5),.6),(1.2,(0,1,2,3,4,5,8),1.)):
    e=score(amp*K+scale**2*sum(P[i] for i in lin),bf,bc); alt=e[:-1]; loss=alt.mean()+.4*alt.max()
    results.append((loss,tuple(e[-1]),tuple(alt.max(0)),(m,ids,kind,w,amp,.7,lin,scale)))
 print('completed basis',m,flush=True)
results.sort(); print('best alternate folds:',*results[:5],sep='\n',flush=True)
# Check the official evaluator directly; its split may differ from the diagnostics.
ranked=[r for r in results if max(r[1])<=.038]+results[:8]+sorted(results,key=lambda r:max(r[1]))[:8]; seen=set(); best=None
for loss,public_proxy,worst,conf in ranked[:24]:
 if conf in seen: continue
 seen.add(conf); write(conf)
 p=subprocess.run([sys.executable,'-I',str(root/'test_dag_candidate.py')],cwd=root,capture_output=True,text=True)
 try: out=json.loads(p.stdout.strip().splitlines()[-1]); val=max(out['cv_metrics'].values())
 except (ValueError,IndexError,KeyError): print(p.stdout,p.stderr,flush=True); continue
 print(conf, 'alternate worst',worst,'official',out,flush=True)
 rank=(not out['accepted'], loss if out['accepted'] else val, loss)
 if best is None or rank<best[0]: best=(rank,conf,worst)
 if out['accepted']: break
if best is not None: write(best[1]); print('selected:',best,flush=True)
