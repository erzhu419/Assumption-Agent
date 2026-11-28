import numpy as np,pickle,runpy
from scipy.optimize import least_squares
from pathlib import Path
path=Path('/holdout_task/node_00.py'); m=runpy.run_path(str(path)); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u,v=p['u'],p['v2']; n=len(u); tr,te=np.split(np.random.default_rng(53).permutation(n),[18]); ls=m['_ols']; soft=m['_soft']; rms=lambda z:np.mean(z*z)**.5
fun={'soft':soft,'tanh':np.tanh,'rat':lambda z:z/(1+z*z),'sin':np.sin,'rat_even':lambda z:1/(1+z*z),'abs':np.abs,'square':np.square,'signed_sq':lambda z:z*abs(z),'cube':lambda z:z**3}
def design(x,p,ori,kind,mode):
 a,b=p; y=np.roll(x,-ori,axis=1); f=fun[kind]; z=f(a*y+b)
 if mode==0:return np.c_[np.ones(len(x)),x,z,x*y]
 if mode==1:return np.c_[np.ones(len(x)),x,z,x*z]
 if mode==2:return np.c_[np.ones(len(x)),x,z,x*y,x*z]
 if mode==3:return np.c_[np.ones(len(x)),x,z,x*y/(1+a*y*y)]
 if mode==4:return np.c_[np.ones(len(x)),x,x*y+b*z]
 if mode==5:return np.c_[np.ones(len(x)),x,x*z+b*f(a*y)]
 if mode==6:return np.c_[np.ones(len(x)),x,z,x*y/(1+a*(x-y)**2)]
 if mode==7:return np.c_[np.ones(len(x)),x,z,x*y+b*x*z]
res=[]
for ori in (1,-1,2):
 for kind in fun:
  for mode in range(8):
   def obj(q):
    d=design(u[tr],q,ori,kind,mode); return (d@ls(d,v[tr])-v[tr]).ravel()
   fit=min((least_squares(obj,q,bounds=([.025,-3],[7,3]),max_nfev=90) for q in [(.35,0),(1.25,.1),(2.15,-.5)]),key=lambda f:f.cost); d=design(u,fit.x,ori,kind,mode); res.append((rms(d[te]@ls(d[tr],v[tr])-v[te]),rms(fit.fun),ori,kind,mode,fit.x))
for x in sorted(res,key=lambda q:q[0])[:24]:print(x)
# Evaluate the best panel-derived candidates with archived readouts, and record configurations for follow-up.
z=m['_calibration'](s['panel_a'])(s['train']['x']); c=m['_branches'](p)[0](z); train=s['train']; scope=m['_features'].__globals__; frozen=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')
print('FEATURES'); print(path.read_text().split('def _features')[1]); print('TEST SPLIT'); print('\n'.join(l for l in Path('/holdout_task/test_dag_candidate.py').read_text().splitlines() if 'hold' in l or 'perm' in l or 'split' in l or 'idx' in l)); print('FROZEN'); print(Path('/holdout_task/_frozen_nodes/node_01.py').read_text());
best=sorted(res,key=lambda q:q[0])[0]; _,_,ori,kind,mode,q=best
prefix=Path('/tmp/design9.py'); prefix.write_text('import numpy as np\n'+Path(__file__).read_text().split('fun=')[1].split('res=[]')[0].join(['fun=','']).replace("'soft':soft","'soft':lambda z:z/(1+abs(z))"))
old=path.read_text(); Path('/tmp/baseline9.py').write_text(old); body=prefix.read_text().split('\n',1)[1]; body += f'\ndef _branch2(u, v):\n    def design2(x, p):\n        return design(x, p, {ori!r}, {kind!r}, {mode!r})\n    def residual(p):\n        d = design2(u, p)\n        return (d @ _ols(d,v)-v).ravel()\n    q = least_squares(residual, {q.tolist()!r}, bounds=([.025,-3],[7,3])).x\n    coef = _ols(design2(u,q),v)\n    return lambda x: design2(x,q) @ coef\n\n'; old=old.replace('def _branches',body+'def _branches'); old=old.replace("fit(panel['v2'], lambda z: 1.0 / (1.0 + z*z), 0.4)","_branch2(u, panel['v2'])"); path.write_text(old)
