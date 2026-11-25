import importlib.util,pickle,time
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver', p+'certified_l0_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def case(X,y,ridge=.35,box=1.5,l0_penalty=.85): return dict(X=X,y=y,ridge=ridge,box=box,l0_penalty=l0_penalty)
def evaluate(label,a,expected=None):
 t=time.monotonic();r=m.solve_certified_box_l0(a);w=np.asarray(r['weights']);e=a['X']@w-a['y'];obj=.5*(e@e+a['ridge']*(w@w))+a['l0_penalty']*sum(abs(w)>1e-9); lb=r['certified_lower_bound'];gap=obj-lb;nodes=r['nodes_explored'];dt=time.monotonic()-t
 assert np.max(np.abs(w),initial=0)<=a['box'] and abs(r['objective']-obj)<1e-10 and 0<=nodes<=600
 if expected is not None: assert lb<=expected+1e-11 and obj>=expected-1e-11,(label,lb,expected,obj)
 print(label,nodes,'%.4g'%gap,'%.3fs'%dt,'FAIL' if gap>1e-5 else '',flush=True)
for k,a in enumerate(pickle.load(open(p+'public_box_l0_instances.pkl','rb'))): evaluate(f'public-{k}',a)
pdim=15
for target in [np.sqrt(2*.85*1.35),.1,5.]:
 for box in [5e-10,1.5]:
  X=np.eye(pdim);y=np.full(pdim,target); z=min(target/1.35,box); b=min(z,1e-9)
  cost=lambda w:.5*((w-target)**2+.35*w*w)+.85*(abs(w)>1e-9)
  evaluate(f'orthogonal-{target:.3g}-{box}',case(X,y,box=box),pdim*min(cost(z),cost(b)))
for perturb in [0.,.002,.03,.2,1.]:
 for k in range(3):
  rng=np.random.default_rng(k); u=rng.normal(size=(32,5)); X=np.repeat(u,3,axis=1)+perturb*rng.normal(size=(32,15)); X/=np.linalg.norm(X,axis=0);y=X[:,::3]@rng.choice([-1.,1.],5)*1.35+.03*rng.normal(size=32)
  evaluate(f'correlated-{perturb}-{k}',case(X,y,.015,1.5,.37))
for k in range(16):
 rng=np.random.default_rng(k); X=rng.normal(size=(25,15)); X/=np.linalg.norm(X,axis=0); y=rng.normal(size=25)
 evaluate(f'dense-{k}',case(X,y,.035,.6,.13))
