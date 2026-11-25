import importlib.util,sys,time,traceback
import numpy as np
v, mode = sys.argv[1:3]
s=importlib.util.spec_from_file_location('solver',v); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
D=np.repeat(np.eye(8),2,axis=1)[:,:15]
for c in (.03,.12):
 if str(c) != mode: continue
 for seed in (2,7,19,42,81):
  X=D+c*np.random.default_rng(seed).normal(size=D.shape)
  for r in (.0001,.03):
   st=time.monotonic(); a=dict(X=X,y=np.full(8,1.2),ridge=r,box=1.5,l0_penalty=.3)
   try:
    ans=m.solve_certified_box_l0(a); print(seed,c,r,ans['nodes_explored'],ans['objective']-ans['certified_lower_bound'],round(time.monotonic()-st,2),flush=True)
   except RuntimeError as e: print(seed,c,r,e,round(time.monotonic()-st,2),flush=True)
