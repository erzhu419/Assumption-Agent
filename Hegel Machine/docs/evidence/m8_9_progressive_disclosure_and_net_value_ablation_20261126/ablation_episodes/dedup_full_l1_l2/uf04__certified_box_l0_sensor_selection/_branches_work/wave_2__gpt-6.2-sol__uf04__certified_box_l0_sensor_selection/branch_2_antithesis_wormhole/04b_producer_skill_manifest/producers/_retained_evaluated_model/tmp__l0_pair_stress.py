import importlib.util, time
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/certified_l0_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
D=np.repeat(np.eye(8),2,axis=1)[:,:15]
for coupling in (0., 1e-5, .003, .03, .12):
 for r in (.0001,.03,.35):
  X=D+coupling*np.random.default_rng(42).normal(size=D.shape)
  a=dict(X=X,y=np.full(8,1.2),ridge=r,box=1.5,l0_penalty=.3)
  st=time.monotonic()
  try:
   ans=m.solve_certified_box_l0(a); gap=ans['objective']-ans['certified_lower_bound']
   if not coupling:
    ref=8*(.3+.5*1.2**2*r/(1+r))
    assert ans['certified_lower_bound']<=ref+1e-10 and abs(ans['objective']-ref)<=1e-7
   print(coupling,r,ans['nodes_explored'],gap,round(time.monotonic()-st,3),flush=True)
  except RuntimeError as e: print(coupling,r,str(e),round(time.monotonic()-st,3),flush=True)
