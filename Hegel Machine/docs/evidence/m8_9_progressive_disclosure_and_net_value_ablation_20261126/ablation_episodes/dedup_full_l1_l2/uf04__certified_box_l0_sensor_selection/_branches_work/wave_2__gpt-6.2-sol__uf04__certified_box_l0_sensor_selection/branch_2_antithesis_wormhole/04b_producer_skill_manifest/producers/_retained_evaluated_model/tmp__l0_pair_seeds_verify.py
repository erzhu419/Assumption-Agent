import importlib.util, time
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/certified_l0_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
D=np.repeat(np.eye(8),2,axis=1)[:,:15]
worst=(0.,0); failures=[]
for seed in (2,7,19,81):
 for coupling in (.003,.03,.12):
  for ridge in (.0001,.03):
   X=D+coupling*np.random.default_rng(seed).normal(size=D.shape)
   a=dict(X=X,y=np.full(8,1.2),ridge=ridge,box=1.5,l0_penalty=.3)
   try:
    ans=m.solve_certified_box_l0(a); w=np.array(ans['weights']); e=X@w-a['y']; v=.5*(e@e+ridge*(w@w))+.3*np.count_nonzero(abs(w)>1e-9); gap=v-ans['certified_lower_bound']; nodes=ans['nodes_explored']
    assert np.all(abs(w)<=1.5) and abs(v-ans['objective'])<1e-10 and 0<=gap<=1e-5 and 1<=nodes<=600
    worst=(max(worst[0],gap),max(worst[1],nodes))
   except (RuntimeError,AssertionError) as e:
    failures.append((seed,coupling,ridge,str(e)))
print('Worst verified gap and nodes:',worst)
print('Failures:',len(failures),failures)
assert not failures, failures
