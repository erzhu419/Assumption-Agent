import importlib.util, pickle, time
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/certified_l0_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_box_l0_instances.pkl','rb') as f: data=pickle.load(f)
for i,a in enumerate(data):
 t=time.monotonic(); r=m.solve_certified_box_l0(a); print('public',i,r['objective'],r['objective']-r['certified_lower_bound'],r['nodes_explored'],round(time.monotonic()-t,4),flush=True)
