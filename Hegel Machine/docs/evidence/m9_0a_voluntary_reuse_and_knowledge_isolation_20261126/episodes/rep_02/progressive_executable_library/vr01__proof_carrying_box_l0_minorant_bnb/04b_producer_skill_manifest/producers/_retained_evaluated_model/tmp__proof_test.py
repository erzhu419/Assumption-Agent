import importlib.util, pickle, time
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/proof_box_l0.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for i,c in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb'))):
 t=time.monotonic(); r=m.solve_proof_carrying_box_l0(c['problem']); print(i,'error',r['objective']-c['gt']['exact_objective'],'gap',r['objective']-r['certified_lower_bound'],'leaves',len(r['subcube_leaves']),'sec',time.monotonic()-t)
