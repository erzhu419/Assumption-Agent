import importlib.util, pickle, time
import numpy as np
spec=importlib.util.spec_from_file_location('thermal_sink_solver','/holdout_task/thermal_sink_solver.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
with open('/holdout_task/public_thermal_cases.pkl','rb') as f: cases=pickle.load(f)
for i,c in enumerate(cases):
    t=time.perf_counter(); p=s.localize_thermal_source(c['measurement']);elapsed=time.perf_counter()-t
    gt=c['gt'];e=[np.linalg.norm(np.subtract(p['source_xy'],gt['source_xy'])),abs(p['amplitude']/gt['amplitude']-1),abs(p['junction_offset']-gt['junction_offset'])];print(i,p,e,elapsed)
    assert set(p)=={'source_xy','amplitude','junction_offset'} and type(p['source_xy']) is list and len(p['source_xy'])==2 and all(type(v) is float for v in [*p['source_xy'],p['amplitude'],p['junction_offset']])
    assert np.all(np.array(e)<=[.035,.08,.008]) and elapsed<15
