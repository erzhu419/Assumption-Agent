import importlib.util
import pickle
import time
import numpy as np
spec = importlib.util.spec_from_file_location('thermal', '/holdout_task/thermal_sink_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_thermal_cases.pkl','rb') as f:
    cases=pickle.load(f)
for i, case in enumerate(cases):
    t=time.perf_counter(); ans=mod.localize_thermal_source(case['measurement']); sec=time.perf_counter()-t
    gt=case['gt']; err=(np.linalg.norm(np.array(ans['source_xy'])-gt['source_xy']), abs(ans['amplitude']/gt['amplitude']-1), abs(ans['junction_offset']-gt['junction_offset']))
    print(i,ans,'errors',err,'seconds',sec)
    assert set(ans)=={'source_xy','amplitude','junction_offset'} and type(ans['amplitude']) is float and type(ans['junction_offset']) is float and all(type(a) is float for a in ans['source_xy'])
    assert np.all(np.array(err)<=[.035,.08,.008]) and sec<15
