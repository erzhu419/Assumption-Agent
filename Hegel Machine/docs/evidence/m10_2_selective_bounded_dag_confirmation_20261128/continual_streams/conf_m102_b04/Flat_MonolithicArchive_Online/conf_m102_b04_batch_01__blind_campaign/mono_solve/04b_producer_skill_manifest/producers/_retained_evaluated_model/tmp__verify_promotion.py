import sys
sys.path.insert(0, '/tmp')
import finalcv as f
import importlib.util
import numpy as np
sp = list(f.sp)
rng = np.random.default_rng(49271)
for _ in range(32):
    p = rng.permutation(72)
    sp.append((p[:52],p[52:]))
new = f.run(sp)
spec = importlib.util.spec_from_file_location('baseline', '/tmp/batch_solver_v01.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
new_solver = f.m
f.m = base
old = f.run(sp)
for label, e in [('baseline',old),('challenger',new)]:
    print(label, '52/20 scalar, trajectory, final horizon:', e[0,[0,1,9]])
    for label2, sl in [('development stress',slice(1,33)),('fresh stress',slice(33,None))]:
        print(label2,'mean', e[sl][:,[0,1,9]].mean(0),'worst',e[sl][:,[0,1,9]].max(0))
    print('worst by horizon', e[1:,2:10].max(0))
assert np.isfinite(new).all()
keep = (new[0,0] <= .038 and (new[0,:2] <= old[0,:2]).all()
        and (new[1:,:2].mean(0) <= old[1:,:2].mean(0) + .001).all()
        and (new[1:,:2].max(0) <= old[1:,:2].max(0) + .001).all()
        and new[1:,9].max() <= old[1:,9].max() + .001)
if not keep:
    from shutil import copyfile
    copyfile('/tmp/batch_solver_v01.py','/holdout_task/batch_solver.py')
print('selected', 'challenger' if keep else 'baseline')
