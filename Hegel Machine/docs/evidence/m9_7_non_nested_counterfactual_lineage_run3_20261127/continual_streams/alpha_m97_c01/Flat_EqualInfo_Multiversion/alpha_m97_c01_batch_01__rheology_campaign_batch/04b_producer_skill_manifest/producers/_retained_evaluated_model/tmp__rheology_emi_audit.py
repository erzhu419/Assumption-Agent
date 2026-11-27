import importlib.util, itertools, pickle
from unittest.mock import patch
import numpy as np
root = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(root+'public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; failures = []
for ch in range(4):
    lab = c['lab_id']==ch; x=c['cmd_stress'][lab]; y=c['true_stress'][lab]
    full=m._line(x,y); grid=np.vstack((x,np.ones(len(x))))
    for n in (3,4,5,10):
        worst = recover = failures_n = count = 0
        for ids in itertools.combinations(range(len(x)),n):
            xx=x[list(ids)]; yy=y[list(ids)]
            with patch.object(m, '_archived_calibration', side_effect=AssertionError('archive consulted on a clean sweep')):
                np.testing.assert_allclose(m._calibrate(xx,yy,ch),m._line(xx,yy),atol=1e-12,rtol=0)
            for i in range(n):
                clean=np.arange(n)!=i; expected=m._line(xx[clean],yy[clean])
                for spike in (-8.,-3.01,3.01,8.):
                    zz=yy.copy(); zz[i]+=spike; actual=m._calibrate(xx,zz,ch)
                    e=np.max(abs((actual-expected)@grid)); recover=max(recover,e)
                    worst=max(worst,np.max(abs((actual-full)@grid)))
                    failures_n+=int(e>1e-10); count+=1
        print(f'lab={ch} rows={n} trials={count} wrong_pair={failures_n} vs_clean={recover:.6g} vs_full={worst:.6g}',flush=True)
        if failures_n: failures.append((ch,n,failures_n))
    np.testing.assert_allclose(m._calibrate(x[:1],y[:1],ch),m._archived_calibration(ch),atol=1e-12,rtol=0)
assert not failures, failures
